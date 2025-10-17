from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

from PyQt5 import QtCore

from ..entities import Point2D
from .custom_points import CustomPointTracker
from .segments import FrameSample, SegmentBuilder, SegmentBuildResult


@dataclass(frozen=True)
class TrackingJobConfig:
    job_id: int
    point_name: str
    span: Tuple[int, int]
    start_pos: Point2D
    end_pos: Point2D
    accepted_frames: Set[int]
    rejected_frames: Set[int]
    baseline_mode: str
    threshold_confidence: float
    min_threshold_pixels: float
    threshold_scale: float
    nms_window: int
    min_segment_len: int
    max_provisionals: int
    min_improvement_px: float
    min_improvement_ratio: float
    smoothing_window: int
    entity_colour: Optional[str]
    frame_loader: Optional[Callable[[int], Optional[object]]]
    preload_weight: float = 0.25


@dataclass
class ActiveJob:
    config: TrackingJobConfig
    worker: "TrackingWorker"
    progress: float = 0.0


class TrackingWorker(QtCore.QThread):
    progress_signal = QtCore.pyqtSignal(int, float)
    result_signal = QtCore.pyqtSignal(int, object)
    failed_signal = QtCore.pyqtSignal(int, str)

    def __init__(self, config: TrackingJobConfig, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:  # pragma: no cover - runs in background thread
        loader = self._config.frame_loader
        if loader is None:
            self.failed_signal.emit(self._config.job_id, "No frame loader configured for tracking.")
            return

        span_start, span_end = self._config.span
        if span_end <= span_start:
            self.failed_signal.emit(self._config.job_id, "Invalid tracking span.")
            return

        frames: Dict[int, FrameSample] = {}
        total_frames = max(1, span_end - span_start)
        preload_weight = max(0.0, min(1.0, self._config.preload_weight))
        compute_weight = 1.0 - preload_weight

        try:
            for index, frame_index in enumerate(range(span_start, span_end + 1)):
                if self.isInterruptionRequested():
                    return
                frame_bgr = loader(frame_index)
                if frame_bgr is None:
                    self.failed_signal.emit(
                        self._config.job_id,
                        f"Unable to load frame {frame_index} for tracking.",
                    )
                    return
                frames[frame_index] = FrameSample(frame_bgr)
                if preload_weight > 0.0:
                    preload_progress = preload_weight * ((index + 1) / (total_frames + 1))
                    self.progress_signal.emit(self._config.job_id, preload_progress)

            builder = SegmentBuilder(
                baseline_mode=self._config.baseline_mode,
                threshold_confidence=self._config.threshold_confidence,
                min_threshold_pixels=self._config.min_threshold_pixels,
                threshold_scale=self._config.threshold_scale,
                nms_window=self._config.nms_window,
                min_segment_len=self._config.min_segment_len,
                max_provisionals=self._config.max_provisionals,
                min_improvement_px=self._config.min_improvement_px,
                min_improvement_ratio=self._config.min_improvement_ratio,
                smoothing_window=self._config.smoothing_window,
                entity_colour=self._config.entity_colour,
            )

            def on_progress(fraction: float) -> None:
                clamped = max(0.0, min(1.0, fraction))
                combined = preload_weight + compute_weight * clamped
                self.progress_signal.emit(self._config.job_id, combined)
                if self.isInterruptionRequested():
                    raise InterruptedError

            result = builder.build(
                span_start=span_start,
                span_end=span_end,
                start_pos=self._config.start_pos,
                end_pos=self._config.end_pos,
                frames=frames,
                accepted_frames=self._config.accepted_frames,
                rejected_frames=self._config.rejected_frames,
                progress_callback=on_progress if compute_weight > 0.0 else None,
            )
        except InterruptedError:
            return
        except Exception as exc:  # pragma: no cover - defensive
            self.failed_signal.emit(self._config.job_id, str(exc))
            return

        if result is None:
            self.failed_signal.emit(self._config.job_id, "Tracking failed to produce a result.")
            return

        self.progress_signal.emit(self._config.job_id, 1.0)
        self.result_signal.emit(self._config.job_id, result)


class TrackingManager(QtCore.QObject):
    overall_progress = QtCore.pyqtSignal(float)
    tracking_started = QtCore.pyqtSignal()
    tracking_finished = QtCore.pyqtSignal()
    segment_ready = QtCore.pyqtSignal(str, tuple)
    segment_failed = QtCore.pyqtSignal(str, tuple, str)

    def __init__(self, tracker: CustomPointTracker, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._tracker = tracker
        self._jobs: Dict[int, ActiveJob] = {}
        self._span_to_job: Dict[Tuple[str, Tuple[int, int]], int] = {}
        self._next_job_id: int = 1
        self._shutting_down: bool = False

    def request_point(self, point_name: Optional[str]) -> None:
        if not point_name:
            return
        spans = self._tracker.span_pairs(point_name)
        if not spans:
            return

        for span in spans:
            if not self._tracker.requires_tracking(point_name, span):
                continue
            key = (point_name, span)
            if key in self._span_to_job:
                continue
            job_id = self._next_job_id
            config = self._build_job_config(job_id, point_name, span)
            if config is None:
                continue
            self._next_job_id += 1
            self._start_job(config)

    def request_all(self) -> None:
        for point_name in self._tracker.point_definitions().keys():
            self.request_point(point_name)

    def reset(self) -> None:
        was_active = bool(self._jobs)
        self._shutting_down = True
        for job in list(self._jobs.values()):
            job.worker.requestInterruption()
        for job in list(self._jobs.values()):
            job.worker.wait(1000)
        self._jobs.clear()
        self._span_to_job.clear()
        self._shutting_down = False
        self.overall_progress.emit(0.0)
        if was_active:
            self.tracking_finished.emit()

    def shutdown(self) -> None:
        self.reset()

    def _build_job_config(
        self,
        job_id: int,
        point_name: str,
        span: Tuple[int, int],
    ) -> Optional[TrackingJobConfig]:
        tracked_point = self._tracker.point_definitions().get(point_name)
        if not tracked_point:
            return None

        start_frame, end_frame = span
        start_pos = tracked_point.keyframes.get(start_frame)
        end_pos = tracked_point.keyframes.get(end_frame)
        if start_pos is None or end_pos is None:
            return None

        loader = self._tracker.get_frame_loader()
        if loader is None:
            return None

        config = TrackingJobConfig(
            job_id=job_id,
            point_name=point_name,
            span=span,
            start_pos=start_pos,
            end_pos=end_pos,
            accepted_frames=set(tracked_point.accepted_keyframe_frames()),
            rejected_frames=self._tracker.rejected_frames_for_span(point_name, span),
            baseline_mode=self._tracker.interpolation_mode,
            threshold_confidence=self._tracker.threshold_confidence,
            min_threshold_pixels=self._tracker.min_threshold_pixels,
            threshold_scale=self._tracker.threshold_scale,
            nms_window=self._tracker.nms_window,
            min_segment_len=self._tracker.min_segment_len,
            max_provisionals=self._tracker.max_provisionals_per_span,
            min_improvement_px=self._tracker.min_improvement_px,
            min_improvement_ratio=self._tracker.min_improvement_ratio,
            smoothing_window=self._tracker.smoothing_window,
            entity_colour=self._tracker.point_colour_hex(point_name),
            frame_loader=loader,
        )
        return config

    def _start_job(self, config: TrackingJobConfig) -> None:
        worker = TrackingWorker(config)
        job = ActiveJob(config=config, worker=worker)
        self._jobs[config.job_id] = job
        self._span_to_job[(config.point_name, config.span)] = config.job_id

        worker.progress_signal.connect(self._on_worker_progress)
        worker.result_signal.connect(self._on_worker_result)
        worker.failed_signal.connect(self._on_worker_failed)
        worker.finished.connect(worker.deleteLater)

        if len(self._jobs) == 1:
            self.tracking_started.emit()
            self.overall_progress.emit(0.0)

        worker.start()

    @QtCore.pyqtSlot(int, float)
    def _on_worker_progress(self, job_id: int, progress: float) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.progress = max(0.0, min(1.0, progress))
        self._emit_overall_progress()

    @QtCore.pyqtSlot(int, object)
    def _on_worker_result(self, job_id: int, payload: object) -> None:
        if self._shutting_down:
            return
        job = self._jobs.get(job_id)
        if not job:
            return
        result: Optional[SegmentBuildResult] = None
        if isinstance(payload, SegmentBuildResult):
            result = payload
        self._tracker.apply_segment_result(job.config.point_name, job.config.span, result)
        self._finalise_job(job_id, success=result is not None, message=None)

    @QtCore.pyqtSlot(int, str)
    def _on_worker_failed(self, job_id: int, message: str) -> None:
        if self._shutting_down:
            return
        self._finalise_job(job_id, success=False, message=message)

    def _finalise_job(self, job_id: int, *, success: bool, message: Optional[str]) -> None:
        job = self._jobs.pop(job_id, None)
        if not job:
            return
        self._span_to_job.pop((job.config.point_name, job.config.span), None)

        if success:
            self.segment_ready.emit(job.config.point_name, job.config.span)
        else:
            self.segment_failed.emit(job.config.point_name, job.config.span, message or "Tracking failed.")

        if self._jobs:
            self._emit_overall_progress()
        else:
            self.overall_progress.emit(1.0 if success else 0.0)
            self.tracking_finished.emit()

    def _emit_overall_progress(self) -> None:
        if not self._jobs:
            return
        average = sum(job.progress for job in self._jobs.values()) / len(self._jobs)
        self.overall_progress.emit(max(0.0, min(1.0, average)))
