
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Set, Tuple
import logging

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
    lk_window_size: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    lk_term_count: int = 30
    lk_term_epsilon: float = 0.01
    lk_min_eig_threshold: float = 1e-4
    feature_quality_threshold: float = 0.4
    min_track_distance: float = 0.0
    batch_size: int = 8
    performance_mode: str = "Balanced"
    thread_priority: str = "Normal"
    cache_enabled: bool = True


@dataclass
class TrackingJobState:
    job_id: int
    point_name: str
    span: Tuple[int, int]
    progress: float = 0.0
    status: str = "Pending"
    message: str = ""


@dataclass
class ActiveJob:
    config: TrackingJobConfig
    worker: "TrackingWorker"
    state: TrackingJobState


class TrackingWorker(QtCore.QThread):
    progress_signal = QtCore.pyqtSignal(int, float)
    result_signal = QtCore.pyqtSignal(int, object)
    failed_signal = QtCore.pyqtSignal(int, str)

    def __init__(
        self,
        config: TrackingJobConfig,
        tracker: CustomPointTracker,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._tracker = tracker
        self._log = logging.getLogger(__name__)

    def run(self) -> None:  # pragma: no cover - runs in background thread
        span_start, span_end = self._config.span
        if span_end <= span_start:
            self.failed_signal.emit(self._config.job_id, "Invalid tracking span.")
            return

        loader = self._config.frame_loader or self._tracker.get_frame_loader()
        if loader is None:
            self.failed_signal.emit(self._config.job_id, "No frame loader configured for tracking.")
            return

        frames: Dict[int, FrameSample] = {}
        total_frames = max(1, span_end - span_start)
        preload_weight = max(0.0, min(1.0, self._config.preload_weight))
        compute_weight = 1.0 - preload_weight

        self._log.debug(
            "TrackingWorker[%s]: start point=%s span=%s preload=%.2f", 
            self._config.job_id,
            self._config.point_name,
            self._config.span,
            preload_weight,
        )

        try:
            for index, frame_index in enumerate(range(span_start, span_end + 1)):
                if self.isInterruptionRequested():
                    return
                sample = self._tracker.ensure_frame_sample(frame_index)
                if sample is None:
                    frame_bgr = loader(frame_index)
                    if frame_bgr is None:
                        self._log.debug(
                            "TrackingWorker[%s]: missing frame %s", self._config.job_id, frame_index
                        )
                        self.failed_signal.emit(
                            self._config.job_id,
                            f"Unable to load frame {frame_index} for tracking.",
                        )
                        return
                    sample = FrameSample(frame_bgr)
                    if self._config.cache_enabled:
                        cached = self._tracker.cache_frame_sample(frame_index, frame_bgr)
                        if cached is not None:
                            sample = cached
                frames[frame_index] = sample
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
                lk_window_size=self._config.lk_window_size,
                lk_max_level=self._config.lk_max_level,
                lk_term_count=self._config.lk_term_count,
                lk_term_epsilon=self._config.lk_term_epsilon,
                lk_min_eig_threshold=self._config.lk_min_eig_threshold,
                feature_quality_threshold=self._config.feature_quality_threshold,
                min_track_distance=self._config.min_track_distance,
                batch_size=self._config.batch_size,
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
            self._log.debug("TrackingWorker[%s]: interrupted", self._config.job_id)
            return
        except Exception as exc:  # pragma: no cover - defensive
            self._log.exception("TrackingWorker[%s]: error", self._config.job_id)
            self.failed_signal.emit(self._config.job_id, str(exc))
            return

        if result is None:
            self._log.debug("TrackingWorker[%s]: result missing", self._config.job_id)
            self.failed_signal.emit(self._config.job_id, "Tracking failed to produce a result.")
            return

        self.progress_signal.emit(self._config.job_id, 1.0)
        self.result_signal.emit(self._config.job_id, result)
        self._log.debug("TrackingWorker[%s]: finished", self._config.job_id)


class TrackingManager(QtCore.QObject):
    overall_progress = QtCore.pyqtSignal(float)
    tracking_started = QtCore.pyqtSignal()
    tracking_finished = QtCore.pyqtSignal()
    segment_ready = QtCore.pyqtSignal(str, tuple)
    segment_failed = QtCore.pyqtSignal(str, tuple, str)
    job_registered = QtCore.pyqtSignal(object)
    job_updated = QtCore.pyqtSignal(object)
    job_removed = QtCore.pyqtSignal(int)

    def __init__(self, tracker: CustomPointTracker, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._tracker = tracker
        self._log = logging.getLogger(__name__)
        self._jobs: Dict[int, ActiveJob] = {}
        self._job_states: Dict[int, TrackingJobState] = {}
        self._paused_jobs: Dict[int, TrackingJobConfig] = {}
        self._span_to_job: Dict[Tuple[str, Tuple[int, int]], int] = {}
        self._next_job_id: int = 1
        self._shutting_down: bool = False
        self._active_workers: Dict[int, TrackingWorker] = {}

    def request_point(self, point_name: Optional[str]) -> None:
        if not self._tracker.tracking_enabled:
            self._log.debug("TrackingManager: tracking disabled, skipping request for %s", point_name)
            return
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
            if any(cfg.point_name == point_name and cfg.span == span for cfg in self._paused_jobs.values()):
                continue
            job_id = self._next_job_id
            config = self._build_job_config(job_id, point_name, span)
            if config is None:
                continue
            self._next_job_id += 1
            self._log.debug("TrackingManager: queue point=%s span=%s job=%s", point_name, span, job_id)
            self._start_job(config)

    def request_all(self) -> None:
        if not self._tracker.tracking_enabled:
            self._log.debug("TrackingManager: tracking disabled, skipping request_all")
            return
        for point_name in self._tracker.point_definitions().keys():
            self.request_point(point_name)

    def reset(self) -> None:
        self._log.debug("TrackingManager: reset requested (active=%s)", len(self._jobs))
        running = list(self._jobs.values())
        self._shutting_down = True
        for job in running:
            job.worker.requestInterruption()
        for job in running:
            if not job.worker.wait(5000):
                self._log.debug("TrackingManager: terminating slow job=%s", job.config.job_id)
                job.worker.terminate()
                job.worker.wait()
            job.worker.deleteLater()
            self._active_workers.pop(job.config.job_id, None)
        self._jobs.clear()
        self._span_to_job.clear()
        self._paused_jobs.clear()
        removed_ids = list(self._job_states.keys())
        self._job_states.clear()
        for job_id in removed_ids:
            self.job_removed.emit(job_id)
        self._next_job_id = 1
        self._shutting_down = False
        self.overall_progress.emit(0.0)
        if running:
            self.tracking_finished.emit()

    def shutdown(self) -> None:
        self.reset()

    def pause_job(self, job_id: int) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job.worker.requestInterruption()
        job.worker.wait(1000)
        self._jobs.pop(job_id, None)
        self._span_to_job.pop((job.config.point_name, job.config.span), None)
        self._paused_jobs[job_id] = job.config
        self._active_workers.pop(job_id, None)
        job.state.status = "Paused"
        self.job_updated.emit(replace(job.state))
        self._emit_overall_progress()
        self._log.debug("TrackingManager: paused job=%s", job_id)
        return True

    def resume_job(self, job_id: int) -> bool:
        config = self._paused_jobs.pop(job_id, None)
        if config is None:
            return False
        state = self._job_states.get(job_id)
        if state is not None:
            state.progress = 0.0
            state.message = ""
        self._start_job(config)
        self._log.debug("TrackingManager: resumed job=%s", job_id)
        return True

    def cancel_job(self, job_id: int) -> bool:
        config = self._paused_jobs.pop(job_id, None)
        job = self._jobs.pop(job_id, None)
        if job is None and config is None:
            return False
        if job is not None:
            job.worker.requestInterruption()
            job.worker.wait(1000)
            self._span_to_job.pop((job.config.point_name, job.config.span), None)
            state = job.state
            self._active_workers.pop(job_id, None)
        else:
            state = self._job_states.get(job_id)
        if state is not None:
            state.status = "Cancelled"
            state.progress = 0.0
            state.message = ""
            self.job_updated.emit(replace(state))
        self._emit_overall_progress()
        self._log.debug("TrackingManager: cancelled job=%s", job_id)
        return True

    def job_states(self) -> Dict[int, TrackingJobState]:
        return {job_id: replace(state) for job_id, state in self._job_states.items()}

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
        if tracked_point.is_occluded(start_frame) or tracked_point.is_occluded(end_frame):
            return None
        start_pos = tracked_point.keyframes.get(start_frame)
        end_pos = tracked_point.keyframes.get(end_frame)
        if start_pos is None or end_pos is None:
            return None
        loader = self._tracker.get_frame_loader()
        if loader is None:
            return None
        window = int(self._tracker.lk_window_size)
        if window % 2 == 0:
            window += 1
        performance = (self._tracker.performance_mode or "Balanced").lower()
        preload = 0.18 if performance == "high" else 0.25
        if not self._tracker.cache_frames_enabled:
            preload = min(preload, 0.2)
        return TrackingJobConfig(
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
            preload_weight=preload,
            lk_window_size=(window, window),
            lk_max_level=self._tracker.lk_max_level,
            lk_term_count=self._tracker.lk_term_count,
            lk_term_epsilon=self._tracker.lk_term_epsilon,
            lk_min_eig_threshold=self._tracker.lk_min_eig_threshold,
            feature_quality_threshold=self._tracker.lk_feature_quality,
            min_track_distance=self._tracker.lk_min_distance,
            batch_size=self._tracker.lk_batch_size,
            performance_mode=self._tracker.performance_mode,
            thread_priority=self._tracker.thread_priority,
            cache_enabled=self._tracker.cache_frames_enabled,
        )

    def _start_job(self, config: TrackingJobConfig) -> None:
        if not self._tracker.tracking_enabled:
            self._log.debug(
                "TrackingManager: tracking disabled, refusing to start job=%s",
                config.job_id,
            )
            return
        state = self._job_states.get(config.job_id)
        is_new = state is None
        if state is None:
            state = TrackingJobState(job_id=config.job_id, point_name=config.point_name, span=config.span)
        state.progress = 0.0
        state.status = "Running"
        state.message = ""
        self._job_states[config.job_id] = state

        worker = TrackingWorker(config, self._tracker)
        job = ActiveJob(config=config, worker=worker, state=state)
        self._jobs[config.job_id] = job
        self._span_to_job[(config.point_name, config.span)] = config.job_id
        self._active_workers[config.job_id] = worker

        worker.progress_signal.connect(self._on_worker_progress)
        worker.result_signal.connect(self._on_worker_result)
        worker.failed_signal.connect(self._on_worker_failed)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(lambda job_id=config.job_id: self._on_worker_thread_finished(job_id))

        if len(self._jobs) == 1:
            self.tracking_started.emit()
            self.overall_progress.emit(0.0)

        worker.start()
        priority = self._priority_value(config)
        if priority is not None:
            worker.setPriority(priority)
        (self.job_registered if is_new else self.job_updated).emit(replace(state))
        self._log.debug(
            "TrackingManager: started job=%s point=%s span=%s active_jobs=%s",
            config.job_id,
            config.point_name,
            config.span,
            len(self._jobs),
        )

    def _on_worker_thread_finished(self, job_id: int) -> None:
        worker = self._active_workers.pop(job_id, None)
        if worker:
            self._log.debug("TrackingManager: worker finished job=%s", job_id)

    def _priority_value(self, config: TrackingJobConfig) -> Optional[QtCore.QThread.Priority]:
        mode = (config.performance_mode or "").lower()
        if mode == "high":
            return QtCore.QThread.HighestPriority
        mapping = {
            "lowest": QtCore.QThread.LowestPriority,
            "low": QtCore.QThread.LowPriority,
            "normal": QtCore.QThread.NormalPriority,
            "high": QtCore.QThread.HighPriority,
            "timecritical": QtCore.QThread.TimeCriticalPriority,
        }
        key = (config.thread_priority or "Normal").replace(" ", "").lower()
        return mapping.get(key, QtCore.QThread.NormalPriority)

    @QtCore.pyqtSlot(int, float)
    def _on_worker_progress(self, job_id: int, progress: float) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.state.progress = max(0.0, min(1.0, progress))
        if job.state.status != "Running":
            job.state.status = "Running"
        self.job_updated.emit(replace(job.state))
        self._emit_overall_progress()

    @QtCore.pyqtSlot(int, object)
    def _on_worker_result(self, job_id: int, payload: object) -> None:
        if self._shutting_down:
            return
        job = self._jobs.get(job_id)
        if not job:
            return
        result: Optional[SegmentBuildResult] = payload if isinstance(payload, SegmentBuildResult) else None
        if result is not None:
            self._tracker.apply_segment_result(job.config.point_name, job.config.span, result)
            self.segment_ready.emit(job.config.point_name, job.config.span)
            self._finalise_job(job_id, status="Completed", progress=1.0)
        else:
            self.segment_failed.emit(job.config.point_name, job.config.span, "Tracking produced no result.")
            self._finalise_job(job_id, status="Failed")
        self._log.debug(
            "TrackingManager: job %s result=%s", job_id, "ok" if result is not None else "empty"
        )

    @QtCore.pyqtSlot(int, str)
    def _on_worker_failed(self, job_id: int, message: str) -> None:
        if self._shutting_down:
            return
        state = self._job_states.get(job_id)
        if state is not None:
            self.segment_failed.emit(state.point_name, state.span, message)
        self._finalise_job(job_id, status="Failed", message=message)
        self._log.debug("TrackingManager: job %s failed %s", job_id, message)

    def _finalise_job(self, job_id: int, *, status: str, message: Optional[str] = None, progress: Optional[float] = None) -> None:
        job = self._jobs.pop(job_id, None)
        if job is not None:
            self._span_to_job.pop((job.config.point_name, job.config.span), None)
        state = self._job_states.get(job_id)
        if state is None:
            return
        if progress is not None:
            state.progress = max(0.0, min(1.0, progress))
        state.status = status
        state.message = message or ""

        if status in {"Completed", "Failed", "Cancelled"}:
            self._job_states.pop(job_id, None)
            self.job_removed.emit(job_id)
        else:
            self.job_updated.emit(replace(state))

        worker = self._active_workers.pop(job_id, None)
        if worker is not None:
            if not worker.wait(5000):
                self._log.debug("TrackingManager: finalise terminate job=%s", job_id)
                worker.terminate()
                worker.wait()

        self._emit_overall_progress()
        if not self._jobs and not self._shutting_down:
            self.tracking_finished.emit()
        self._log.debug(
            "TrackingManager: finalised job=%s status=%s remaining=%s",
            job_id,
            status,
            len(self._jobs),
        )

    def _emit_overall_progress(self) -> None:
        if self._jobs:
            average = sum(job.state.progress for job in self._jobs.values()) / len(self._jobs)
            self.overall_progress.emit(max(0.0, min(1.0, average)))
            return
        if not self._job_states:
            self.overall_progress.emit(0.0)
            return
        if all(state.status == "Completed" for state in self._job_states.values()):
            self.overall_progress.emit(1.0)
        else:
            self.overall_progress.emit(0.0)
