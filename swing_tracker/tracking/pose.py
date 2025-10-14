from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np

from ..models import LandmarkData, TrackIssue


@dataclass
class PoseFrameResult:
    landmarks: Dict[int, LandmarkData]
    issues: List[TrackIssue]
    resolved_points: List[str]


class PoseTracker:
    DETECTION_ISSUE_NAME = "Pose - Detection"

    def __init__(
        self,
        visibility_threshold: float = 0.5,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 2,
    ) -> None:
        self.visibility_threshold = visibility_threshold
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._connections: Sequence[Tuple[int, int]] = tuple(mp.solutions.pose.POSE_CONNECTIONS)

    @property
    def connections(self) -> Sequence[Tuple[int, int]]:
        return self._connections

    def process(self, frame_bgr: np.ndarray, frame_index: int) -> PoseFrameResult:
        height, width = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb_frame)

        landmarks: Dict[int, LandmarkData] = {}
        issues: List[TrackIssue] = []
        resolved: Set[str] = set()

        if results.pose_landmarks:
            resolved.add(self.DETECTION_ISSUE_NAME)
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                position = (float(landmark.x * width), float(landmark.y * height))
                confidence = float(landmark.visibility)
                landmarks[idx] = LandmarkData(position, confidence)
                landmark_name = mp.solutions.pose.PoseLandmark(idx).name
                point_name = f"Pose - {landmark_name}"
                if confidence < self.visibility_threshold:
                    issues.append(
                        TrackIssue(
                            frame_index=frame_index,
                            point_name=point_name,
                            confidence=confidence,
                            note="Pose landmark visibility below threshold.",
                        )
                    )
                else:
                    resolved.add(point_name)
        else:
            issues.append(
                TrackIssue(
                    frame_index=frame_index,
                    point_name=self.DETECTION_ISSUE_NAME,
                    confidence=0.0,
                    note="Pose detection failed; no landmarks returned.",
                )
            )

        return PoseFrameResult(landmarks=landmarks, issues=issues, resolved_points=sorted(resolved))

    def close(self) -> None:
        self._pose.close()
