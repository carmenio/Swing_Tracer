from typing import Dict, Iterable, List, Optional, Tuple

from .models import TrackIssue


class IssueLog:
    def __init__(self) -> None:
        self._issues: Dict[Tuple[int, str], TrackIssue] = {}

    def record_issue(self, issue: TrackIssue) -> bool:
        key = (issue.frame_index, issue.point_name)
        existing = self._issues.get(key)
        if existing:
            updated = False
            if existing.confidence != issue.confidence:
                existing.confidence = issue.confidence
                updated = True
            if existing.note != issue.note:
                existing.note = issue.note
                updated = True
            if existing.hidden:
                existing.hidden = False
                updated = True
            return updated
        self._issues[key] = issue
        return True

    def remove_issue(self, frame_index: int, point_name: str) -> Optional[TrackIssue]:
        key = (frame_index, point_name)
        return self._issues.pop(key, None)

    def clear_from_frame(self, frame_index: int) -> bool:
        removal_keys = [key for key in self._issues if key[0] >= frame_index]
        if not removal_keys:
            return False
        for key in removal_keys:
            self._issues.pop(key, None)
        return True

    def clear_for_point(self, point_name: str) -> bool:
        removal_keys = [key for key in self._issues if key[1] == point_name]
        if not removal_keys:
            return False
        for key in removal_keys:
            self._issues.pop(key, None)
        return True

    def remove_point_issues_after(self, point_name: str, frame_index: int) -> bool:
        removal_keys = [key for key in self._issues if key[1] == point_name and key[0] > frame_index]
        if not removal_keys:
            return False
        for key in removal_keys:
            self._issues.pop(key, None)
        return True

    def clear(self) -> None:
        self._issues.clear()

    def issues(self) -> Iterable[TrackIssue]:
        return self._issues.values()

    def find(self, frame_index: int, point_name: str) -> Optional[TrackIssue]:
        return self._issues.get((frame_index, point_name))

    def set_hidden(self, frame_index: int, point_name: str, hidden: bool) -> bool:
        issue = self.find(frame_index, point_name)
        if not issue:
            return False
        if issue.hidden == hidden:
            return False
        issue.hidden = hidden
        return True

    def issues_by_filter(self, mode: str) -> List[TrackIssue]:
        if mode == "Hidden":
            filtered = [issue for issue in self._issues.values() if issue.hidden]
        elif mode == "All":
            filtered = list(self._issues.values())
        else:
            filtered = [issue for issue in self._issues.values() if not issue.hidden]
        filtered.sort(key=lambda issue: (issue.frame_index, issue.point_name))
        return filtered
