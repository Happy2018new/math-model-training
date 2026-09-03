"""计算问题四候选任务的时间冲突集合和角度冲突集合。"""

from __future__ import annotations

from collections.abc import Iterable

from .define import TaskCandidate


MIN_PHOTO_ANGLE_DEG = 60.0
ConflictPair = tuple[str, str]


def _pair_id(first: TaskCandidate, second: TaskCandidate) -> ConflictPair:
    """返回顺序固定的一对冲突候选编号。"""
    first_id, second_id = sorted((first.candidate_id, second.candidate_id))
    return first_id, second_id


def _flatten(
    candidates_by_target: dict[str, list[TaskCandidate]],
) -> list[TaskCandidate]:
    """将按目标编号分组的候选任务展平为一个列表。"""
    return [
        candidate
        for candidates in candidates_by_target.values()
        for candidate in candidates
    ]


def time_conflict(first: TaskCandidate, second: TaskCandidate) -> bool:
    """判断两个准备区间是否重叠。

    准备区间使用半开区间 ``[准备开始时刻, 执行时刻)``。因此，一个任务
    恰好结束时，另一个任务可以恰好从该时刻开始准备。
    """
    return (
        first.prepare_start < second.execute_time
        and second.prepare_start < first.execute_time
    )


def circular_angle_difference(first_angle: float, second_angle: float) -> float:
    """返回两个方向角之间较小的圆周夹角，单位为度。"""
    difference = abs(first_angle - second_angle)
    return min(difference, 360.0 - difference)


def angle_conflict(first: TaskCandidate, second: TaskCandidate) -> bool:
    """判断同一拍照目标的两个候选是否违反最小 60 度视角差要求。"""
    if first.task_type != "photo" or second.task_type != "photo":
        return False
    if first.task_id != second.task_id:
        return False
    if first.angle_deg is None or second.angle_deg is None:
        raise ValueError("拍照候选必须包含拍照方向角")
    return circular_angle_difference(first.angle_deg, second.angle_deg) < MIN_PHOTO_ANGLE_DEG


def build_time_conflicts(
    candidates: Iterable[TaskCandidate],
) -> set[ConflictPair]:
    """计算所有准备区间重叠的候选任务对。

    先按准备开始时刻排序。若后续候选的准备开始时刻已不早于当前候选的
    执行时刻，则它和再后面的候选都不可能与当前候选发生时间冲突。
    """
    ordered = sorted(candidates, key=lambda candidate: candidate.prepare_start)
    conflicts: set[ConflictPair] = set()

    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if second.prepare_start >= first.execute_time:
                break
            if time_conflict(first, second):
                conflicts.add(_pair_id(first, second))
    return conflicts


def build_angle_conflicts(
    photo_candidates: dict[str, list[TaskCandidate]],
) -> set[ConflictPair]:
    """计算同一拍照目标中方向角差小于 60 度的候选任务对。"""
    conflicts: set[ConflictPair] = set()
    for candidates in photo_candidates.values():
        for index, first in enumerate(candidates):
            for second in candidates[index + 1 :]:
                if angle_conflict(first, second):
                    conflicts.add(_pair_id(first, second))
    return conflicts


def calculate_conflict_sets(
    shoot_candidates: dict[str, list[TaskCandidate]],
    photo_candidates: dict[str, list[TaskCandidate]],
) -> tuple[
    dict[str, TaskCandidate], set[ConflictPair], set[ConflictPair]
]:
    """返回按编号索引的候选、时间冲突对和角度冲突对。"""
    all_candidates = _flatten(shoot_candidates) + _flatten(photo_candidates)
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in all_candidates
    }
    if len(candidate_by_id) != len(all_candidates):
        raise ValueError("候选任务编号必须唯一")

    return (
        candidate_by_id,
        build_time_conflicts(all_candidates),
        build_angle_conflicts(photo_candidates),
    )
