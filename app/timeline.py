"""时间轴构建：校验输入、全局对齐、并为新增字符在空档内均匀分配时间。

新增字符按相邻匹配锚点划分连续段：一次顺序遍历即可确定每段的左右边界
（左锚点 end 或录音起点 0；右锚点 start 或录音终点 total_duration），
随后在段内逐字赋时（地板法均分），整体 O(n)。
"""
from __future__ import annotations

from .alignment import align_chars
from .errors import InvalidTimelineError
from .schemas import AlignRequest, RevisedCharTimeline


def validate_request(req: AlignRequest, max_recognized: int,
                     max_revised: int) -> None:
    """校验输入参数与时间轴；无效时抛出 InvalidTimelineError（明确中文信息）。"""
    if req.total_duration < 0:
        raise InvalidTimelineError(
            f"录音总时长 total_duration 必须为非负整数，当前为 {req.total_duration}")

    if len(req.recognized) > max_recognized:
        raise InvalidTimelineError(
            f"识别字符数量 {len(req.recognized)} 超过上限 {max_recognized}，"
            "可通过环境变量 MAX_RECOGNIZED_CHARS 调整")
    if len(req.revised_text) > max_revised:
        raise InvalidTimelineError(
            f"修订文本长度 {len(req.revised_text)} 超过上限 {max_revised}，"
            "可通过环境变量 MAX_REVISED_LENGTH 调整")

    for idx, item in enumerate(req.recognized):
        if len(item.char) != 1:
            raise InvalidTimelineError(
                f"第 {idx} 项识别字符必须且只能包含 1 个字符，当前为 {item.char!r}")
        if item.start < 0 or item.end < 0:
            raise InvalidTimelineError(
                f"第 {idx} 项字符 {item.char!r} 的起止时间不能为负数："
                f"start={item.start}, end={item.end}")
        if item.end < item.start:
            raise InvalidTimelineError(
                f"第 {idx} 项字符 {item.char!r} 的结束时间 {item.end} 不能早于"
                f"起始时间 {item.start}")
        if item.start > req.total_duration or item.end > req.total_duration:
            raise InvalidTimelineError(
                f"第 {idx} 项字符 {item.char!r} 的时间 [{item.start}, {item.end}] "
                f"超出录音总时长 {req.total_duration}")
        if idx > 0:
            prev = req.recognized[idx - 1]
            if item.start < prev.start:
                raise InvalidTimelineError(
                    f"识别字符必须按起始时间排序：第 {idx - 1} 项 start={prev.start}，"
                    f"第 {idx} 项 start={item.start}")
            if item.start < prev.end:
                raise InvalidTimelineError(
                    f"识别字符时间区间不得重叠：第 {idx - 1} 项 "
                    f"[{prev.start},{prev.end}] 与第 {idx} 项 "
                    f"[{item.start},{item.end}] 重叠")

    # 录音时长为 0 时不允许出现带正时长的字符（边界 0..0 仍合法）。
    if req.total_duration == 0 and any(
            (it.start != 0 or it.end != 0) for it in req.recognized):
        raise InvalidTimelineError(
            "录音总时长为 0 时，所有识别字符时间必须均为 0")


def _fill_run(chars: list[RevisedCharTimeline], run_start: int,
              run_end: int, gap_left: int, gap_right: int) -> None:
    """为一个连续新增段 target[run_start:run_end] 单次赋时。

    在整数区间 [gap_left, gap_right] 内按段内字数均匀切分（端点固定、
    地板法取整），逐字写入 start/end。空档宽度为 0 时各字时间相同。
    """
    count = run_end - run_start
    if count <= 0:
        return
    if gap_right == gap_left:
        for j in range(run_start, run_end):
            chars[j].start = gap_left
            chars[j].end = gap_right
        return
    span = gap_right - gap_left
    for k in range(count):
        chars[run_start + k].start = gap_left + span * k // count
        chars[run_start + k].end = gap_left + span * (k + 1) // count


def build_timeline(req: AlignRequest, backend: str) -> dict:
    """执行对齐并生成逐字时间轴。"""
    source = "".join(item.char for item in req.recognized)
    target = req.revised_text
    total = req.total_duration

    mapping = align_chars(source, target)

    # 锚点：对齐到识别字符（完全匹配或替换纠错）的修订字符。
    # 按修订下标升序保存 (修订下标, 识别下标)，供一次顺序遍历划分新增段。
    anchors: list[tuple[int, int]] = [
        (j, src_idx) for j, src_idx in enumerate(mapping) if src_idx >= 0
    ]

    matched_count = len(anchors)
    inserted_count = len(target) - matched_count
    deleted_count = len(source) - matched_count

    # 先为每个修订位置占一个 computed 槽，锚点随后覆盖、新增段由 _fill_run 赋时。
    chars: list[RevisedCharTimeline] = [
        RevisedCharTimeline(char=ch, start=0, end=0, source="computed")
        for ch in target
    ]
    for j, src_idx in anchors:
        item = req.recognized[src_idx]
        chars[j] = RevisedCharTimeline(
            char=target[j],
            start=item.start,
            end=item.end,
            source="matched",
            matched_index=src_idx,
            recognized_char=item.char,
        )

    # 按相邻锚点划分连续新增段，每段只计算一次边界并逐字赋时。
    prev_pos = -1
    prev_src_end = 0  # 无前锚点时左边界为录音起点 0
    for j, src_idx in anchors:
        if j > prev_pos + 1:
            # 段位于前锚点（或录音起点）与当前锚点之间。
            _fill_run(chars, prev_pos + 1, j, prev_src_end,
                      req.recognized[src_idx].start)
        prev_pos = j
        prev_src_end = req.recognized[src_idx].end
    if prev_pos + 1 < len(target):
        # 末段位于最后一个锚点（或无锚点时的录音起点）与录音终点之间。
        _fill_run(chars, prev_pos + 1, len(target), prev_src_end, total)

    return {
        "total_duration": total,
        "chars": chars,
        "inserted_count": inserted_count,
        "deleted_count": deleted_count,
        "matched_count": matched_count,
        "backend": backend,
    }
