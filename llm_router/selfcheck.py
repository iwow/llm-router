"""L3 后置自检：启发式自洽性 + 结构化校验 + 失败信号。

用于在生成后判断"当前档是否够用"，若不够则触发升级重试（反馈回路）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class SelfCheckResult:
    ok: bool
    needs_upgrade: bool
    reason: str
    confidence: float


# 低置信短语：命中说明模型自己也没把握
LOW_CONFIDENCE_PHRASES = [
    "我不确定", "不太确定", "我不清楚", "无法确定", "难以判断", "可能错误",
    "i'm not sure", "i don't know", "i cannot", "抱歉", "无法回答",
]


class SelfChecker:
    def check(self, query: str, response: str, tier: str) -> SelfCheckResult:
        resp = (response or "").strip()

        # 1. 空响应 / 异常短路
        if not resp:
            return SelfCheckResult(False, True, "empty response", 0.0)

        # 2. 低置信短语
        low = resp.lower()
        for p in LOW_CONFIDENCE_PHRASES:
            if p in low:
                return SelfCheckResult(False, True, f"low confidence: {p}", 0.2)

        # 3. 结构化输出要求未满足（要求 json 但返回非 json）
        q = query.lower()
        if "json" in q:
            if not self._looks_like_json(resp):
                return SelfCheckResult(False, True, "json required but not returned", 0.3)

        # 4. 疑似截断（结尾无句读、明显未收尾）
        if len(resp) > 200 and not re.search(r"[。.!?？\n]$", resp):
            return SelfCheckResult(False, True, "possibly truncated", 0.4)

        return SelfCheckResult(True, False, "", 0.9)

    @staticmethod
    def _looks_like_json(s: str) -> bool:
        s = s.strip().strip("`")
        try:
            json.loads(s)
            return True
        except (ValueError, TypeError):
            # 允许 json 内嵌在文本中（如 ```json ... ```）
            return bool(re.search(r"\{[\s\S]*\}", s))
