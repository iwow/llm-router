"""路由器：主路由 + 滞回状态机 + 熔断器 + LRU 缓存 + 执行流程。

阈值与策略（与设计文档一致）：
- 主路由（新会话）：D<40 → hy3；40≤D<70 → flash；D≥70 → pro。
- 滞回：升级单向快速（目标档高于当前立即升）；
        降级保守（连续 N 次目标档更低 + 冷却期才降）。
- 熔断：单档近窗口错误率 > 阈值则临时摘除，路由到相邻档。
- L3 自检失败 → 升级重试（最高到 pro）。
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Optional, List

from .scorer import DifficultyScorer, DifficultyResult
from .registry import TIERS, TIER_INDEX, TIER_CONFIGS, LLMClient, MockClient
from .selfcheck import SelfChecker, SelfCheckResult


# ---------------- 阈值常量 ----------------
T_LOW = 40            # hy3 上界（主路由）
T_UPGRADE = 70        # 升级阈值
T_DOWNGRADE = 55      # 降级阈值
DOWNGRADE_CONSECUTIVE = 3
DOWNGRADE_COOLDOWN_S = 600.0
BREAKER_WINDOW_S = 300.0
BREAKER_ERROR_RATE = 0.3
BREAKER_COOLDOWN_S = 300.0


@dataclass
class RouteDecision:
    tier: str
    score: float
    confidence: float
    reason: str
    breakdown: dict


@dataclass
class ExecutionResult:
    tier: str                 # 最终实际使用的档
    response: str
    decision: RouteDecision
    selfcheck: Optional[SelfCheckResult]
    escalated: bool
    used_fallback: bool
    cache_hit: bool


def primary_route(score: float) -> str:
    if score < T_LOW:
        return "hy3"
    if score < T_UPGRADE:
        return "flash"
    return "pro"


class CircuitBreaker:
    """滑动窗口熔断器：错误率超阈值则 open，冷却期后 half-open 放行。"""

    def __init__(self, window_s: float = BREAKER_WINDOW_S,
                 error_rate: float = BREAKER_ERROR_RATE,
                 cooldown_s: float = BREAKER_COOLDOWN_S):
        self.window_s = window_s
        self.error_rate = error_rate
        self.cooldown_s = cooldown_s
        self._events: deque = deque()          # (ts, ok)
        self._opened_at: float = 0.0

    def record(self, ok: bool) -> None:
        now = time.time()
        self._events.append((now, ok))
        while self._events and now - self._events[0][0] > self.window_s:
            self._events.popleft()

    def is_open(self) -> bool:
        now = time.time()
        if not self._events:
            return False
        # 冷却期内的 open 状态
        if self._opened_at and now - self._opened_at < self.cooldown_s:
            return True
        errs = sum(1 for _, ok in self._events if not ok)
        total = len(self._events)
        if total >= 3 and errs / total > self.error_rate:
            self._opened_at = now
            return True
        self._opened_at = 0.0
        return False

    def reset(self) -> None:
        self._events.clear()
        self._opened_at = 0.0


class Router:
    def __init__(self, client: LLMClient, scorer: Optional[DifficultyScorer] = None,
                 selfcheck: Optional[SelfChecker] = None, cache_size: int = 256):
        self.client = client
        self.scorer = scorer or DifficultyScorer()
        self.selfcheck = selfcheck or SelfChecker()
        self.cache: OrderedDict = OrderedDict()
        self.cache_size = cache_size

        # 会话态
        self.current_tier = "hy3"
        self._consecutive_low = 0
        self._last_downgrade_at = 0.0

        # 熔断器
        self._breakers = {t: CircuitBreaker() for t in TIERS}

        # 统计
        self.total_calls = 0
        self.tier_calls = {t: 0 for t in TIERS}
        self.estimated_cost = 0.0

    # ------------------------------------------------------------- 路由决策

    def route(self, query: str, history: Optional[List[str]] = None) -> RouteDecision:
        result = self.scorer.score(query, history)
        target = primary_route(result.score)
        chosen, reason = self._apply_hysteresis(target)
        return RouteDecision(
            tier=chosen,
            score=result.score,
            confidence=result.confidence,
            reason=reason,
            breakdown=result.breakdown,
        )

    def _apply_hysteresis(self, target: str) -> tuple:
        """滞回状态机：升级快、降级慢。"""
        cur = self.current_tier
        if TIER_INDEX[target] > TIER_INDEX[cur]:
            self._consecutive_low = 0
            return target, f"upgrade {cur}->{target} (target higher)"
        if TIER_INDEX[target] < TIER_INDEX[cur]:
            self._consecutive_low += 1
            now = time.time()
            cooldown_ok = now - self._last_downgrade_at >= DOWNGRADE_COOLDOWN_S
            if self._consecutive_low >= DOWNGRADE_CONSECUTIVE and cooldown_ok:
                self._consecutive_low = 0
                self._last_downgrade_at = now
                return target, f"downgrade {cur}->{target} (consecutive low + cooldown)"
            return cur, f"hold {cur} (downgrade pending {self._consecutive_low}/{DOWNGRADE_CONSECUTIVE})"
        self._consecutive_low = 0
        return cur, f"hold {cur} (target == current)"

    # --------------------------------------------------------------- 执行

    def execute(self, query: str, history: Optional[List[str]] = None) -> ExecutionResult:
        """完整执行：路由 → 熔断 → 生成 → L3 自检 → 升级重试。"""
        cache_hit = False
        decision = self._cached_route(query, history)
        if decision is None:
            decision = self.route(query, history)
            self._cache_put(query, decision)
        else:
            cache_hit = True

        # 会话档由滞回决策推进；熔断降级只影响本次实际调用档，不改会话档
        self._update_current(decision.tier)
        tier, used_fallback = self._apply_circuit_breaker(decision.tier)

        response, selfcheck, escalated, final_tier = self._generate_with_check(
            query, history, tier)

        return ExecutionResult(
            tier=final_tier,
            response=response,
            decision=decision,
            selfcheck=selfcheck,
            escalated=escalated,
            used_fallback=used_fallback,
            cache_hit=cache_hit,
        )

    def _apply_circuit_breaker(self, tier: str) -> tuple:
        """若目标档熔断，降级到相邻可用档。"""
        if not self._breakers[tier].is_open():
            return tier, False
        idx = TIER_INDEX[tier]
        for fallback in reversed(TIERS[:idx]):
            if not self._breakers[fallback].is_open():
                return fallback, True
        # 全部熔断：用最便宜档硬着头皮上
        return TIERS[0], True

    def _generate_with_check(self, query, history, tier):
        """生成 + L3 自检 + 升级重试（最高到 pro）。返回最终 tier。"""
        messages = self._build_messages(query, history)
        response = self._call(tier, messages)
        check = self.selfcheck.check(query, response, tier)

        escalated = False
        while check.needs_upgrade and TIER_INDEX[tier] < TIER_INDEX["pro"]:
            escalated = True
            nxt = TIERS[TIER_INDEX[tier] + 1]
            if self._breakers[nxt].is_open():
                break
            tier = nxt
            self._update_current(tier)
            response = self._call(tier, messages)
            check = self.selfcheck.check(query, response, tier)

        return response, check, escalated, tier

    def _call(self, tier: str, messages: List[dict]) -> str:
        self.total_calls += 1
        self.tier_calls[tier] += 1
        try:
            resp = self.client.generate(tier, messages)
            self._breakers[tier].record(True)
            self.estimated_cost += TIER_CONFIGS[tier].cost_per_1k_tokens
            return resp
        except Exception:
            self._breakers[tier].record(False)
            raise

    def _build_messages(self, query: str, history: Optional[List[str]]) -> List[dict]:
        msgs: List[dict] = []
        if history:
            for h in history:
                msgs.append({"role": "user", "content": h})
        msgs.append({"role": "user", "content": query})
        return msgs

    def _update_current(self, tier: str) -> None:
        self.current_tier = tier

    # --------------------------------------------------------------- 缓存

    def _cached_route(self, query: str, history) -> Optional[RouteDecision]:
        if history:   # 有上下文时不走缓存，保证状态一致
            return None
        return self.cache.get(query)

    def _cache_put(self, query: str, decision: RouteDecision) -> None:
        self.cache[query] = decision
        self.cache.move_to_end(query)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    # --------------------------------------------------------------- 统计

    def stats(self) -> dict:
        return {
            "current_tier": self.current_tier,
            "total_calls": self.total_calls,
            "tier_calls": dict(self.tier_calls),
            "estimated_cost": round(self.estimated_cost, 5),
            "breakers_open": {t: self._breakers[t].is_open() for t in TIERS},
        }
