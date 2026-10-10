"""自适应大模型路由器包。

用法：
    from llm_router import Router, MockClient
    router = Router(MockClient())
    r = router.execute("你好，今天天气怎么样？")
    print(r.tier, r.response)
"""

from .scorer import DifficultyScorer, DifficultyResult
from .registry import TIERS, TIER_CONFIGS, LLMClient, MockClient, OpenAICompatClient
from .selfcheck import SelfChecker, SelfCheckResult
from .router import (
    Router,
    CircuitBreaker,
    RouteDecision,
    ExecutionResult,
    primary_route,
)

__all__ = [
    "Router", "CircuitBreaker", "RouteDecision", "ExecutionResult", "primary_route",
    "DifficultyScorer", "DifficultyResult",
    "TIERS", "TIER_CONFIGS", "LLMClient", "MockClient", "OpenAICompatClient",
    "SelfChecker", "SelfCheckResult",
]
