"""模型注册表 + 客户端抽象 + 离线 Mock / OpenAI 兼容客户端。

- TIER_CONFIGS 定义三档的成本与延迟（用于路由决策与成本统计）。
- LLMClient 是协议；MockClient 零依赖离线测试；
- OpenAICompatClient 用标准库 urllib 调任意 OpenAI 兼容的 /chat/completions 端点。
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import List, Optional, Protocol


TIERS = ["hy3", "flash", "pro"]
TIER_INDEX = {t: i for i, t in enumerate(TIERS)}


@dataclass
class TierConfig:
    name: str
    cost_per_1k_tokens: float   # 每 1k token 成本（元，示意值）
    latency_ms: int


TIER_CONFIGS = {
    "hy3": TierConfig("hy3", 0.0, 100),
    "flash": TierConfig("flash", 0.001, 300),
    "pro": TierConfig("pro", 0.01, 1500),
}


class LLMClient(Protocol):
    def generate(self, tier: str, messages: List[dict], **kw) -> str:
        ...


class MockClient:
    """离线 Mock：可配置失败率与延迟，用于验证路由/熔断/升级逻辑。"""

    def __init__(self, fail_rate: float = 0.0, latency_ms: int = 0, seed: int = 0):
        self.fail_rate = fail_rate
        self.latency_ms = latency_ms
        self._state = seed
        self.calls: List[str] = []   # 记录调用过的 tier，便于断言

    def generate(self, tier: str, messages: List[dict], **kw) -> str:
        self.calls.append(tier)
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        # 简单 LCG，确定性随机，便于复现测试
        self._state = (self._state * 1103515245 + 12345) & 0x7FFFFFFF
        if (self._state / 0x7FFFFFFF) < self.fail_rate:
            raise RuntimeError(f"[{tier}] mock failure")
        last = messages[-1]["content"] if messages else ""
        return f"[{tier}] ok: {last[:40]}"


class OpenAICompatClient:
    """用标准库调 OpenAI 兼容接口。map 是把 tier 映射到真实模型名的字典。"""

    def __init__(self, api_key: str, base_url: str, model_map: dict,
                 timeout: int = 30, max_tokens: int = 1024):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_map = model_map
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(self, tier: str, messages: List[dict], **kw) -> str:
        model = self.model_map.get(tier, tier)
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", self.max_tokens),
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")
