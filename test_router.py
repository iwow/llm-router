"""路由器核心逻辑验证（离线，零网络）。

覆盖：
1. 主路由阈值（D<40→hy3 / 40≤D<70→flash / D≥70→pro）
2. 三个基准样例的难度分落在预期区间
3. 滞回：升级单向快速、降级保守（连续 3 次）
4. 熔断器：错误率超阈值摘除
5. 升级重试：flash 低置信 → 自动升 pro
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_router.scorer import DifficultyScorer
from llm_router.router import Router, CircuitBreaker, primary_route
from llm_router.registry import MockClient

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_primary_route():
    print("\n[1] 主路由阈值")
    check("D<40 -> hy3", primary_route(5) == "hy3")
    check("D=39.9 -> hy3", primary_route(39.9) == "hy3")
    check("D=40 -> flash", primary_route(40) == "flash")
    check("D=55 -> flash", primary_route(55) == "flash")
    check("D=69.9 -> flash", primary_route(69.9) == "flash")
    check("D=70 -> pro", primary_route(70) == "pro")
    check("D=95 -> pro", primary_route(95) == "pro")


def test_scorer_bands():
    print("\n[2] 基准样例难度分区间")
    s = DifficultyScorer()
    cases = [
        ("你好，今天天气怎么样？", 0, 40),
        ("1+1 等于几？", 0, 40),
        ("帮我写个 Python 函数，读取 CSV 并计算每列的平均值", 40, 70),
        ("我有一个分布式系统偶发性读放大，帮我定位根因并设计修复方案", 70, 101),
        ("用数学归纳法证明前 n 个自然数的平方和公式，并论证推导过程", 70, 101),
    ]
    for q, lo, hi in cases:
        sc = s.score(q).score
        ok = lo <= sc < hi
        check(f"D={sc} 在 [{lo},{hi})  {q[:24]}...", ok, f"score={sc}")


def test_hysteresis():
    print("\n[3] 滞回：升级快、降级慢")
    r = Router(MockClient())

    # 初始 hy3，遇到高难度立即升级
    d1 = r.route("帮我定位分布式系统的读放大根因并设计修复方案")
    check("高难度立即升级到 pro", d1.tier == "pro", d1.reason)
    r._update_current("pro")

    # 连续简单问题，需 3 次才降级
    for i in range(1, 4):
        d = r.route("你好呀")
        if i < 3:
            check(f"第{i}次简单问题仍 hold pro", d.tier == "pro", d.reason)
        else:
            check("第3次简单问题降级到 hy3", d.tier == "hy3", d.reason)


def test_circuit_breaker():
    print("\n[4] 熔断器")
    cb = CircuitBreaker()
    for _ in range(3):
        cb.record(False)
    cb.record(True)
    check("错误率 75% 触发熔断", cb.is_open() is True)


def test_escalation():
    print("\n[5] 升级重试（L3 自检低置信 → 升 pro）")

    class ScriptedClient:
        def __init__(self):
            self.responses = ["我不确定，可能不对", "确定答案：读放大源于缺少索引"]

        def generate(self, tier, messages, **kw):
            return self.responses.pop(0)

    r = Router(ScriptedClient())
    r._update_current("flash")
    res = r.execute("分析这个 bug 的根因")
    check("flash 低置信自动升级", res.escalated is True)
    check("最终用 pro 返回", res.tier == "pro", f"tier={res.tier}")
    check("拿到确定答案", res.response.startswith("确定答案"))


if __name__ == "__main__":
    test_primary_route()
    test_scorer_bands()
    test_hysteresis()
    test_circuit_breaker()
    test_escalation()
    print(f"\n{'='*40}\n结果: {PASS} 通过, {FAIL} 失败")
    sys.exit(1 if FAIL else 0)
