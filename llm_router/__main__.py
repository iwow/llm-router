"""演示入口：python -m llm_router

用 Mock 客户端离线跑一组样例，打印每个样例的难度分、档位决策与原因，
最后输出成本统计，验证阈值切得对不对。
"""

from __future__ import annotations

from .router import Router
from .registry import MockClient

SAMPLES = [
    "你好，今天天气怎么样？",
    "1+1 等于几？",
    "帮我写个 Python 函数，读取 CSV 并计算每列的平均值",
    "用 python 写一个类，实现线程安全的 LRU 缓存，并分析时间复杂度",
    "我有一个分布式系统偶发性读放大，帮我定位根因并设计修复方案",
    "用数学归纳法证明前 n 个自然数的平方和公式，并论证推导过程",
    "对比一下微服务和单体架构的优劣，给出选型权衡，输出成 markdown 表格",
    "帮我起草一份软件外包合同，重点覆盖知识产权条款和违约责任，并标注法律风险",
]


def main() -> None:
    router = Router(MockClient(latency_ms=0))
    print("=" * 78)
    print("自适应路由演示（Mock 客户端，零网络）")
    print("=" * 78)
    for q in SAMPLES:
        r = router.execute(q)
        print(f"\nQ: {q}")
        print(f"   D={r.decision.score:>5}  tier={r.tier:<5}  置信度={r.decision.confidence}")
        print(f"   reason: {r.decision.reason}")
        if r.escalated:
            print("   [已升级重试]")
        bd = r.decision.breakdown
        print(f"   维度分解: task={bd['task']} domain={bd['domain']} "
              f"output={bd['output']} context={bd['context']} novelty={bd['novelty']}")

    print("\n" + "=" * 78)
    s = router.stats()
    print(f"当前档: {s['current_tier']}  总调用: {s['total_calls']}")
    print(f"各档调用: {s['tier_calls']}")
    print(f"估算成本: ¥{s['estimated_cost']}")


if __name__ == "__main__":
    main()
