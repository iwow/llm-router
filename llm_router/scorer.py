"""L1/L2 难度打分器 —— 零依赖，规则引擎 + 加权合成。

设计要点：
- 难度不是单维，拆成 5 个正交信号，加权求和成 D ∈ [0, 100]。
- 线性加权表达不了"强推理 × 强领域"的复合难度，因此额外引入复合加成：
  当 task 与 domain 同时偏高时，难度近似两者的乘积而非和。
- L1 规则引擎处理"明显"场景，命中即高置信；置信度随偏离阈值而降低，供滞回决策使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DifficultyResult:
    score: float
    confidence: float
    breakdown: dict
    evidence: List[str]


# 各维度满分（与设计文档一致）
W_TASK = 35       # 任务复杂度（推理链/多步）
W_DOMAIN = 25     # 领域专业度
W_OUTPUT = 20     # 输出约束
W_CONTEXT = 15    # 上下文复杂度
W_NOVELTY = 5     # 新颖度/歧义度

# 一般推理动词
REASONING_VERBS = [
    "分析", "证明", "设计", "推导", "评估", "对比", "权衡", "优化", "重构",
    "架构", "论证", "推演", "归纳", "综合", "review", "设计评审", "code review",
    "起草", "制定", "拟定", "撰写", "编写", "实现", "开发", "构建",
]

# 问题解决/调试动词（高难度强信号）
DEBUG_VERBS = [
    "定位", "根因", "诊断", "排查", "调试", "debug", "修复", "解决", "复现",
]

# 多步/步骤标记
MULTI_STEP_MARKERS = ["首先", "然后", "接着", "步骤", "第一步", "为什么", "为何",
                      "如何实现", "怎么做", "原理", "机制"]

# 领域：代码/系统
CODE_TOKENS = [
    "python", "java", "javascript", "typescript", "go", "golang", "rust", "c++",
    "sql", "函数", "类", "接口", "算法", "数据结构", "bug", "报错", "异常",
    "部署", "数据库", "api", "多线程", "并发", "正则", "递归", "复杂度",
    "内存泄漏", "死锁", "代码", "编程", "脚本", "系统", "分布式", "性能",
    "索引", "缓存", "读写", "网络", "服务", "内存", "集群", "节点", "延迟",
    "吞吐", "读放大", "架构", "微服务", "单体",
]

# 领域：数学/科学
MATH_TOKENS = [
    "证明", "定理", "方程", "导数", "积分", "概率", "统计", "矩阵", "收敛",
    "优化", "数值", "线性代数", "微积分", "归纳法", "公式", "级数", "数列",
    "平方和", "求和",
]

# 领域：法律/金融
LEGAL_FIN_TOKENS = [
    "法律", "合同", "条款", "合规", "诉讼", "知识产权", "专利", "股票",
    "估值", "财报", "基金", "期权", "风险", "投行", "尽职调查", "收益率",
    "违约", "责任",
]

# 输出约束
OUTPUT_STRUCTURED = ["json", "表格", "markdown", "yaml", "csv", "格式"]
OUTPUT_LONG = ["报告", "总结", "长文", "综述", "文档", "论文", "方案"]
OUTPUT_CITATION = ["引用", "apa", "参考文献", "出处", "来源", "标注"]

# 歧义/冲突标记
AMBIGUITY_TOKENS = ["但是", "然而", "可是", "不过", "矛盾", "冲突", "模糊", "偶发"]


class DifficultyScorer:
    """规则引擎打分器。无外部依赖，可单独测试与替换。"""

    def score(self, text: str, history: Optional[List[str]] = None) -> DifficultyResult:
        raw = text or ""
        t = raw.lower()
        evidence: List[str] = []

        task = self._task_score(t, evidence)
        domain = self._domain_score(t, evidence)
        output = self._output_score(t, evidence)
        context = self._context_score(raw, history, evidence)
        novelty = self._novelty_score(t, evidence)

        score = task + domain + output + context + novelty
        composite = 0.0
        if task >= 30 and domain >= 20:
            composite = 12.0
            evidence.append("composite:强推理×强领域(+12)")
        elif task >= 12 and domain >= 20:
            composite = 8.0
            evidence.append("composite:中推理×强领域(+8)")
        score += composite

        score = max(0.0, min(100.0, score))
        confidence = self._confidence(score)

        return DifficultyResult(
            score=round(score, 1),
            confidence=round(confidence, 2),
            breakdown={"task": round(task, 1), "domain": round(domain, 1),
                       "output": round(output, 1), "context": round(context, 1),
                       "novelty": round(novelty, 1), "composite": round(composite, 1)},
            evidence=evidence,
        )

    # ------------------------------------------------------------------ 维度

    def _task_score(self, t: str, ev: List[str]) -> float:
        """任务复杂度：推理/调试动词 + 多步标记 + 代码任务，满 35。"""
        score = 0.0
        for v in REASONING_VERBS:
            if v in t:
                score += 12.0
                ev.append(f"task:推理[{v}]")
        for v in DEBUG_VERBS:
            if v in t:
                score += 12.0
                ev.append(f"task:调试[{v}]")
        for m in MULTI_STEP_MARKERS:
            if m in t:
                score += 5.0
                ev.append(f"task:多步[{m}]")
        if any(tok in t for tok in CODE_TOKENS):
            score += 15.0
            ev.append("task:代码任务")
        return min(W_TASK, score)

    def _domain_score(self, t: str, ev: List[str]) -> float:
        """领域专业度：代码/数学/法律金融术语密度，满 25。"""
        score = 0.0
        for group, label in ((CODE_TOKENS, "code"), (MATH_TOKENS, "math"),
                             (LEGAL_FIN_TOKENS, "legal_fin")):
            for tok in group:
                if tok in t:
                    score += 10.0
                    ev.append(f"domain:{label}[{tok}]")
        return min(W_DOMAIN, score)

    def _output_score(self, t: str, ev: List[str]) -> float:
        """输出约束：结构化/长文/引用要求，满 20。"""
        score = 0.0
        for tok in OUTPUT_STRUCTURED:
            if tok in t:
                score += 7.0
                ev.append(f"output:结构化[{tok}]")
        for tok in OUTPUT_LONG:
            if tok in t:
                score += 8.0
                ev.append(f"output:长文[{tok}]")
        for tok in OUTPUT_CITATION:
            if tok in t:
                score += 6.0
                ev.append(f"output:引用[{tok}]")
        return min(W_OUTPUT, score)

    def _context_score(self, raw: str, history: Optional[List[str]], ev: List[str]) -> float:
        """上下文复杂度：对历史强依赖（指代词/衔接词），满 15。"""
        if not history:
            return 0.0
        score = 0.0
        for d in ("它", "这个", "那个", "上述", "之前", "接着", "继续", "还有", "那么"):
            if d in raw:
                score += 5.0
                ev.append(f"context:指代[{d}]")
        score += min(5.0, len(history) * 1.5)
        return min(W_CONTEXT, score)

    def _novelty_score(self, t: str, ev: List[str]) -> float:
        """新颖度/歧义度：指令冲突/矛盾，满 5。"""
        score = 0.0
        for tok in AMBIGUITY_TOKENS:
            if tok in t:
                score += 2.5
                ev.append(f"novelty:歧义[{tok}]")
        return min(W_NOVELTY, score)

    # --------------------------------------------------------------- 置信度

    def _confidence(self, score: float) -> float:
        """偏离阈值越远置信越高；落在阈值附近（35-45 / 65-75）置信低。"""
        for lo, hi in ((35.0, 45.0), (65.0, 75.0)):
            if lo <= score <= hi:
                center = (lo + hi) / 2
                dist = abs(score - center) / ((hi - lo) / 2)
                return round(max(0.55, 1.0 - 0.45 * (1 - dist)), 2)
        return 0.95
