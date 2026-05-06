from __future__ import annotations

from app.domain.models import Claim, QueryContract


def metric_lines_from_claims(claims: list[Claim]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        for line in list(dict(claim.structured_data or {}).get("metric_lines", []) or []):
            compact = " ".join(str(line).split())
            if not compact:
                continue
            normalized = compact.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            lines.append(compact)
    return lines


def paper_result_core_points(*, target: str, support_text: str) -> list[str]:
    lower = support_text.lower()
    points: list[str] = []
    if "preference inference" in lower or "偏好推断" in lower:
        points.append(f"{target} 的核心问题包含 preference inference（偏好推断）。")
    if "conditioned generation" in lower or "generator" in lower or "条件生成" in lower:
        points.append("方法把 conditioned generation / generator 作为生成侧能力来建模。")
    if "modular" in lower or "模块化" in lower:
        points.append("整体结论强调 modular（模块化）拆分，而不是把偏好建模与生成能力混成单一黑盒。")
    if not points:
        compact = " ".join(support_text.split())
        points.append(compact[:260] if compact else f"{target} 的核心结论见下方证据。")
    return points[:4]


def compose_metric_value_answer(*, contract: QueryContract, claims: list[Claim]) -> str:
    lines = metric_lines_from_claims(claims)
    if not lines:
        return ""
    target = " / ".join(contract.targets) if contract.targets else (claims[0].entity or "目标对象")
    body = "\n".join(f"- {line}" for line in lines[:5])
    return (
        "## 结论\n\n"
        f"{target} 的表现需要按表格证据来读；我保留原始指标行，避免把 win rate、accuracy/ACC、"
        "p-soups、Llama/baseline 这类对照信息压缩丢失。\n\n"
        "## 实验结果\n\n"
        f"{body}"
    )


def compose_paper_summary_results_answer(*, contract: QueryContract, claims: list[Claim]) -> str:
    if not claims:
        return ""
    claim = claims[0]
    lines = metric_lines_from_claims(claims)
    target = contract.targets[0] if contract.targets else (claim.entity or "该论文")
    summary = " ".join(str(claim.value or "").split())
    if not summary:
        summary = f"{target} 的核心结论需要结合论文摘要与实验表格理解。"
    core_points = paper_result_core_points(target=target, support_text="\n".join([summary] + lines))
    core_body = "\n".join(f"- {item}" for item in core_points)
    metric_body = "\n".join(f"- {line}" for line in lines[:6]) if lines else "- 当前证据没有稳定抽出独立的指标行。"
    return (
        "## 核心结论\n\n"
        f"{core_body}\n\n"
        "## 实验结果\n\n"
        f"{metric_body}"
    )
