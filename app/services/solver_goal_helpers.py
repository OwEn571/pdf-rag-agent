from __future__ import annotations

import re

from app.domain.models import QueryContract, ResearchPlan


def claim_goals(*, contract: QueryContract, plan: ResearchPlan) -> set[str]:
    goals = {
        str(item).strip()
        for item in [
            *list(plan.required_claims or []),
            *list(getattr(contract, "answer_slots", []) or []),
            *list(contract.requested_fields or []),
            *[
                str(note).split("=", 1)[1]
                for note in contract.notes
                if str(note).startswith("answer_slot=") and "=" in str(note)
            ],
        ]
        if str(item).strip()
    }
    if not goals or goals <= {"answer"}:
        goals.update(fallback_goals_from_query(contract.clean_query, targets=contract.targets))
    for modality in contract.required_modalities:
        if modality == "figure":
            goals.add("figure_conclusion")
        elif modality in {"table", "caption"} and looks_like_metric_goal(contract.clean_query, goals):
            goals.add("metric_value")
    if "formula" in goals:
        goals.add("source")
    if "training_component" in goals:
        goals.update({"mechanism", "reward_model_requirement", "evidence"})
    return goals


def looks_like_metric_goal(query: str, goals: set[str]) -> bool:
    if goals & {"metric_value", "setting"}:
        return True
    normalized = " ".join(str(query or "").lower().split())
    return any(token in normalized for token in ["多少", "数值", "准确率", "得分", "score", "accuracy", "metric", "win rate"])


def fallback_goals_from_query(query: str, *, targets: list[str]) -> set[str]:
    raw_query = str(query or "")
    normalized = " ".join(raw_query.lower().split())
    compact = re.sub(r"\s+", "", normalized)
    goals: set[str] = set()
    if any(
        token in normalized or token in compact
        for token in [
            "最早",
            "最先",
            "最初",
            "首次",
            "第一个提出",
            "第一篇提出",
            "第一篇论文",
            "谁提出",
            "提出的",
            "origin",
            "first proposed",
            "first introduced",
        ]
    ):
        goals.update({"paper_title", "year"})
    if any(token in normalized for token in ["公式", "损失函数", "objective", "loss", "gradient", "梯度"]):
        goals.add("formula")
    if any(token in normalized for token in ["后续", "followup", "follow-up", "successor"]):
        goals.add("followup_papers")
    if any(token in normalized for token in ["figure", "fig.", "图", "caption"]):
        goals.add("figure_conclusion")
    if any(token in normalized for token in ["结果", "实验", "核心结论", "贡献", "summary", "result"]):
        goals.update({"summary", "results"})
    if looks_like_metric_goal(query, goals):
        goals.add("metric_value")
    if any(token in normalized for token in ["推荐", "值得", "入门", "recommend"]):
        goals.add("recommended_papers")
    if (targets and any(token in raw_query for token in ["是什么", "什么意思", "定义"])) or any(
        token in normalized for token in ["what is", "what are"]
    ):
        goals.update({"entity_type", "definition", "mechanism"})
    return goals or {"answer"}
