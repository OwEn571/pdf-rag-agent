from __future__ import annotations

from typing import Literal

RESEARCH_SLOT_PROFILES: dict[str, dict[str, object]] = {
    "origin": {
        "relation": "origin_lookup",
        "requested_fields": ["paper_title", "year", "evidence"],
        "required_modalities": ["paper_card", "page_text"],
        "answer_shape": "narrative",
        "precision_requirement": "exact",
    },
    "formula": {
        "relation": "formula_lookup",
        "requested_fields": ["formula", "variable_explanation", "source"],
        "required_modalities": ["page_text", "table"],
        "answer_shape": "bullets",
        "precision_requirement": "exact",
    },
    "followup_research": {
        "relation": "followup_research",
        "requested_fields": ["followup_papers", "relationship", "evidence"],
        "required_modalities": ["paper_card", "page_text"],
        "answer_shape": "bullets",
        "precision_requirement": "high",
    },
    "figure": {
        "relation": "figure_question",
        "requested_fields": ["figure_conclusion", "caption", "evidence"],
        "required_modalities": ["figure", "caption", "page_text"],
        "answer_shape": "bullets",
        "precision_requirement": "high",
    },
    "metric_value": {
        "relation": "metric_value_lookup",
        "requested_fields": ["metric_value", "setting", "evidence"],
        "required_modalities": ["table", "caption", "page_text"],
        "answer_shape": "narrative",
        "precision_requirement": "exact",
    },
    "paper_summary": {
        "relation": "paper_summary_results",
        "requested_fields": ["summary", "results", "evidence"],
        "required_modalities": ["page_text", "paper_card", "table", "caption"],
        "answer_shape": "narrative",
        "precision_requirement": "high",
    },
    "paper_recommendation": {
        "relation": "paper_recommendation",
        "requested_fields": ["recommended_papers", "rationale"],
        "required_modalities": ["paper_card", "page_text"],
        "answer_shape": "bullets",
        "precision_requirement": "high",
    },
    "comparison": {
        "relation": "general_question",
        "requested_fields": ["comparison", "relationship", "evidence"],
        "required_modalities": ["paper_card", "page_text"],
        "answer_shape": "narrative",
        "precision_requirement": "high",
    },
    "topology_recommendation": {
        "relation": "topology_recommendation",
        "requested_fields": ["best_topology", "langgraph_recommendation"],
        "required_modalities": ["page_text", "paper_card"],
        "answer_shape": "bullets",
        "precision_requirement": "high",
    },
    "topology_discovery": {
        "relation": "topology_discovery",
        "requested_fields": ["relevant_papers", "topology_types"],
        "required_modalities": ["page_text", "paper_card"],
        "answer_shape": "bullets",
        "precision_requirement": "high",
    },
    "entity_definition": {
        "relation": "entity_definition",
        "requested_fields": ["definition", "mechanism", "role_in_context"],
        "required_modalities": ["page_text", "paper_card", "table"],
        "answer_shape": "narrative",
        "precision_requirement": "high",
    },
    "concept_definition": {
        "relation": "concept_definition",
        "requested_fields": ["definition", "mechanism", "examples"],
        "required_modalities": ["page_text", "paper_card"],
        "answer_shape": "narrative",
        "precision_requirement": "high",
    },
    "training_component": {
        "relation": "general_question",
        "requested_fields": ["mechanism", "reward_model_requirement", "evidence"],
        "required_modalities": ["page_text", "paper_card"],
        "answer_shape": "narrative",
        "precision_requirement": "high",
    },
    "general_answer": {
        "relation": "general_question",
        "requested_fields": ["answer"],
        "required_modalities": ["page_text", "paper_card"],
        "answer_shape": "narrative",
        "precision_requirement": "high",
    },
}

RELATION_ANSWER_SLOTS: dict[str, list[str]] = {
    "greeting": ["greeting"],
    "self_identity": ["self_identity"],
    "capability": ["capability"],
    "clarify_user_intent": ["clarify"],
    "library_status": ["library_status"],
    "library_recommendation": ["library_recommendation"],
    "library_citation_ranking": ["citation_ranking"],
    "memory_followup": ["previous_rationale"],
    "memory_synthesis": ["comparison"],
    "origin_lookup": ["origin"],
    "formula_lookup": ["formula"],
    "followup_research": ["followup_research"],
    "entity_definition": ["entity_definition"],
    "topology_discovery": ["topology_discovery"],
    "topology_recommendation": ["topology_recommendation"],
    "figure_question": ["figure"],
    "paper_summary_results": ["paper_summary"],
    "metric_value_lookup": ["metric_value"],
    "concept_definition": ["concept_definition"],
    "paper_recommendation": ["paper_recommendation"],
    "general_question": ["general_answer"],
}


def answer_slots_from_relation(relation: str) -> list[str]:
    return list(RELATION_ANSWER_SLOTS.get(str(relation or "").strip(), ["general_answer"]))


def research_profile_slots(*, slots: list[str], clean_query: str, targets: list[str]) -> list[str]:
    profile_slots: list[str] = []
    for slot in slots or ["general_answer"]:
        key = "_".join(str(slot or "").strip().lower().replace("-", "_").split())
        if key == "definition":
            key = "entity_definition" if targets and not str(clean_query or "").startswith("什么是") else "concept_definition"
        if key not in RESEARCH_SLOT_PROFILES:
            key = "general_answer"
        if key not in profile_slots:
            profile_slots.append(key)
    return profile_slots or ["general_answer"]


def research_relation_from_slots(*, slots: list[str], clean_query: str, targets: list[str]) -> str:
    profile_slots = research_profile_slots(slots=slots, clean_query=clean_query, targets=targets)
    first_profile = RESEARCH_SLOT_PROFILES.get(profile_slots[0] if profile_slots else "general_answer", {})
    return str(first_profile.get("relation") or "general_question")


def research_requirements_from_slots(
    *,
    slots: list[str],
    targets: list[str],
    clean_query: str,
) -> tuple[list[str], list[str], str, Literal["exact", "high", "normal"]]:
    profile_slots = research_profile_slots(slots=slots, clean_query=clean_query, targets=targets)
    requested_fields: list[str] = []
    required_modalities: list[str] = []
    shapes: list[str] = []
    precision_values: list[str] = []
    for slot in profile_slots:
        profile = RESEARCH_SLOT_PROFILES.get(slot) or RESEARCH_SLOT_PROFILES["general_answer"]
        requested_fields.extend(str(item) for item in list(profile.get("requested_fields", []) or []) if str(item))
        required_modalities.extend(str(item) for item in list(profile.get("required_modalities", []) or []) if str(item))
        shapes.append(str(profile.get("answer_shape") or "narrative"))
        precision_values.append(str(profile.get("precision_requirement") or "high"))
    answer_shape = "table" if "table" in shapes else "bullets" if "bullets" in shapes else "narrative"
    precision_requirement: Literal["exact", "high", "normal"] = (
        "exact" if "exact" in precision_values else "high" if "high" in precision_values else "normal"
    )
    return (
        list(dict.fromkeys(requested_fields or ["answer"])),
        list(dict.fromkeys(required_modalities or ["page_text", "paper_card"])),
        answer_shape,
        precision_requirement,
    )
