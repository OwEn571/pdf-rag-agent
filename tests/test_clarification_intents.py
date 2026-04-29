from __future__ import annotations

import json
from types import SimpleNamespace

from app.domain.models import CandidatePaper, DisambiguationJudgeDecision, EvidenceBlock, QueryContract, SessionContext, VerificationReport
from app.services.clarification_intents import (
    CLARIFICATION_INTENT_MARKERS,
    CLARIFICATION_OPTION_SCHEMA_VERSION,
    ambiguity_option_context_text,
    ambiguity_option_matches_context,
    acronym_evidence_from_corpus,
    acronym_options_from_evidence,
    ambiguity_clarification_question,
    ambiguity_options_from_notes,
    apply_disambiguation_judge_recommendation,
    clarification_tracking_key,
    candidate_origin_signal_score,
    candidate_title_alignment_score,
    candidate_usage_signal_score,
    clarification_option_description,
    clarification_option_id,
    clarification_option_public_payload,
    clarification_options_from_contract_notes,
    clarification_string_list,
    contract_from_selected_clarification_option,
    contract_needs_evidence_disambiguation,
    contract_with_auto_resolved_ambiguity,
    contract_with_ambiguity_options,
    clear_pending_clarification,
    contract_from_pending_clarification,
    disambiguation_judge_human_prompt,
    disambiguation_judge_option_payload,
    disambiguation_judge_summary,
    disambiguation_judge_system_prompt,
    disambiguation_missing_fields,
    evidence_disambiguation_options,
    extract_acronym_expansion_from_text,
    finalize_acronym_disambiguation_options,
    judge_allows_auto_resolve,
    looks_like_clarification_choice_text,
    normalize_acronym_meaning,
    normalize_clarification_options,
    next_clarification_attempt,
    option_from_clarification_choice,
    pending_clarification_selection_index,
    remember_clarification_attempt,
    reset_clarification_tracking,
    resolve_disambiguation_judge_decision,
    select_pending_clarification_option,
    selected_option_from_judge_decision,
    selected_clarification_paper_id,
    store_pending_clarification,
)
from app.services.intent_marker_matching import query_matches_any


def test_clarification_choice_text_detects_selection_cues() -> None:
    assert looks_like_clarification_choice_text("我说的是第二个选项")
    assert looks_like_clarification_choice_text("choose the one about alignx")
    assert not looks_like_clarification_choice_text("alignx paper")
    assert query_matches_any("select", "", CLARIFICATION_INTENT_MARKERS["choice"])


def test_pending_clarification_selection_index_detects_digits_and_ordinals() -> None:
    assert pending_clarification_selection_index("选 2") == 1
    assert pending_clarification_selection_index("第二个") == 1
    assert pending_clarification_selection_index("the third") == 2
    assert pending_clarification_selection_index("没有明确选择") is None


def test_clarification_option_public_payload_preserves_protocol_fields() -> None:
    payload = clarification_option_public_payload(
        {
            "option_id": "opt-1",
            "kind": "acronym_meaning",
            "title": "AlignX",
            "display_reason": "best match",
            "judge_recommended": True,
            "debug_only": "drop",
        }
    )

    assert payload["schema_version"] == CLARIFICATION_OPTION_SCHEMA_VERSION
    assert payload["option_id"] == "opt-1"
    assert payload["display_reason"] == "best match"
    assert payload["judge_recommended"] is True
    assert "debug_only" not in payload


def test_clarification_string_list_coerces_common_shapes() -> None:
    assert clarification_string_list([" A ", "", "B"]) == ["A", "B"]
    assert sorted(clarification_string_list({"B", "A"})) == ["A", "B"]
    assert clarification_string_list(" value ") == ["value"]
    assert clarification_string_list(None) == []


def test_clarification_option_description_and_id_are_stable() -> None:
    assert clarification_option_description({"snippet": "  alpha   beta "}, title="T", year="2026") == "alpha beta"
    assert clarification_option_description({}, title="T", year="2026") == "T · 2026"

    first = clarification_option_id(
        kind="acronym_meaning",
        target="DPO",
        label="Direct Preference Optimization",
        paper_id="p1",
        title="Paper",
        index=0,
    )
    second = clarification_option_id(
        kind="acronym_meaning",
        target="DPO",
        label="Direct Preference Optimization",
        paper_id="p1",
        title="Paper",
        index=0,
    )
    assert first == second
    assert first.startswith("acronym-meaning-dpo-")


def test_ambiguity_options_from_notes_reads_valid_payloads_only() -> None:
    valid = {"title": "Preference Bridged Alignment", "option_id": "pba"}
    notes = [
        "plain note",
        "ambiguity_option=not-json",
        "ambiguity_option=" + json.dumps({"option_id": "missing-title"}),
        "ambiguity_option=" + json.dumps(valid),
    ]

    assert ambiguity_options_from_notes(notes) == [valid]


def test_selected_clarification_paper_id_reads_direct_and_payload_notes() -> None:
    direct = QueryContract(clean_query="x", notes=["selected_paper_id=p1"])
    payload = QueryContract(
        clean_query="x",
        notes=["selected_ambiguity_option=" + json.dumps({"paper_id": "p2"})],
    )

    assert selected_clarification_paper_id(direct) == "p1"
    assert selected_clarification_paper_id(payload) == "p2"
    assert selected_clarification_paper_id(QueryContract(clean_query="x", notes=["selected_ambiguity_option=bad"])) == ""


def test_option_from_clarification_choice_matches_id_index_and_fields() -> None:
    options = [
        {"option_id": "a", "paper_id": "p1", "meaning": "Alpha", "label": "First"},
        {"option_id": "b", "paper_id": "p2", "meaning": "Beta", "label": "Second"},
    ]

    assert option_from_clarification_choice({"option_id": "b"}, options) == options[1]
    assert option_from_clarification_choice({"index": 0}, options) == options[0]
    assert option_from_clarification_choice({"paper_id": "p2"}, options) == options[1]
    assert option_from_clarification_choice({"meaning": "alpha"}, options) == options[0]
    assert option_from_clarification_choice({"label": "second"}, options) == options[1]
    assert option_from_clarification_choice({"option_id": "missing"}, options) is None


def test_select_pending_clarification_option_matches_textual_choice() -> None:
    options = [
        {"meaning": "Direct Preference Optimization", "label": "DPO", "title": "DPO Paper"},
        {"meaning": "Preference Bridged Alignment", "label": "PBA", "title": "PBA Paper"},
    ]

    assert select_pending_clarification_option(clean_query="第二个", options=options) == options[1]
    assert select_pending_clarification_option(clean_query="我说的是 Direct Preference Optimization", options=options) == options[0]
    assert select_pending_clarification_option(clean_query="PBA Paper", options=options) == options[1]
    assert select_pending_clarification_option(clean_query="没有选择", options=options) is None


def test_contract_with_ambiguity_options_replaces_old_option_notes() -> None:
    contract = QueryContract(clean_query="x", notes=["keep", "ambiguity_option=" + json.dumps({"title": "Old"})])
    option = {"option_id": "new", "kind": "acronym_meaning", "title": "New", "debug": "drop"}

    updated = contract_with_ambiguity_options(contract=contract, options=[option])

    assert updated.notes[0] == "keep"
    assert len([note for note in updated.notes if note.startswith("ambiguity_option=")]) == 1
    payload = json.loads(updated.notes[1].split("=", 1)[1])
    assert payload["title"] == "New"
    assert "debug" not in payload


def test_normalize_clarification_options_fills_protocol_fields() -> None:
    contract = QueryContract(
        clean_query="PBA是什么",
        relation="entity_definition",
        targets=["PBA"],
        requested_fields=["definition"],
        required_modalities=["page_text"],
        answer_slots=["definition"],
    )
    options = normalize_clarification_options(
        [{"paper_id": "p1", "title": "AlignX", "year": "2025", "meaning": "Preference Bridged Alignment"}],
        contract=contract,
        kind="acronym_meaning",
        source="unit_test",
    )

    assert options[0]["schema_version"] == CLARIFICATION_OPTION_SCHEMA_VERSION
    assert options[0]["target"] == "PBA"
    assert options[0]["label"] == "Preference Bridged Alignment"
    assert options[0]["source"] == "unit_test"
    assert options[0]["source_requested_fields"] == ["definition"]
    assert options[0]["source_required_modalities"] == ["page_text"]
    assert options[0]["source_answer_slots"] == ["definition"]
    assert options[0]["option_id"].startswith("acronym-meaning-pba-")


def test_clarification_options_from_contract_notes_normalizes_payloads() -> None:
    contract = contract_with_ambiguity_options(
        contract=QueryContract(clean_query="PBA是什么", relation="entity_definition", targets=["PBA"]),
        options=[{"paper_id": "p1", "title": "AlignX", "meaning": "Preference Bridged Alignment"}],
    )

    options = clarification_options_from_contract_notes(contract)

    assert options[0]["kind"] == "acronym_meaning"
    assert "source" in options[0]
    assert options[0]["target"] == "PBA"


def test_contract_from_selected_clarification_option_preserves_formula_slots() -> None:
    contract = contract_from_selected_clarification_option(
        clean_query="选第二个",
        target="PBA",
        selected={
            "meaning": "Preference Bridged Alignment",
            "title": "AlignX",
            "paper_id": "paper-1",
            "source_relation": "formula_lookup",
            "source_answer_slots": ["formula"],
        },
    )

    assert contract.relation == "formula_lookup"
    assert contract.answer_slots == ["formula"]
    assert contract.requested_fields == ["formula", "variable_explanation", "source"]
    assert "selected_paper_id=paper-1" in contract.notes


def test_disambiguation_ranking_signals_distinguish_origin_from_usage() -> None:
    origin_score = candidate_origin_signal_score("Our main contribution is Direct Preference Optimization (DPO).")
    usage_score = candidate_usage_signal_score("This paper adopts Direct Preference Optimization for tuning.")

    assert origin_score >= 0.35
    assert usage_score >= 0.25
    assert candidate_title_alignment_score(
        target="DPO",
        label="Direct Preference Optimization",
        meaning="Direct Preference Optimization",
        title_alias_text="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
    ) == 1.0


def test_disambiguation_judge_payload_includes_metadata_and_signals() -> None:
    paper = CandidatePaper(
        paper_id="DPO",
        title="Direct Preference Optimization",
        match_reason="title match",
        metadata={
            "aliases": "DPO",
            "generated_summary": "This paper proposes Direct Preference Optimization.",
        },
    )
    option = {
        "option_id": "dpo",
        "target": "DPO",
        "meaning": "Direct Preference Optimization",
        "paper_id": "DPO",
        "title": "Direct Preference Optimization",
        "snippet": "Our main contribution is Direct Preference Optimization (DPO).",
        "source_requested_fields": ("formula", "source"),
    }

    payload = disambiguation_judge_option_payload(option=option, paper=paper)

    assert payload["paper_aliases"] == "DPO"
    assert payload["match_reason"] == "title match"
    assert payload["source_requested_fields"] == ["formula", "source"]
    assert payload["ranking_signals"]["candidate_role_hint"] == "direct_definition_or_origin"


def test_selected_option_and_judge_summary_are_stable() -> None:
    options = [{"option_id": "a", "paper_id": "p1"}, {"option_id": "b", "paper_id": "p2"}]
    decision = DisambiguationJudgeDecision(
        decision="ask_human",
        selected_option_id="b",
        selected_paper_id="p2",
        confidence=0.72,
    )

    assert selected_option_from_judge_decision(decision=decision, options=options) == options[1]
    assert disambiguation_judge_summary(options=options, judge_decision=decision) == "options=2, judge=ask_human, confidence=0.72"
    assert disambiguation_judge_summary(options=options, judge_decision=None) == "options=2, judge=unavailable"


def test_disambiguation_judge_prompts_include_contract_and_candidate_payloads() -> None:
    contract = QueryContract(
        clean_query="PBA是什么",
        relation="entity_definition",
        targets=["PBA"],
        requested_fields=["definition"],
        notes=["ambiguity_option=hidden", "keep"],
    )

    prompt = disambiguation_judge_human_prompt(
        contract=contract,
        candidate_options=[{"option_id": "pba", "paper_id": "p1"}],
    )
    payload = json.loads(prompt)

    assert "通用论文/实体候选消歧裁判器" in disambiguation_judge_system_prompt()
    assert payload["user_query"] == "PBA是什么"
    assert payload["query_contract"]["requested_fields"] == ["definition"]
    assert payload["query_contract"]["notes"] == ["keep"]
    assert payload["candidate_options"] == [{"option_id": "pba", "paper_id": "p1"}]


def test_disambiguation_judge_recommendation_and_auto_contract_notes() -> None:
    options = [
        {"option_id": "a", "paper_id": "p1", "title": "First", "index": 1},
        {"option_id": "b", "paper_id": "p2", "title": "Second", "index": 0},
    ]
    decision = DisambiguationJudgeDecision(
        decision="auto_resolve",
        selected_option_id="b",
        selected_paper_id="p2",
        confidence=0.91,
        reason="Second has direct title alignment.",
        rejected_options=[{"option_id": "a", "reason": "Only a related usage."}],
    )

    annotated = apply_disambiguation_judge_recommendation(
        options=options,
        decision=decision,
        recommend_threshold=0.7,
    )
    updated = contract_with_auto_resolved_ambiguity(
        contract=QueryContract(
            clean_query="PBA是什么",
            notes=[
                "keep",
                "ambiguity_option=old",
                "selected_paper_id=old",
                "disambiguation_judge_confidence=0.100",
            ],
        ),
        selected=annotated[0],
        decision=decision,
    )

    assert judge_allows_auto_resolve(decision, threshold=0.9)
    assert annotated[0]["option_id"] == "b"
    assert annotated[0]["judge_recommended"] is True
    assert annotated[1]["display_reason"] == "Only a related usage."
    assert "keep" in updated.notes
    assert "auto_resolved_by_llm_judge" in updated.notes
    assert "selected_paper_id=p2" in updated.notes
    assert "disambiguation_judge_confidence=0.910" in updated.notes
    assert not any(note.startswith("ambiguity_option=old") for note in updated.notes)


def test_resolve_disambiguation_judge_decision_auto_resolves_contract() -> None:
    options = [
        {"option_id": "a", "paper_id": "p1", "title": "First", "index": 1},
        {"option_id": "b", "paper_id": "p2", "title": "Second", "index": 0},
    ]
    decision = DisambiguationJudgeDecision(
        decision="auto_resolve",
        selected_option_id="b",
        selected_paper_id="p2",
        confidence=0.91,
        reason="Second has direct title alignment.",
    )

    result = resolve_disambiguation_judge_decision(
        contract=QueryContract(clean_query="PBA是什么"),
        options=options,
        judge_decision=decision,
        auto_resolve_threshold=0.85,
        recommend_threshold=0.7,
    )

    assert result.auto_resolve is True
    assert result.selected_option == options[1]
    assert result.verification is None
    assert result.observation_tool == "resolve_ambiguity"
    assert result.observation_summary == "options=2, judge=auto_resolve, confidence=0.91"
    assert result.observation_payload["options"] == options
    assert "auto_resolved_by_llm_judge" in result.contract.notes
    assert "selected_paper_id=p2" in result.contract.notes


def test_resolve_disambiguation_judge_decision_marks_clarification_with_recommendation() -> None:
    options = [
        {"option_id": "a", "paper_id": "p1", "title": "First", "index": 1},
        {"option_id": "b", "paper_id": "p2", "title": "Second", "index": 0},
    ]
    decision = DisambiguationJudgeDecision(
        decision="ask_human",
        selected_option_id="b",
        selected_paper_id="p2",
        confidence=0.74,
        reason="Second is more likely but not enough for auto-resolve.",
    )

    result = resolve_disambiguation_judge_decision(
        contract=QueryContract(clean_query="PBA是什么", targets=["PBA"]),
        options=options,
        judge_decision=decision,
        auto_resolve_threshold=0.85,
        recommend_threshold=0.7,
    )

    assert result.auto_resolve is False
    assert result.observation_tool == "detect_ambiguity"
    assert result.options[0]["option_id"] == "b"
    assert result.options[0]["judge_recommended"] is True
    assert result.verification is not None
    assert result.verification.status == "clarify"
    assert result.verification.recommended_action == "clarify_ambiguous_entity"
    assert any(note.startswith("ambiguity_option=") for note in result.contract.notes)


def test_acronym_helpers_extract_normalize_and_match_context() -> None:
    expansion = extract_acronym_expansion_from_text(
        text="Preference-Bridged Alignment (PBA) is used for personalized alignment.",
        acronym="PBA",
    )
    paper = CandidatePaper(
        paper_id="pba",
        title="AlignX",
        metadata={"aliases": "PBA", "generated_summary": "Preference Bridged Alignment is a method."},
    )
    option = {"meaning": expansion, "title": "AlignX", "snippet": "PBA is evaluated on PPAIR."}
    option["context_text"] = ambiguity_option_context_text(option, paper=paper)

    assert expansion == "Preference-Bridged Alignment"
    assert normalize_acronym_meaning("Behaviour-based Alignment") == "behavior based alignment"
    assert ambiguity_option_matches_context(option=option, context_targets=["PPAIR"])


def test_acronym_options_from_evidence_groups_expansions_and_drops_plain_duplicates() -> None:
    papers = [
        CandidatePaper(paper_id="p1", title="Plain PBA"),
        CandidatePaper(paper_id="p2", title="AlignX"),
    ]
    evidence = [
        EvidenceBlock(
            doc_id="plain",
            paper_id="p1",
            title="Plain PBA",
            file_path="",
            page=1,
            block_type="page_text",
            snippet="PBA appears without expansion.",
            score=1.0,
        ),
        EvidenceBlock(
            doc_id="expanded",
            paper_id="p2",
            title="AlignX",
            file_path="",
            page=2,
            block_type="page_text",
            snippet="Preference-Bridged Alignment (PBA) improves personalization.",
            score=2.0,
        ),
    ]

    options = acronym_options_from_evidence(
        target="PBA",
        papers=papers,
        evidence=evidence,
        paper_lookup=lambda _: None,
    )

    assert [item["paper_id"] for item in options] == ["p2"]
    assert options[0]["meaning"] == "Preference-Bridged Alignment"
    assert "Preference-Bridged Alignment" in options[0]["context_text"]


def test_acronym_evidence_from_corpus_scores_expansions_and_formula_hints() -> None:
    paper_docs = [SimpleNamespace(metadata={"paper_id": "p1"})]
    block_docs = [
        SimpleNamespace(
            metadata={
                "doc_id": "b1",
                "title": "AlignX",
                "page": 3,
                "block_type": "page_text",
                "formula_hint": 1,
            },
            page_content="Preference-Bridged Alignment (PBA) defines the objective.",
        ),
        SimpleNamespace(
            metadata={"doc_id": "b2", "title": "AlignX", "page": 4, "block_type": "page_text"},
            page_content="No target here.",
        ),
    ]

    evidence = acronym_evidence_from_corpus(
        target="PBA",
        limit=5,
        paper_documents=lambda: paper_docs,
        block_documents_for_paper=lambda paper_id, limit: block_docs if paper_id == "p1" and limit == 320 else [],
    )

    assert [item.doc_id for item in evidence] == ["b1"]
    assert evidence[0].score == 9.0
    assert evidence[0].snippet.startswith("Preference-Bridged Alignment")


def test_finalize_acronym_disambiguation_options_filters_context_and_excluded_titles() -> None:
    options = [
        {"paper_id": "p1", "title": "Old Focus", "meaning": "First", "context_text": "First PBA"},
        {"paper_id": "p2", "title": "AlignX", "meaning": "Preference Bridged Alignment", "context_text": "PPAIR benchmark"},
        {"paper_id": "p3", "title": "Other", "meaning": "Policy Based Alignment", "context_text": "PPAIR comparison"},
    ]

    finalized = finalize_acronym_disambiguation_options(
        options=options,
        contract=QueryContract(clean_query="PBA in PPAIR", targets=["PBA", "PPAIR"]),
        target="PBA",
        excluded_titles={"old focus"},
    )

    assert [item["paper_id"] for item in finalized] == ["p2", "p3"]
    assert all(item["schema_version"] == CLARIFICATION_OPTION_SCHEMA_VERSION for item in finalized)
    assert finalize_acronym_disambiguation_options(
        options=options[:2],
        contract=QueryContract(clean_query="PBA in missing context", targets=["PBA", "missing"]),
        target="PBA",
        excluded_titles=set(),
    ) == []


def test_disambiguation_missing_fields_uses_ambiguous_slots() -> None:
    contract = QueryContract(clean_query="PBA是什么", notes=["ambiguous_slot=target"])

    assert disambiguation_missing_fields(contract) == ["target"]
    assert disambiguation_missing_fields(QueryContract(clean_query="PBA是什么")) == ["disambiguation"]


def test_contract_needs_evidence_disambiguation_for_acronym_goals_and_notes() -> None:
    assert contract_needs_evidence_disambiguation(
        QueryContract(clean_query="PBA是什么", targets=["PBA"], requested_fields=["definition"])
    )
    assert contract_needs_evidence_disambiguation(
        QueryContract(clean_query="PBA?", targets=["PBA"], notes=["ambiguous_slot=target"])
    )
    assert not contract_needs_evidence_disambiguation(
        QueryContract(clean_query="AlignX是什么", targets=["AlignX"], requested_fields=["definition"])
    )
    assert not contract_needs_evidence_disambiguation(QueryContract(clean_query="PBA?", targets=[]))


def test_evidence_disambiguation_options_skips_resolved_contract_without_loading_sources() -> None:
    contract = QueryContract(
        clean_query="PBA是什么",
        targets=["PBA"],
        requested_fields=["definition"],
        notes=["resolved_human_choice"],
    )

    def fail() -> list[dict[str, object]]:
        raise AssertionError("source callback should not be called")

    assert evidence_disambiguation_options(
        contract=contract,
        target_binding_exists=False,
        is_negative_correction=False,
        initial_options=fail,
        broad_options=fail,
        corpus_options=fail,
        excluded_titles=set(),
    ) == []


def test_evidence_disambiguation_options_respects_existing_target_binding() -> None:
    contract = QueryContract(clean_query="PBA是什么", targets=["PBA"], requested_fields=["definition"])

    def fail() -> list[dict[str, object]]:
        raise AssertionError("source callback should not be called")

    assert evidence_disambiguation_options(
        contract=contract,
        target_binding_exists=True,
        is_negative_correction=False,
        initial_options=fail,
        broad_options=fail,
        corpus_options=fail,
        excluded_titles=set(),
    ) == []


def test_evidence_disambiguation_options_prefers_broader_formula_options() -> None:
    calls: list[str] = []
    contract = QueryContract(clean_query="PBA公式是什么", targets=["PBA"], requested_fields=["formula"])

    def initial() -> list[dict[str, object]]:
        calls.append("initial")
        return [{"paper_id": "p1", "title": "AlignX", "meaning": "Preference Bridged Alignment"}]

    def broad() -> list[dict[str, object]]:
        calls.append("broad")
        return [
            {"paper_id": "p1", "title": "AlignX", "meaning": "Preference Bridged Alignment"},
            {"paper_id": "p2", "title": "Policy Gradient Notes", "meaning": "Policy Based Alignment"},
        ]

    def corpus() -> list[dict[str, object]]:
        raise AssertionError("corpus fallback should not be needed")

    options = evidence_disambiguation_options(
        contract=contract,
        target_binding_exists=False,
        is_negative_correction=False,
        initial_options=initial,
        broad_options=broad,
        corpus_options=corpus,
        excluded_titles=set(),
    )

    assert calls == ["initial", "broad"]
    assert [item["paper_id"] for item in options] == ["p1", "p2"]
    assert {item["source"] for item in options} == {"evidence_disambiguation"}


def test_evidence_disambiguation_options_uses_corpus_formula_fallback() -> None:
    calls: list[str] = []
    contract = QueryContract(clean_query="PBA公式是什么", targets=["PBA"], requested_fields=["formula"])

    def initial() -> list[dict[str, object]]:
        calls.append("initial")
        return [{"paper_id": "p1", "title": "AlignX", "meaning": "Preference Bridged Alignment"}]

    def broad() -> list[dict[str, object]]:
        calls.append("broad")
        return [{"paper_id": "p1", "title": "AlignX", "meaning": "Preference Bridged Alignment"}]

    def corpus() -> list[dict[str, object]]:
        calls.append("corpus")
        return [
            {"paper_id": "p1", "title": "AlignX", "meaning": "Preference Bridged Alignment"},
            {"paper_id": "p2", "title": "Policy Gradient Notes", "meaning": "Policy Based Alignment"},
        ]

    options = evidence_disambiguation_options(
        contract=contract,
        target_binding_exists=False,
        is_negative_correction=False,
        initial_options=initial,
        broad_options=broad,
        corpus_options=corpus,
        excluded_titles={"policy gradient notes"},
    )

    assert calls == ["initial", "broad", "corpus"]
    assert options == []


def test_clarification_tracking_helpers_manage_attempts() -> None:
    session = SessionContext(session_id="s1")
    contract = QueryContract(clean_query="PBA是什么", relation="entity_definition", targets=["PBA"])
    verification = VerificationReport(status="clarify", missing_fields=["target"], recommended_action="clarify_ambiguous_entity")
    key = clarification_tracking_key(
        contract=contract,
        verification=verification,
        options=[{"option_id": "pba-alignx"}],
    )

    assert next_clarification_attempt(session=session, key=key) == 1
    remember_clarification_attempt(session=session, key=key)
    assert session.last_clarification_key == key
    assert session.clarification_attempts == 1
    assert next_clarification_attempt(session=session, key=key) == 2
    remember_clarification_attempt(session=session, key=key)
    assert session.clarification_attempts == 2
    reset_clarification_tracking(session)
    assert session.last_clarification_key == ""
    assert session.clarification_attempts == 0


def test_pending_clarification_helpers_store_and_clear_session_state() -> None:
    session = SessionContext(session_id="s1")
    contract = QueryContract(clean_query="PBA是什么", targets=["PBA"])
    options = [{"option_id": "pba-alignx", "title": "AlignX"}]

    store_pending_clarification(session=session, contract=contract, options=options)
    assert session.pending_clarification_type == "ambiguity"
    assert session.pending_clarification_target == "PBA"
    assert session.pending_clarification_options == options

    clear_pending_clarification(session)
    assert session.pending_clarification_type == ""
    assert session.pending_clarification_target == ""
    assert session.pending_clarification_options == []


def test_contract_from_pending_clarification_resolves_choice_payload_and_text() -> None:
    session = SessionContext(
        session_id="s1",
        pending_clarification_type="ambiguity",
        pending_clarification_target="PBA",
        pending_clarification_options=[
            {"option_id": "first", "target": "PBA", "meaning": "First Meaning", "paper_id": "p1"},
            {"option_id": "second", "target": "PBA", "meaning": "Second Meaning", "paper_id": "p2"},
        ],
    )

    by_payload = contract_from_pending_clarification(
        clean_query="选择第二个",
        session=session,
        clarification_choice={"option_id": "second"},
    )
    by_text = contract_from_pending_clarification(clean_query="我说第二个", session=session)

    assert by_payload is not None
    assert by_payload.targets == ["PBA"]
    assert "selected_paper_id=p2" in by_payload.notes
    assert by_text is not None
    assert "selected_paper_id=p2" in by_text.notes
    assert contract_from_pending_clarification(clean_query="没有明确选择", session=session) is None


def test_ambiguity_clarification_question_renders_contract_or_pending_options() -> None:
    contract = contract_with_ambiguity_options(
        contract=QueryContract(clean_query="PBA是什么", relation="entity_definition", targets=["PBA"]),
        options=[
            {
                "title": "AlignX",
                "year": "2025",
                "meaning": "Preference Bridged Alignment",
                "display_label": "推荐候选",
                "display_reason": "title match",
            }
        ],
    )

    question = ambiguity_clarification_question(contract=contract, session=SessionContext(session_id="s1"))

    assert "`PBA` 在本地论文库里有多个可能含义" in question
    assert "推荐候选：Preference Bridged Alignment，见《AlignX》（2025）：title match" in question

    pending = SessionContext(
        session_id="s1",
        pending_clarification_type="ambiguity",
        pending_clarification_target="PBA",
        pending_clarification_options=[{"title": "AlignX", "meaning": "Preference Bridged Alignment"}],
    )
    assert ambiguity_clarification_question(contract=QueryContract(clean_query="继续", targets=["PBA"]), session=pending)
    assert ambiguity_clarification_question(contract=QueryContract(clean_query="继续", targets=["DPO"]), session=pending) == ""
