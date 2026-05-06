from __future__ import annotations

from typing import Any, Callable

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, SessionContext, VerificationReport
from app.services.agent.runtime_helpers import claim_focus_titles, reflect_agent_state_decision
from app.services.clarification.intents import acronym_options_from_evidence as build_acronym_options_from_evidence
from app.services.contracts.conversation_memory import target_binding_from_memory


EmitFn = Callable[[str, dict[str, Any]], None]


def agent_reflect(
    *,
    agent: Any,
    state: dict[str, Any],
    session: SessionContext,
    emit: EmitFn,
    execution_steps: list[dict[str, Any]],
) -> None:
    verification = state.get("verification")
    if not isinstance(verification, VerificationReport):
        verification = VerificationReport(
            status="clarify",
            missing_fields=["verified_claims"],
            recommended_action="clarify_after_reflection",
        )
        state["verification"] = verification
    contract: QueryContract = state["contract"]
    claims: list[Claim] = state["claims"]
    papers: list[CandidatePaper] = state["screened_papers"]
    evidence: list[EvidenceBlock] = state["evidence"]

    def paper_title_lookup(paper_id: str) -> str | None:
        doc = agent.retriever.paper_doc_by_id(paper_id)
        if doc is None:
            return None
        return str((doc.metadata or {}).get("title", ""))

    focus_titles = claim_focus_titles(claims=claims, papers=papers, paper_title_lookup=paper_title_lookup)
    target = str(contract.targets[0] or "").strip() if contract.targets else ""
    reflection = reflect_agent_state_decision(
        contract=contract,
        claims=claims,
        focus_titles=focus_titles,
        verification=verification,
        excluded_titles=state["excluded_titles"],
        target_binding_exists=bool(target and target_binding_from_memory(session=session, target=target)),
        ambiguity_option_count=lambda: len(
            build_acronym_options_from_evidence(
                target=target,
                papers=papers,
                evidence=evidence,
                paper_lookup=agent._candidate_from_paper_id,
            )
        ),
    )
    if reflection.get("decision") == "clarify":
        state["verification"] = VerificationReport(
            status="clarify",
            missing_fields=[str(item) for item in reflection.get("missing_fields", ["agent_reflection"])],
            recommended_action=str(reflection.get("recommended_action", "clarify_after_reflection")),
        )
    state["reflection"] = reflection
    emit("reflection", reflection)
    execution_steps.append({"node": "agent_reflection", "summary": str(reflection.get("decision", state["verification"].status))})
