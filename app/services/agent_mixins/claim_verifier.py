from __future__ import annotations

from app.domain.models import (
    CandidatePaper,
    Claim,
    EvidenceBlock,
    QueryContract,
    ResearchPlan,
    VerificationReport,
)
from app.services.claims.verification_helpers import (
    CLAIM_VERIFIER_MARKERS,
    claim_value_looks_like_formula,
    formula_claim_matches_target,
    formula_evidence_supports_target,
    is_identity_alias_match,
    is_initialism_alias_match,
    looks_like_metric_verification_goal,
    paper_identity_matches_targets,
    targets_supported,
    verification_goals,
)
from app.services.claims.type_verifiers import (
    origin_claim_has_intro_support,
    verify_concept_definition_claims,
    verify_entity_definition_claims,
    verify_figure_question_claims,
    verify_followup_research_claims,
    verify_formula_lookup_claims,
    verify_general_question_claims,
    verify_metric_value_lookup_claims,
    verify_origin_lookup_claims,
    verify_paper_recommendation_claims,
    verify_topology_recommendation_claims,
)
from app.services.claims.verifier_pipeline import verify_claims_with_generic_fallback
from app.services.claims.llm_verifier import (
    coerce_verifier_string_list,
    verify_claims_with_schema_llm,
    verify_formula_claims_with_llm,
)


class ClaimVerifierMixin:
    def _verify_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport:
        if not claims:
            return VerificationReport(status="retry", missing_fields=plan.required_claims, recommended_action="expand_recall")
        # ── Deterministic evidence-id audit ──
        # Every claim must cite evidence_ids that exist in the actual evidence
        # or paper document id lists.  Claims that cite fabricated ids are
        # flagged BEFORE any LLM verification runs.
        real_doc_ids: set[str] = {str(item.doc_id or "").strip() for item in evidence if str(item.doc_id or "").strip()}
        real_doc_ids |= {str(item.paper_id or "").strip() for item in papers if str(item.paper_id or "").strip()}
        # Also collect block_doc_ids from papers if available
        for paper in papers:
            for doc_id in list(getattr(paper, "doc_ids", []) or []):
                real_doc_ids.add(str(doc_id).strip())
        real_doc_ids.discard("")
        orphan_claims: list[str] = []
        for claim in claims:
            cited_ids = {str(eid or "").strip() for eid in list(claim.evidence_ids or [])}
            cited_ids.discard("")
            if cited_ids and not cited_ids.issubset(real_doc_ids):
                orphan_claims.append(
                    f"{claim.claim_type or 'unknown'}: {str(claim.entity or claim.value or '')[:80]}"
                )
            # P0-3: Also audit claim.paper_ids against real_doc_ids (white-list)
            cited_paper_ids = {str(pid or "").strip() for pid in list(claim.paper_ids or [])}
            cited_paper_ids.discard("")
            if cited_paper_ids and not cited_paper_ids.issubset(real_doc_ids):
                orphan_claims.append(
                    f"{claim.claim_type or 'unknown'}: paper_id {str(list(cited_paper_ids - real_doc_ids))}"
                )
        if orphan_claims:
            return VerificationReport(
                status="clarify",
                unsupported_claims=orphan_claims,
                missing_fields=[f"evidence_not_found_for_{len(orphan_claims)}_claims"],
                recommended_action=(
                    "Some claims cite document IDs that do not exist in the current evidence. "
                    "Re-run search with broader queries or ask the user for clarification."
                ),
            )
        # ── End evidence-id audit ──
        if contract.allow_web_search and any(
            item.claim_type == "web_research" and item.evidence_ids
            for item in claims
        ):
            return VerificationReport(status="pass")
        schema_report = self._verify_claims_with_schema(
            contract=contract,
            plan=plan,
            claims=claims,
            papers=papers,
            evidence=evidence,
        )
        if schema_report is not None:
            return schema_report
        report = self._verify_claims_with_generic_fallback(
            contract=contract,
            plan=plan,
            claims=claims,
            papers=papers,
            evidence=evidence,
        )
        return report or VerificationReport(status="pass")

    def _verify_claims_with_schema(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_claims_with_schema_llm(
            clients=self.clients,
            contract=contract,
            plan=plan,
            claims=claims,
            papers=papers,
            evidence=evidence,
        )

    def _verify_claims_with_generic_fallback(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        goals = self._verification_goals(contract=contract, plan=plan)
        return verify_claims_with_generic_fallback(
            contract=contract,
            plan=plan,
            claims=claims,
            papers=papers,
            evidence=evidence,
            goals=goals,
            checks={
                "origin": self._verify_origin_lookup_claims,
                "entity": self._verify_entity_definition_claims,
                "followup": self._verify_followup_research_claims,
                "paper_recommendation": self._verify_paper_recommendation_claims,
                "topology": self._verify_topology_recommendation_claims,
                "figure": self._verify_figure_question_claims,
                "metric": self._verify_metric_value_lookup_claims,
                "formula": self._verify_formula_lookup_claims,
                "general": self._verify_general_question_claims,
                "concept": self._verify_concept_definition_claims,
            },
        )

    @staticmethod
    def _verification_goals(*, contract: QueryContract, plan: ResearchPlan) -> set[str]:
        return verification_goals(contract=contract, plan=plan)

    @staticmethod
    def _looks_like_metric_verification_goal(query: str, goals: set[str]) -> bool:
        return looks_like_metric_verification_goal(query, goals)

    def _verify_origin_lookup_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_origin_lookup_claims(
            contract=contract,
            claims=claims,
            papers=papers,
            evidence=evidence,
            origin_supports_claim=lambda contract, claim, papers, evidence: self._origin_claim_has_intro_support(
                contract=contract,
                claim=claim,
                papers=papers,
                evidence=evidence,
            ),
        )

    def _origin_claim_has_intro_support(
        self,
        *,
        contract: QueryContract,
        claim: Claim,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> bool:
        return origin_claim_has_intro_support(
            contract=contract,
            claim=claim,
            papers=papers,
            evidence=evidence,
            paper_lookup=self._candidate_from_paper_id,
            paper_doc_lookup=self.retriever.paper_doc_by_id,
        )

    def _verify_entity_definition_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_entity_definition_claims(
            contract=contract,
            claims=claims,
            papers=papers,
            evidence=evidence,
            targets_supported_fn=lambda targets, papers, evidence: self._targets_supported(
                targets=targets,
                papers=papers,
                evidence=evidence,
            ),
        )

    def _verify_followup_research_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_followup_research_claims(claims=claims)

    def _verify_paper_recommendation_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_paper_recommendation_claims(claims=claims)

    def _verify_topology_recommendation_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_topology_recommendation_claims(claims=claims, evidence=evidence)

    def _verify_figure_question_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_figure_question_claims(
            contract=contract,
            claims=claims,
            papers=papers,
            paper_identity_matches_targets=lambda paper, targets: self._paper_identity_matches_targets(
                paper=paper,
                targets=targets,
            ),
        )

    def _verify_metric_value_lookup_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_metric_value_lookup_claims(claims=claims)

    def _verify_formula_lookup_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_formula_lookup_claims(
            contract=contract,
            claims=claims,
            papers=papers,
            evidence=evidence,
            claim_value_looks_like_formula=lambda value: self._claim_value_looks_like_formula(value),
            verify_formula_claims_with_llm=lambda contract, claims, papers, evidence: self._verify_formula_claims_with_llm(
                contract=contract,
                claims=claims,
                papers=papers,
                evidence=evidence,
            ),
            formula_claim_matches_target=lambda contract, claim, papers, evidence: self._formula_claim_matches_target(
                contract=contract,
                claim=claim,
                papers=papers,
                evidence=evidence,
            ),
        )

    def _verify_formula_claims_with_llm(
        self,
        *,
        contract: QueryContract,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_formula_claims_with_llm(
            clients=self.clients,
            contract=contract,
            claims=claims,
            papers=papers,
            evidence=evidence,
        )

    @staticmethod
    def _coerce_verifier_string_list(value: object) -> list[str]:
        return coerce_verifier_string_list(value)

    @staticmethod
    def _claim_value_looks_like_formula(value: str) -> bool:
        return claim_value_looks_like_formula(value)

    def _formula_claim_matches_target(
        self,
        *,
        contract: QueryContract,
        claim: Claim,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> bool:
        return formula_claim_matches_target(
            contract=contract,
            claim=claim,
            papers=papers,
            evidence=evidence,
        )

    def _formula_evidence_supports_target(self, *, target: str, evidence: list[EvidenceBlock]) -> bool:
        return formula_evidence_supports_target(target=target, evidence=evidence)

    def _verify_concept_definition_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_concept_definition_claims(
            contract=contract,
            claims=claims,
            papers=papers,
            evidence=evidence,
            targets_supported_fn=lambda targets, papers, evidence: self._targets_supported(
                targets=targets,
                papers=papers,
                evidence=evidence,
            ),
        )

    def _verify_general_question_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        return verify_general_question_claims(
            contract=contract,
            papers=papers,
            evidence=evidence,
            targets_supported_fn=lambda targets, papers, evidence: self._targets_supported(
                targets=targets,
                papers=papers,
                evidence=evidence,
            ),
        )

    def _targets_supported(
        self,
        *,
        targets: list[str],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> bool:
        return targets_supported(targets=targets, papers=papers, evidence=evidence)

    def _paper_identity_matches_targets(self, *, paper: CandidatePaper, targets: list[str]) -> bool:
        return paper_identity_matches_targets(
            paper=paper,
            targets=targets,
            canonicalize_target=self.retriever.canonicalize_target,
            normalize_entity_text=self.retriever._normalize_entity_text,
        )

    @staticmethod
    def _is_identity_alias_match(*, candidate: str, target: str) -> bool:
        return is_identity_alias_match(candidate=candidate, target=target)

    @staticmethod
    def _is_initialism_alias_match(*, candidate_name: str, target: str) -> bool:
        return is_initialism_alias_match(candidate_name=candidate_name, target=target)
