from __future__ import annotations

import re
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, SessionContext
from app.services.entity_definition_helpers import entity_definition_claim, entity_definition_evidence_ids
from app.services.followup_candidate_helpers import selected_followup_candidate_title
from app.services.followup_claim_helpers import followup_research_claim
from app.services import formula_text_helpers as formula_helpers
from app.services import metric_text_helpers as metric_helpers
from app.services import origin_selection_helpers as origin_helpers
from app.services.confidence import coerce_confidence_value
from app.services.evidence_presentation import (
    build_figure_contexts,
    evidence_ids_for_paper,
    extract_topology_terms,
    figure_fallback_summary,
    formula_terms,
    safe_year,
)
from app.services.figure_intents import extract_figure_benchmarks, figure_signal_score
from app.services.intent_marker_matching import MarkerProfile, query_matches_any
from app.services.paper_claim_helpers import default_text_claims, paper_recommendation_claim, paper_summary_claims
from app.services.query_shaping import matches_target
from app.services.schema_claim_helpers import (
    claims_from_schema_payload,
    schema_claim_human_prompt,
    schema_claim_system_prompt,
    should_use_schema_claim_solver,
)
from app.services.solver_goal_helpers import append_unique_claims, claim_goals
from app.services.topology_recommendation_helpers import (
    is_unusable_topology_recommendation_text,
    topology_discovery_claim,
    topology_recommendation_from_payload,
    topology_recommendation_human_prompt,
    topology_recommendation_claim,
    topology_recommendation_system_prompt,
)
from app.services.visual_claim_helpers import (
    figure_conclusion_claim_from_vlm_payload,
    figure_conclusion_text_claim,
    figure_vlm_human_content,
    figure_vlm_system_prompt,
    table_metric_claim_from_vlm_payload,
    table_vlm_human_content,
    table_vlm_system_prompt,
)


SOLVER_PIPELINE_MARKERS: dict[str, MarkerProfile] = {
    "origin_intro": ("introduce", "introduces", "introduced", "propose", "proposes", "proposed"),
}


class SolverPipelineMixin:
    def _run_solvers(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
        use_web_search: bool,
        max_web_results: int,
    ) -> list[Claim]:
        if self._should_use_schema_claim_solver(contract=contract, plan=plan):
            schema_claims = self._solve_claims_with_schema(
                contract=contract,
                plan=plan,
                papers=papers,
                evidence=evidence,
                session=session,
            )
            if schema_claims:
                return schema_claims
        _ = (use_web_search, max_web_results)
        return self._solve_claims_with_deterministic_fallback(
            contract=contract,
            plan=plan,
            papers=papers,
            evidence=evidence,
            session=session,
        )

    @classmethod
    def _should_use_schema_claim_solver(cls, *, contract: QueryContract, plan: ResearchPlan) -> bool:
        return should_use_schema_claim_solver(contract=contract, plan=plan)

    def _solve_claims_with_schema(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        if self.clients.chat is None or not evidence:
            return []
        payload = self.clients.invoke_json(
            system_prompt=schema_claim_system_prompt(),
            human_prompt=schema_claim_human_prompt(
                contract=contract,
                plan=plan,
                papers=papers,
                evidence=evidence,
                conversation_context=self._session_conversation_context(session, max_chars=12000),
            ),
            fallback={},
        )
        return claims_from_schema_payload(payload, contract=contract, papers=papers, evidence=evidence)

    def _solve_claims_with_deterministic_fallback(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        goals = claim_goals(contract=contract, plan=plan)
        claims: list[Claim] = []

        if goals & {"paper_title", "year", "origin"}:
            append_unique_claims(claims, self._solve_origin_lookup_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if "formula" in goals:
            append_unique_claims(claims, self._solve_formula(contract=contract, papers=papers, evidence=evidence))
        if goals & {"followup_papers", "candidate_relationship", "strict_followup"}:
            append_unique_claims(claims, self._solve_followup_research(contract=contract, papers=papers, evidence=evidence, session=session))
        if "figure_conclusion" in goals or "figure" in contract.required_modalities:
            append_unique_claims(claims, self._solve_figure(contract=contract, papers=papers, evidence=evidence))
        if "recommended_papers" in goals:
            append_unique_claims(claims, self._solve_paper_recommendation_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if goals & {"best_topology", "langgraph_recommendation"}:
            append_unique_claims(claims, self._solve_topology_recommendation_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if goals & {"relevant_papers", "topology_types"}:
            append_unique_claims(claims, self._solve_topology_discovery_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if goals & {"summary", "results", "key_findings"}:
            append_unique_claims(claims, self._solve_paper_summary_results_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if "reward_model_requirement" in goals:
            append_unique_claims(claims, self._solve_default_text_answer(contract=contract, papers=papers, evidence=evidence, session=session))
        if goals & {"entity_type", "role_in_context"}:
            append_unique_claims(claims, self._solve_entity_definition_text(contract=contract, papers=papers, evidence=evidence, session=session))
        elif goals & {"definition", "mechanism", "examples"}:
            append_unique_claims(claims, self._solve_concept_definition_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if goals & {"metric_value", "setting"}:
            append_unique_claims(claims, self._solve_table(contract=contract, papers=papers, evidence=evidence))
            if not any(claim.claim_type == "metric_value" for claim in claims):
                append_unique_claims(claims, self._solve_metric_context_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if not claims:
            append_unique_claims(claims, self._solve_default_text_answer(contract=contract, papers=papers, evidence=evidence, session=session))
        return claims

    def _solve_text(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        session: SessionContext,
        evidence: list[EvidenceBlock] | None = None,
    ) -> list[Claim]:
        plan = ResearchPlan(required_claims=list(contract.requested_fields or ["answer"]))
        return self._solve_claims_with_deterministic_fallback(
            contract=contract,
            plan=plan,
            papers=papers,
            evidence=evidence or [],
            session=session,
        )

    def _solve_origin_lookup_text(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        session: SessionContext,
        evidence: list[EvidenceBlock] | None = None,
    ) -> list[Claim]:
        selected = self._select_origin_paper(contract=contract, papers=papers, evidence=evidence)
        if selected is None:
            return []
        supporting_ids = evidence_ids_for_paper(evidence or [], selected.paper_id, limit=2)
        return [origin_helpers.origin_lookup_claim(contract=contract, paper=selected, evidence_ids=supporting_ids)]

    def _solve_entity_definition_text(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        paper, supporting_evidence = self._select_entity_supporting_paper(
            contract=contract,
            papers=papers,
            evidence=evidence,
        )
        if paper is None:
            return []
        relevant_evidence = supporting_evidence or [item for item in evidence if item.paper_id == paper.paper_id][:4]
        label = self._infer_entity_type(contract=contract, papers=[paper], evidence=relevant_evidence)
        return [
            entity_definition_claim(
                contract=contract,
                paper=paper,
                label=label,
                evidence_ids=entity_definition_evidence_ids(
                    contract=contract,
                    paper=paper,
                    evidence=relevant_evidence,
                    target_matcher=matches_target,
                ),
                definition_lines=self._entity_supporting_lines(relevant_evidence, kind="definition"),
                mechanism_lines=self._entity_supporting_lines(relevant_evidence, kind="mechanism"),
                application_lines=self._entity_supporting_lines(relevant_evidence, kind="application"),
            )
        ]

    def _solve_concept_definition_text(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        claim = self._build_concept_definition_claim(contract=contract, papers=papers, evidence=evidence)
        return [claim] if claim is not None else []

    def _solve_topology_discovery_text(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        topology_terms = extract_topology_terms(evidence)
        claim = topology_discovery_claim(
            papers=papers,
            topology_terms=topology_terms,
            evidence_ids_for_paper=lambda paper_id: evidence_ids_for_paper(evidence, paper_id, limit=2),
        )
        return [claim] if claim is not None else []

    def _solve_topology_recommendation_text(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        topology_terms = extract_topology_terms(evidence)
        if not topology_terms and not evidence:
            return []
        recommendation = self._derive_topology_recommendation(evidence=evidence, topology_terms=topology_terms)
        return [topology_recommendation_claim(recommendation=recommendation, topology_terms=topology_terms, evidence=evidence)]

    def _solve_paper_summary_results_text(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        if not papers:
            return []
        if contract.targets:
            focused = [
                paper
                for paper in papers
                if self._paper_identity_matches_targets(paper=paper, targets=contract.targets)
                or origin_helpers.paper_has_origin_intro_support(paper=paper, targets=contract.targets)
            ]
            if focused:
                papers = focused
        return paper_summary_claims(
            entity=contract.targets[0] if contract.targets else "",
            papers=papers,
            metric_lines=self._extract_metric_lines(evidence),
            summary_for_paper=self._paper_summary_text,
            evidence_ids_for_paper=lambda paper_id, limit: evidence_ids_for_paper(evidence, paper_id, limit=limit),
        )

    def _solve_paper_recommendation_text(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        claim = paper_recommendation_claim(
            entity=contract.targets[0] if contract.targets else contract.clean_query,
            papers=papers,
            reason_for_paper=self._paper_recommendation_reason,
        )
        return [claim] if claim is not None else []

    def _solve_metric_context_text(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        if not papers and not evidence:
            return []
        metric_evidence = metric_helpers.ranked_metric_context_evidence(
            contract=contract,
            papers=papers,
            evidence=evidence,
            token_weights=self.settings.solver_metric_token_weights,
            paper_target_matcher=lambda paper, targets: self._paper_identity_matches_targets(paper=paper, targets=targets),
        )
        selected_paper, selected_papers, paper_ids = metric_helpers.metric_paper_selection(
            papers=papers,
            ranked_evidence=metric_evidence,
        )
        if selected_paper is None:
            return []
        return [
            metric_helpers.metric_context_claim(
                entity=contract.targets[0] if contract.targets else selected_paper.title,
                selected_paper=selected_paper,
                selected_papers=selected_papers,
                metric_lines=self._extract_metric_lines(metric_evidence or evidence),
                metric_evidence=metric_evidence,
                fallback_evidence_ids=evidence_ids_for_paper(evidence, selected_paper.paper_id, limit=4),
                paper_ids=paper_ids or [selected_paper.paper_id],
            )
        ]

    def _solve_default_text_answer(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        if not papers:
            return []
        return default_text_claims(
            entity=contract.targets[0] if contract.targets else "",
            papers=papers,
            summary_for_paper=self._paper_summary_text,
            evidence_ids_for_paper=lambda paper_id, limit: evidence_ids_for_paper(evidence, paper_id, limit=limit),
        )

    def _select_origin_paper(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> CandidatePaper | None:
        if not papers:
            return None
        candidate_pool = list(papers)
        paper_by_id = {item.paper_id: item for item in candidate_pool}
        for paper in self._origin_candidates_from_corpus(contract=contract):
            if paper.paper_id not in paper_by_id:
                paper_by_id[paper.paper_id] = paper
                candidate_pool.append(paper)
        if contract.targets:
            identity_matched = [
                item
                for item in candidate_pool
                if self._paper_identity_matches_targets(paper=item, targets=contract.targets)
                or origin_helpers.paper_has_origin_intro_support(paper=item, targets=contract.targets)
            ]
            if identity_matched:
                candidate_pool = identity_matched
        if not evidence:
            return self._pick_origin_paper_with_intro_support(contract=contract, papers=candidate_pool)
        target_aliases = origin_helpers.origin_target_aliases(contract.targets)
        scored: list[tuple[float, float, CandidatePaper]] = []
        for paper in candidate_pool:
            support = [item for item in evidence if item.paper_id == paper.paper_id]
            if target_aliases:
                support = [
                    item
                    for item in support
                    if any(
                        matches_target(haystack, alias)
                        for alias in target_aliases
                        for haystack in [item.snippet, item.caption, item.title]
                        if haystack
                    )
                ]
            score = 0.0
            intro_score = 0.0
            for item in support:
                score += float(item.score)
                snippet = item.snippet.lower()
                if query_matches_any(snippet, "", SOLVER_PIPELINE_MARKERS["origin_intro"]):
                    score += 1.5
                if " is a " in snippet or " is an " in snippet:
                    score += 0.8
                intro_score += origin_helpers.origin_target_intro_score(item.snippet, target_aliases)
                score += origin_helpers.origin_target_definition_score(item.snippet, target_aliases)
            if target_aliases:
                paper_text = origin_helpers.origin_paper_text(paper)
                if any(matches_target(paper_text, alias) for alias in target_aliases):
                    score += 0.8
                intro_score += origin_helpers.origin_target_intro_score(paper_text, target_aliases)
                score += origin_helpers.origin_target_definition_score(paper_text, target_aliases)
            scored.append((intro_score, score, paper))
        scored.sort(key=lambda item: (-item[0], -item[1], safe_year(item[2].year), -item[2].score, item[2].title))
        if scored and scored[0][0] > 0:
            return scored[0][2]
        return self._pick_origin_paper_with_intro_support(contract=contract, papers=candidate_pool)

    def _origin_candidates_from_corpus(self, *, contract: QueryContract) -> list[CandidatePaper]:
        aliases = origin_helpers.origin_target_aliases(contract.targets)
        if not aliases:
            return []
        candidates: list[tuple[float, CandidatePaper]] = []
        for doc in self.retriever.paper_documents():
            meta = dict(doc.metadata or {})
            paper_id = str(meta.get("paper_id", "") or "").strip()
            if not paper_id:
                continue
            paper = self._candidate_from_paper_id(paper_id)
            if paper is None:
                continue
            text = "\n".join(
                [
                    str(doc.page_content or ""),
                    str(meta.get("aliases", "")),
                    str(meta.get("abstract_note", "")),
                    str(meta.get("generated_summary", "")),
                ]
            )
            score = origin_helpers.origin_target_intro_score(text, aliases)
            if score <= 0:
                continue
            candidates.append((score, paper.model_copy(update={"score": max(float(paper.score), score)})))
        candidates.sort(key=lambda item: (-item[0], safe_year(item[1].year), item[1].title))
        return [paper for _, paper in candidates[:8]]

    def _pick_origin_paper_with_intro_support(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
    ) -> CandidatePaper | None:
        aliases = origin_helpers.origin_target_aliases(contract.targets)
        if not aliases:
            return origin_helpers.pick_origin_paper(papers)
        scored = [
            (origin_helpers.origin_target_intro_score(origin_helpers.origin_paper_text(paper), aliases), paper)
            for paper in papers
        ]
        scored = [(score, paper) for score, paper in scored if score > 0]
        scored.sort(key=lambda item: (-item[0], safe_year(item[1].year), -item[1].score, item[1].title))
        return scored[0][1] if scored else None

    def _solve_followup_research(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        session: SessionContext,
        evidence: list[EvidenceBlock] | None = None,
    ) -> list[Claim]:
        if not papers:
            return []
        seed_papers = self._resolve_followup_seed_papers(contract=contract, candidates=papers, session=session)
        selected_candidate_title = selected_followup_candidate_title(contract)
        candidate_pool = self._expand_followup_candidate_pool(
            contract=contract,
            seed_papers=seed_papers,
            initial_candidates=papers,
        )
        followups = self._rank_followup_candidates(
            contract=contract,
            seed_papers=seed_papers,
            candidates=candidate_pool,
            evidence=evidence or [],
        )
        if not followups:
            return []
        return [
            followup_research_claim(
                entity=contract.targets[0] if contract.targets else "",
                seed_papers=seed_papers,
                followups=followups,
                selected_candidate_title=selected_candidate_title,
            )
        ]

    def _solve_formula(self, *, contract: QueryContract, papers: list[CandidatePaper], evidence: list[EvidenceBlock]) -> list[Claim]:
        if not papers:
            return []
        claims: list[Claim] = []
        target_terms = formula_helpers.formula_target_terms(contract)
        for paper in papers:
            paper_evidence = [item for item in evidence if item.paper_id == paper.paper_id]
            if not paper_evidence:
                continue
            matched_targets = formula_helpers.formula_matched_targets(
                paper=paper,
                evidence=paper_evidence,
                target_terms=target_terms,
                target_matcher=matches_target,
            )
            if target_terms and not matched_targets:
                continue
            formula_blocks = formula_helpers.select_formula_blocks(
                paper_evidence,
                block_scorer=lambda text: self._formula_block_score(text, contract=contract),
            )
            formula_payload = self._extract_formula_claim_payload(
                contract=contract,
                formula_blocks=formula_blocks,
                fallback_evidence=paper_evidence,
            )
            claim = formula_helpers.formula_claim_from_payload(
                contract=contract,
                paper=paper,
                matched_targets=matched_targets,
                formula_payload=formula_payload,
                formula_blocks=formula_blocks,
                fallback_evidence_ids=evidence_ids_for_paper(evidence, paper.paper_id, limit=3),
                fallback_term_text="\n".join(item.snippet for item in paper_evidence[:3]),
                term_extractor=formula_terms,
            )
            if claim is None:
                continue
            claims.append(claim)
            if len(claims) >= 6:
                break
        return claims

    def _extract_formula_claim_payload(
        self,
        *,
        contract: QueryContract,
        formula_blocks: list[EvidenceBlock],
        fallback_evidence: list[EvidenceBlock],
    ) -> dict[str, Any]:
        selected_evidence = formula_blocks[:4] or fallback_evidence[:4]
        llm_payload = self._llm_extract_formula_claim_payload(contract=contract, evidence=selected_evidence)
        if llm_payload:
            return llm_payload
        return formula_helpers.fallback_formula_payload(selected_evidence, term_extractor=formula_terms)

    def _llm_extract_formula_claim_payload(self, *, contract: QueryContract, evidence: list[EvidenceBlock]) -> dict[str, Any]:
        if self.clients.chat is None or not evidence:
            return {}
        payload = self.clients.invoke_json(
            system_prompt=formula_helpers.formula_extractor_system_prompt(),
            human_prompt=formula_helpers.formula_extractor_human_prompt(contract=contract, evidence=evidence),
            fallback={},
        )
        return formula_helpers.llm_formula_payload_from_response(
            payload,
            allowed_evidence_ids={item.doc_id for item in evidence},
            term_extractor=formula_terms,
        )

    @staticmethod
    def _normalize_formula_variable_symbol(symbol: str) -> str:
        return formula_helpers.normalize_formula_variable_symbol(symbol)

    @classmethod
    def _normalize_extracted_formula_text(cls, text: str) -> str:
        return formula_helpers.normalize_extracted_formula_text(text)

    def _solve_table(self, *, contract: QueryContract, papers: list[CandidatePaper], evidence: list[EvidenceBlock]) -> list[Claim]:
        ranked_blocks = metric_helpers.ranked_table_metric_blocks(
            contract=contract,
            papers=papers,
            evidence=evidence,
            token_weights=self.settings.solver_metric_token_weights,
            paper_target_matcher=lambda paper, targets: self._paper_identity_matches_targets(paper=paper, targets=targets),
        )
        if not ranked_blocks:
            return []
        selected_paper, _, paper_ids = metric_helpers.metric_paper_selection(
            papers=papers,
            ranked_evidence=ranked_blocks,
        )
        if selected_paper is None:
            return []
        evidence_ids = [item.doc_id for item in ranked_blocks[:4]]
        vlm_claim = self._solve_table_with_vlm(
            contract=contract,
            ranked_blocks=ranked_blocks,
            evidence_ids=evidence_ids,
            paper_ids=paper_ids or [selected_paper.paper_id],
            selected_paper=selected_paper,
        )
        if vlm_claim is not None:
            return [vlm_claim]
        lines = self._extract_metric_lines(ranked_blocks)
        return [
            metric_helpers.text_table_metric_claim(
                entity=contract.targets[0] if contract.targets else selected_paper.title,
                metric_lines=lines,
                evidence_ids=evidence_ids,
                paper_ids=paper_ids or [selected_paper.paper_id],
                selected_paper=selected_paper,
            )
        ]

    def _solve_table_with_vlm(
        self,
        *,
        contract: QueryContract,
        ranked_blocks: list[EvidenceBlock],
        evidence_ids: list[str],
        paper_ids: list[str],
        selected_paper: CandidatePaper,
    ) -> Claim | None:
        if not getattr(self.settings, "enable_table_vlm", False):
            return None
        content = table_vlm_human_content(
            contract=contract,
            ranked_blocks=ranked_blocks,
            render_page_image=lambda file_path, page: self._render_page_image_data_url(file_path=file_path, page=page),
        )
        if not any(block.get("type") == "image_url" for block in content):
            return None
        payload = self.clients.invoke_multimodal_json(
            system_prompt=table_vlm_system_prompt(),
            human_content=content,
            fallback={"claims": [], "draft_answer": ""},
        )
        return table_metric_claim_from_vlm_payload(
            payload,
            entity=contract.targets[0] if contract.targets else selected_paper.title,
            evidence_ids=evidence_ids,
            paper_ids=paper_ids,
        )

    def _solve_figure(self, *, contract: QueryContract, papers: list[CandidatePaper], evidence: list[EvidenceBlock]) -> list[Claim]:
        figure_contexts = build_figure_contexts(evidence)
        if not figure_contexts:
            return []
        primary_context = figure_contexts[0]
        entity = contract.targets[0] if contract.targets else primary_context["title"]
        fallback_text = figure_fallback_summary(figure_contexts)
        evidence_benchmarks = extract_figure_benchmarks("\n".join(item.snippet for item in evidence[:10]))
        if len(evidence_benchmarks) >= 3:
            benchmark_suffix = " 图中提到的 benchmark 包括：" + "、".join(evidence_benchmarks) + "。"
            if benchmark_suffix not in fallback_text:
                remaining = max(80, 700 - len(benchmark_suffix))
                fallback_text = fallback_text[:remaining].rstrip() + benchmark_suffix
        payload = {"claims": [], "draft_answer": ""}
        if self.settings.enable_figure_vlm:
            content = figure_vlm_human_content(
                contract=contract,
                figure_contexts=figure_contexts,
                render_page_image=lambda file_path, page: self._render_page_image_data_url(file_path=file_path, page=page),
            )
            if any(block.get("type") == "image_url" for block in content):
                payload = self.clients.invoke_multimodal_json(
                    system_prompt=figure_vlm_system_prompt(),
                    human_content=content,
                    fallback={"claims": [], "draft_answer": ""},
                )
        vlm_claim = figure_conclusion_claim_from_vlm_payload(
            payload,
            entity=entity,
            evidence_ids=primary_context["doc_ids"],
            paper_id=primary_context["paper_id"],
            fallback_text=fallback_text,
            signal_score=figure_signal_score,
        )
        if vlm_claim is not None:
            return [vlm_claim]
        text_summary = self._summarize_figure_text(contract=contract, fallback_text=fallback_text, evidence=evidence)
        if text_summary:
            return [
                figure_conclusion_text_claim(
                    entity=entity,
                    text=text_summary,
                    figure_context=primary_context,
                    mode="text_summary",
                    confidence=0.82,
                )
            ]
        return [
            figure_conclusion_text_claim(
                entity=entity,
                text=fallback_text,
                figure_context=primary_context,
                mode="caption_fallback",
                confidence=0.74,
            )
        ]

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        return coerce_confidence_value(
            value,
            default=0.82,
            label_scores={"high": 0.88, "medium": 0.72, "low": 0.55},
        )

    def _derive_topology_recommendation(self, *, evidence: list[EvidenceBlock], topology_terms: list[str]) -> dict[str, str]:
        payload = self.clients.invoke_json(
            system_prompt=topology_recommendation_system_prompt(),
            human_prompt=topology_recommendation_human_prompt(topology_terms=topology_terms, evidence=evidence),
            fallback={},
        )
        return topology_recommendation_from_payload(payload, topology_terms=topology_terms)

    @staticmethod
    def _is_unusable_topology_recommendation_text(text: str) -> bool:
        return is_unusable_topology_recommendation_text(text)

    def _extract_metric_lines(self, evidence: list[EvidenceBlock]) -> list[str]:
        return metric_helpers.extract_metric_lines(evidence, token_weights=self.settings.solver_metric_token_weights)

    def _formula_block_score(self, text: str, *, contract: QueryContract | None = None) -> float:
        return formula_helpers.formula_block_score(
            text,
            query=contract.clean_query if contract is not None else None,
            token_weights=self.settings.retrieval_formula_token_weights,
        )

    @staticmethod
    def _formula_query_wants_gradient(query: str) -> bool:
        return formula_helpers.formula_query_wants_gradient(query)
