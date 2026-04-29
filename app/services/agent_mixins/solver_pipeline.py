from __future__ import annotations

import json
import re
from typing import Any

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, SessionContext
from app.services.followup_claim_helpers import followup_research_claim
from app.services import formula_text_helpers as formula_helpers
from app.services import metric_text_helpers as metric_helpers
from app.services import origin_selection_helpers as origin_helpers
from app.services.confidence import coerce_confidence_value
from app.services.paper_claim_helpers import default_text_claim, paper_recommendation_claim, paper_summary_claim
from app.services.prompt_safety import DOCUMENT_SAFETY_INSTRUCTION, wrap_untrusted_document_text
from app.services.schema_claim_helpers import claims_from_schema_payload, should_use_schema_claim_solver
from app.services.solver_goal_helpers import claim_goals, fallback_goals_from_query, looks_like_metric_goal
from app.services.topology_recommendation_helpers import (
    fallback_topology_recommendation,
    is_unusable_topology_recommendation_text,
    topology_discovery_claim,
    topology_recommendation_claim,
)
from app.services.visual_claim_helpers import (
    figure_conclusion_claim_from_vlm_payload,
    table_metric_claim_from_vlm_payload,
)


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
            system_prompt=(
                "你是论文研究助手的通用证据 Claim 抽取器。"
                "不要根据 relation 分支套模板。"
                "只基于输入 evidence 和 papers，输出 JSON："
                "{claims:[{claim_type, entity, value, structured_data, evidence_ids, paper_ids, confidence, required}]}。"
                "claim_type 应来自用户目标和 requested_fields，例如 definition/formula/metric_value/"
                "paper_summary/followup_research/recommendation/general_answer。"
                "每条 claim 必须引用能支撑它的 evidence_ids；证据不足就返回空 claims。"
                "不要编造 evidence 中不存在的论文、指标、公式或结论。"
                f"{DOCUMENT_SAFETY_INSTRUCTION}"
            ),
            human_prompt=json.dumps(
                {
                    "query": contract.clean_query,
                    "intent_adapter": {
                        "relation": contract.relation,
                        "targets": contract.targets,
                        "requested_fields": contract.requested_fields,
                        "answer_shape": contract.answer_shape,
                        "precision_requirement": contract.precision_requirement,
                        "notes": contract.notes,
                    },
                    "required_claims": plan.required_claims,
                    "papers": [
                        {
                            "paper_id": item.paper_id,
                            "title": item.title,
                            "year": item.year,
                        }
                        for item in papers[:8]
                    ],
                    "evidence": [
                        {
                            "doc_id": item.doc_id,
                            "paper_id": item.paper_id,
                            "title": item.title,
                            "page": item.page,
                            "block_type": item.block_type,
                            "caption": item.caption,
                            "snippet": wrap_untrusted_document_text(
                                item.snippet,
                                doc_id=item.doc_id,
                                title=item.title,
                                source=item.block_type or "pdf",
                                max_chars=1200,
                            ),
                        }
                        for item in evidence[:40]
                    ],
                    "conversation_context": self._session_conversation_context(session, max_chars=12000),
                },
                ensure_ascii=False,
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
        goals = self._claim_goals(contract=contract, plan=plan)
        claims: list[Claim] = []

        def extend_once(new_claims: list[Claim]) -> None:
            existing = {(claim.claim_type, claim.entity, claim.value) for claim in claims}
            for claim in new_claims:
                key = (claim.claim_type, claim.entity, claim.value)
                if key not in existing:
                    existing.add(key)
                    claims.append(claim)

        if goals & {"paper_title", "year", "origin"}:
            extend_once(self._solve_origin_lookup_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if "formula" in goals:
            extend_once(self._solve_formula(contract=contract, papers=papers, evidence=evidence))
        if goals & {"followup_papers", "candidate_relationship", "strict_followup"}:
            extend_once(self._solve_followup_research(contract=contract, papers=papers, evidence=evidence, session=session))
        if "figure_conclusion" in goals or "figure" in contract.required_modalities:
            extend_once(self._solve_figure(contract=contract, papers=papers, evidence=evidence))
        if "recommended_papers" in goals:
            extend_once(self._solve_paper_recommendation_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if goals & {"best_topology", "langgraph_recommendation"}:
            extend_once(self._solve_topology_recommendation_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if goals & {"relevant_papers", "topology_types"}:
            extend_once(self._solve_topology_discovery_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if goals & {"summary", "results", "key_findings"}:
            extend_once(self._solve_paper_summary_results_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if "reward_model_requirement" in goals:
            extend_once(self._solve_default_text_answer(contract=contract, papers=papers, evidence=evidence, session=session))
        if goals & {"entity_type", "role_in_context"}:
            extend_once(self._solve_entity_definition_text(contract=contract, papers=papers, evidence=evidence, session=session))
        elif goals & {"definition", "mechanism", "examples"}:
            extend_once(self._solve_concept_definition_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if goals & {"metric_value", "setting"}:
            extend_once(self._solve_table(contract=contract, papers=papers, evidence=evidence))
            if not any(claim.claim_type == "metric_value" for claim in claims):
                extend_once(self._solve_metric_context_text(contract=contract, papers=papers, evidence=evidence, session=session))
        if not claims:
            extend_once(self._solve_default_text_answer(contract=contract, papers=papers, evidence=evidence, session=session))
        return claims

    @staticmethod
    def _claim_goals(*, contract: QueryContract, plan: ResearchPlan) -> set[str]:
        return claim_goals(contract=contract, plan=plan)

    @staticmethod
    def _looks_like_metric_goal(query: str, goals: set[str]) -> bool:
        return looks_like_metric_goal(query, goals)

    @staticmethod
    def _fallback_goals_from_query(query: str, *, targets: list[str]) -> set[str]:
        return fallback_goals_from_query(query, targets=targets)

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
        supporting_ids = self._evidence_ids_for_paper(evidence, selected.paper_id, limit=2)
        if not supporting_ids and selected.doc_ids:
            supporting_ids = list(selected.doc_ids[:1])
        return [
            Claim(
                claim_type="origin",
                entity=self._origin_display_entity(contract=contract, paper=selected),
                value=selected.title,
                structured_data={"year": selected.year, "paper_title": selected.title},
                evidence_ids=supporting_ids,
                paper_ids=[selected.paper_id],
                confidence=0.94,
            )
        ]

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
        evidence_ids = [item.doc_id for item in relevant_evidence[:3]]
        if not evidence_ids:
            paper_text = "\n".join(
                [
                    paper.title,
                    str(paper.metadata.get("paper_card_text", "")),
                    str(paper.metadata.get("generated_summary", "")),
                    str(paper.metadata.get("abstract_note", "")),
                ]
            )
            if paper.doc_ids and (
                not contract.targets or any(self._matches_target(paper_text, target) for target in contract.targets if target)
            ):
                evidence_ids = list(paper.doc_ids[:1])
        return [
            Claim(
                claim_type="entity_definition",
                entity=contract.targets[0] if contract.targets else "",
                value=label,
                structured_data={
                    "paper_title": paper.title,
                    "description": "",
                    "definition_lines": self._entity_supporting_lines(relevant_evidence, kind="definition"),
                    "mechanism_lines": self._entity_supporting_lines(relevant_evidence, kind="mechanism"),
                    "application_lines": self._entity_supporting_lines(relevant_evidence, kind="application"),
                },
                evidence_ids=evidence_ids,
                paper_ids=[paper.paper_id],
                confidence=0.9,
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
        topology_terms = self._extract_topology_terms(evidence)
        claim = topology_discovery_claim(
            papers=papers,
            topology_terms=topology_terms,
            evidence_ids_for_paper=lambda paper_id: self._evidence_ids_for_paper(evidence, paper_id, limit=2),
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
        topology_terms = self._extract_topology_terms(evidence)
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
                or self._paper_has_origin_intro_support(paper=paper, targets=contract.targets)
            ]
            if focused:
                papers = focused
        claims: list[Claim] = []
        metric_bits = self._extract_metric_lines(evidence)
        for paper in papers[:4]:
            summary_text = self._paper_summary_text(paper.paper_id)
            evidence_ids = self._evidence_ids_for_paper(evidence, paper.paper_id, limit=4)
            if not summary_text and not evidence_ids:
                continue
            claims.append(
                paper_summary_claim(
                    entity=contract.targets[0] if contract.targets else paper.title,
                    paper=paper,
                    summary_text=summary_text,
                    metric_lines=metric_bits,
                    evidence_ids=evidence_ids,
                )
            )
        return claims

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
        paper_by_id = {paper.paper_id: paper for paper in papers}
        metric_evidence = sorted(
            [item for item in evidence if self._metric_line_score(item.snippet) > 0 or item.block_type in {"table", "caption"}],
            key=lambda item: (-self._metric_block_score(item=item, contract=contract, paper_by_id=paper_by_id), item.page, item.doc_id),
        )
        paper_ids = list(dict.fromkeys(item.paper_id for item in metric_evidence[:4] if item.paper_id))
        selected_papers = [paper_by_id[paper_id] for paper_id in paper_ids if paper_id in paper_by_id]
        selected_paper = selected_papers[0] if selected_papers else max(papers, key=lambda item: item.score) if papers else None
        if selected_paper is None:
            return []
        return [
            metric_helpers.metric_context_claim(
                entity=contract.targets[0] if contract.targets else selected_paper.title,
                selected_paper=selected_paper,
                selected_papers=selected_papers,
                metric_lines=self._extract_metric_lines(metric_evidence or evidence),
                metric_evidence=metric_evidence,
                fallback_evidence_ids=self._evidence_ids_for_paper(evidence, selected_paper.paper_id, limit=4),
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
        claims: list[Claim] = []
        for paper in papers[:4]:
            summary = self._paper_summary_text(paper.paper_id)
            evidence_ids = self._evidence_ids_for_paper(evidence, paper.paper_id, limit=3)
            if not summary and not evidence_ids:
                continue
            claims.append(
                default_text_claim(
                    entity=contract.targets[0] if contract.targets else paper.title,
                    paper=paper,
                    summary=summary,
                    evidence_ids=evidence_ids,
                )
            )
        return claims

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
                or self._paper_has_origin_intro_support(paper=item, targets=contract.targets)
            ]
            if identity_matched:
                candidate_pool = identity_matched
        if not evidence:
            return self._pick_origin_paper_with_intro_support(contract=contract, papers=candidate_pool)
        target_aliases = self._origin_target_aliases(contract.targets)
        scored: list[tuple[float, float, CandidatePaper]] = []
        for paper in candidate_pool:
            support = [item for item in evidence if item.paper_id == paper.paper_id]
            if target_aliases:
                support = [
                    item
                    for item in support
                    if any(
                        self._matches_target(haystack, alias)
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
                if any(token in snippet for token in ["introduce", "introduces", "introduced", "propose", "proposes", "proposed"]):
                    score += 1.5
                if " is a " in snippet or " is an " in snippet:
                    score += 0.8
                intro_score += self._origin_target_intro_score(item.snippet, target_aliases)
                score += self._origin_target_definition_score(item.snippet, target_aliases)
            if target_aliases:
                paper_text = self._origin_paper_text(paper)
                if any(self._matches_target(paper_text, alias) for alias in target_aliases):
                    score += 0.8
                intro_score += self._origin_target_intro_score(paper_text, target_aliases)
                score += self._origin_target_definition_score(paper_text, target_aliases)
            scored.append((intro_score, score, paper))
        scored.sort(key=lambda item: (-item[0], -item[1], self._safe_year(item[2].year), -item[2].score, item[2].title))
        if scored and scored[0][0] > 0:
            return scored[0][2]
        return self._pick_origin_paper_with_intro_support(contract=contract, papers=candidate_pool)

    def _origin_candidates_from_corpus(self, *, contract: QueryContract) -> list[CandidatePaper]:
        aliases = self._origin_target_aliases(contract.targets)
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
            score = self._origin_target_intro_score(text, aliases)
            if score <= 0:
                continue
            candidates.append((score, paper.model_copy(update={"score": max(float(paper.score), score)})))
        candidates.sort(key=lambda item: (-item[0], self._safe_year(item[1].year), item[1].title))
        return [paper for _, paper in candidates[:8]]

    def _pick_origin_paper_with_intro_support(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
    ) -> CandidatePaper | None:
        aliases = self._origin_target_aliases(contract.targets)
        if not aliases:
            return self._pick_origin_paper(papers)
        scored = [
            (self._origin_target_intro_score(self._origin_paper_text(paper), aliases), paper)
            for paper in papers
        ]
        scored = [(score, paper) for score, paper in scored if score > 0]
        scored.sort(key=lambda item: (-item[0], self._safe_year(item[1].year), -item[1].score, item[1].title))
        return scored[0][1] if scored else None

    def _paper_has_origin_intro_support(self, *, paper: CandidatePaper, targets: list[str]) -> bool:
        return origin_helpers.paper_has_origin_intro_support(paper=paper, targets=targets)

    @staticmethod
    def _origin_paper_text(paper: CandidatePaper) -> str:
        return origin_helpers.origin_paper_text(paper)

    @staticmethod
    def _origin_target_aliases(targets: list[str]) -> list[str]:
        return origin_helpers.origin_target_aliases(targets)

    def _origin_display_entity(self, *, contract: QueryContract, paper: CandidatePaper) -> str:
        return origin_helpers.origin_display_entity(targets=contract.targets, paper=paper)

    def _origin_target_intro_score(self, text: str, aliases: list[str]) -> float:
        return origin_helpers.origin_target_intro_score(text, aliases)

    def _origin_target_definition_score(self, text: str, aliases: list[str]) -> float:
        return origin_helpers.origin_target_definition_score(text, aliases)

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
        selected_candidate_title = self._selected_followup_candidate_title(contract)
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
        target_terms = [str(item).strip() for item in contract.targets if str(item).strip()]
        for paper in papers:
            paper_evidence = [item for item in evidence if item.paper_id == paper.paper_id]
            if not paper_evidence:
                continue
            target_context = "\n".join([paper.title, str(paper.metadata.get("paper_card_text", "")), *[item.snippet for item in paper_evidence[:8]]])
            matched_targets = [target for target in target_terms if self._matches_target(target_context, target)]
            if target_terms and not matched_targets:
                continue
            scored_formula_blocks = [
                (self._formula_block_score(item.snippet, contract=contract), item)
                for item in paper_evidence
            ]
            strong_formula_blocks = [
                item
                for score, item in sorted(scored_formula_blocks, key=lambda row: (-row[0], row[1].page, row[1].doc_id))
                if score >= 3.0
            ]
            formula_blocks = strong_formula_blocks or [
                item
                for item in paper_evidence
                if "formula" in item.snippet.lower() or "objective" in item.snippet.lower()
            ]
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
                fallback_evidence_ids=self._evidence_ids_for_paper(evidence, paper.paper_id, limit=3),
                fallback_term_text="\n".join(item.snippet for item in paper_evidence[:3]),
                term_extractor=self._formula_terms,
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
        return formula_helpers.fallback_formula_payload(selected_evidence, term_extractor=self._formula_terms)

    def _llm_extract_formula_claim_payload(self, *, contract: QueryContract, evidence: list[EvidenceBlock]) -> dict[str, Any]:
        if self.clients.chat is None or not evidence:
            return {}
        payload = self.clients.invoke_json(
            system_prompt=(
                "你是论文公式抽取器。只从给定 evidence 中抽取用户目标对应的公式，"
                "不要凭常识补全 DPO/PPO/PBA 等已知公式，不要使用白名单模板。"
                "如果 evidence 中存在多个不同定义，只选择当前 paper/evidence 中最直接支撑用户目标的公式。"
                "只输出 JSON：formula_text 或 formula_latex, formula_format(latex|text), "
                "variables, evidence_ids, confidence。"
                "如果 formula_format=latex，formula_text/formula_latex 必须是不带 $$ 的标准 LaTeX 公式体，"
                "能够直接放入 Markdown 的 $$...$$ 中由 KaTeX 渲染；不要输出半 Unicode 半 LaTeX 的粘连形式。"
                "把数学符号转写为 LaTeX 命令，例如 ∇θ 写作 \\nabla_{\\theta}，πθ 写作 \\pi_{\\theta}，"
                "πref 写作 \\pi_{\\mathrm{ref}}，log 写作 \\log。"
                "当用户只是问“公式/目标函数/损失”且没有明确问梯度、更新或推导时，优先抽取 scalar objective/loss；"
                "不要返回以 \\nabla 或 ∇ 开头的 gradient 公式。只有用户明确问 gradient/梯度/更新/导数时才抽取梯度公式。"
                "variables 必须是数组，每项为 {symbol, description}，description 只解释 evidence 中能支持的变量含义。"
                "如果用户用中文提问，variables.description 必须用中文；公式符号、论文标题和专有名词可以保留英文。"
                "variables.symbol 也必须使用可直接放在 $...$ 中渲染的 LaTeX，不要输出 ∇θLDPO 这类粘连符号。"
                "formula_text 必须是 evidence 中明确出现或可由相邻断行直接拼接出来的公式；证据不足返回空字符串。"
            ),
            human_prompt=json.dumps(
                {
                    "query": contract.clean_query,
                    "targets": contract.targets,
                    "answer_slots": list(getattr(contract, "answer_slots", []) or []),
                    "requested_fields": contract.requested_fields,
                    "evidence": [
                        {
                            "doc_id": item.doc_id,
                            "paper_id": item.paper_id,
                            "title": item.title,
                            "page": item.page,
                            "block_type": item.block_type,
                            "snippet": item.snippet[:1600],
                        }
                        for item in evidence[:8]
                    ],
                },
                ensure_ascii=False,
            ),
            fallback={},
        )
        return formula_helpers.llm_formula_payload_from_response(
            payload,
            allowed_evidence_ids={item.doc_id for item in evidence},
            term_extractor=self._formula_terms,
        )

    @staticmethod
    def _normalize_formula_variable_symbol(symbol: str) -> str:
        return formula_helpers.normalize_formula_variable_symbol(symbol)

    @classmethod
    def _normalize_extracted_formula_text(cls, text: str) -> str:
        return formula_helpers.normalize_extracted_formula_text(text)

    def _solve_table(self, *, contract: QueryContract, papers: list[CandidatePaper], evidence: list[EvidenceBlock]) -> list[Claim]:
        table_blocks = [
            item
            for item in evidence
            if item.block_type in {"table", "caption"}
            or (item.block_type == "page_text" and self._metric_line_score(item.snippet) >= 3)
        ]
        if not table_blocks:
            return []
        paper_by_id = {paper.paper_id: paper for paper in papers}
        ranked_blocks = sorted(
            table_blocks,
            key=lambda item: (
                -self._metric_block_score(item=item, contract=contract, paper_by_id=paper_by_id),
                item.page,
                item.doc_id,
            ),
        )
        primary_paper = paper_by_id.get(ranked_blocks[0].paper_id) if ranked_blocks else None
        fallback_paper = max(papers, key=lambda item: item.score) if papers else None
        selected_paper = primary_paper or fallback_paper
        if selected_paper is None:
            return []
        evidence_ids = [item.doc_id for item in ranked_blocks[:4]]
        paper_ids = list(dict.fromkeys(item.paper_id for item in ranked_blocks[:4] if item.paper_id))
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
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "你是论文表格视觉理解求解器。请结合表格抽取文本和页面图片回答用户查询。"
                    "只输出 JSON，字段为 claims 和 draft_answer。"
                    "claims 中每项包含 claim, metric_lines, confidence；不要编造看不见的数值。"
                    f"\nquery={contract.clean_query}"
                ),
            }
        ]
        rendered_pages: set[tuple[str, int]] = set()
        for idx, item in enumerate(ranked_blocks[:3], start=1):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"[table_context_{idx}] title={item.title} page={item.page} block_type={item.block_type}\n"
                        f"caption={item.caption}\ntable_text={item.snippet[:1200]}"
                    ),
                }
            )
            if item.block_type not in {"table", "caption"}:
                continue
            page_key = (item.file_path, item.page)
            if page_key in rendered_pages:
                continue
            rendered_pages.add(page_key)
            image_url = self._render_page_image_data_url(file_path=item.file_path, page=item.page)
            if image_url:
                content.append({"type": "image_url", "image_url": {"url": image_url}})
        if not any(block.get("type") == "image_url" for block in content):
            return None
        payload = self.clients.invoke_multimodal_json(
            system_prompt="你是论文表格视觉理解求解器。只输出 JSON。",
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
        figure_contexts = self._build_figure_contexts(evidence)
        if not figure_contexts:
            return []
        fallback_text = self._figure_fallback_summary(figure_contexts)
        evidence_benchmarks = self._extract_figure_benchmarks("\n".join(item.snippet for item in evidence[:10]))
        if len(evidence_benchmarks) >= 3:
            benchmark_suffix = " 图中提到的 benchmark 包括：" + "、".join(evidence_benchmarks) + "。"
            if benchmark_suffix not in fallback_text:
                remaining = max(80, 700 - len(benchmark_suffix))
                fallback_text = fallback_text[:remaining].rstrip() + benchmark_suffix
        payload = {"claims": [], "draft_answer": ""}
        if self.settings.enable_figure_vlm:
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        "你是论文图像理解求解器。只输出 JSON，字段为 claims 和 draft_answer。"
                        "claims 中每项包含 claim, confidence。不要编造看不见的数值。"
                        f"\nquery={contract.clean_query}"
                    ),
                }
            ]
            for idx, context in enumerate(figure_contexts, start=1):
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"[figure_context_{idx}] title={context['title']} page={context['page']}\n"
                            f"caption={context['caption']}\nfigure_text={context['figure_text']}\npage_text={context['page_text']}"
                        ),
                    }
                )
                image_url = self._render_page_image_data_url(file_path=context["file_path"], page=context["page"])
                if image_url:
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
            if any(block.get("type") == "image_url" for block in content):
                payload = self.clients.invoke_multimodal_json(
                    system_prompt="你是论文图像理解求解器。只输出 JSON。",
                    human_content=content,
                    fallback={"claims": [], "draft_answer": ""},
                )
        vlm_claim = figure_conclusion_claim_from_vlm_payload(
            payload,
            entity=contract.targets[0] if contract.targets else figure_contexts[0]["title"],
            evidence_ids=figure_contexts[0]["doc_ids"],
            paper_id=figure_contexts[0]["paper_id"],
            fallback_text=fallback_text,
            signal_score=self._figure_signal_score,
        )
        if vlm_claim is not None:
            return [vlm_claim]
        text_summary = self._summarize_figure_text(contract=contract, fallback_text=fallback_text, evidence=evidence)
        if text_summary:
            return [
                Claim(
                    claim_type="figure_conclusion",
                    entity=contract.targets[0] if contract.targets else figure_contexts[0]["title"],
                    value=text_summary,
                    structured_data={"mode": "text_summary"},
                    evidence_ids=figure_contexts[0]["doc_ids"],
                    paper_ids=[figure_contexts[0]["paper_id"]],
                    confidence=0.82,
                )
            ]
        return [
            Claim(
                claim_type="figure_conclusion",
                entity=contract.targets[0] if contract.targets else figure_contexts[0]["title"],
                value=fallback_text,
                structured_data={"mode": "caption_fallback"},
                evidence_ids=figure_contexts[0]["doc_ids"],
                paper_ids=[figure_contexts[0]["paper_id"]],
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
            system_prompt=(
                "你是 topology 证据分析器。"
                "请只输出 JSON，字段为 overall_best, engineering_best, rationale, summary。"
                "必须严格基于给定证据，不要使用外部知识。"
            ),
            human_prompt=json.dumps(
                {
                    "topology_terms": topology_terms,
                    "evidence": [item.snippet[:260] for item in evidence[:6]],
                },
                ensure_ascii=False,
            ),
            fallback={},
        )
        summary = str(payload.get("summary", "")).strip()
        if summary and not self._is_unusable_topology_recommendation_text(summary):
            return {
                "overall_best": str(payload.get("overall_best", "")).strip(),
                "engineering_best": str(payload.get("engineering_best", "")).strip(),
                "rationale": str(payload.get("rationale", "")).strip(),
                "summary": summary,
            }
        return fallback_topology_recommendation(topology_terms)

    @staticmethod
    def _is_unusable_topology_recommendation_text(text: str) -> bool:
        return is_unusable_topology_recommendation_text(text)

    def _extract_metric_lines(self, evidence: list[EvidenceBlock]) -> list[str]:
        return metric_helpers.extract_metric_lines(evidence, token_weights=self.settings.solver_metric_token_weights)

    def _metric_line_score(self, text: str) -> int:
        return metric_helpers.metric_line_score(text, token_weights=self.settings.solver_metric_token_weights)

    def _metric_block_score(
        self,
        *,
        item: EvidenceBlock,
        contract: QueryContract,
        paper_by_id: dict[str, CandidatePaper],
    ) -> float:
        paper = paper_by_id.get(item.paper_id)
        target_paper_match = bool(
            paper is not None
            and contract.targets
            and self._paper_identity_matches_targets(paper=paper, targets=contract.targets)
        )
        return metric_helpers.metric_block_score(
            item=item,
            contract=contract,
            paper_by_id=paper_by_id,
            token_weights=self.settings.solver_metric_token_weights,
            target_paper_match=target_paper_match,
        )

    def _formula_block_score(self, text: str, *, contract: QueryContract | None = None) -> float:
        return formula_helpers.formula_block_score(
            text,
            query=contract.clean_query if contract is not None else None,
            token_weights=self.settings.retrieval_formula_token_weights,
        )

    @staticmethod
    def _formula_query_wants_gradient(query: str) -> bool:
        return formula_helpers.formula_query_wants_gradient(query)
