from __future__ import annotations

import json
import re

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, VerificationReport
from app.services import origin_selection_helpers as origin_helpers
from app.services.prompt_safety import DOCUMENT_SAFETY_INSTRUCTION, wrap_untrusted_document_text


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
        if self.clients.chat is None:
            return None
        if not any(dict(claim.structured_data or {}).get("source") == "schema_claim_solver" for claim in claims):
            return None
        payload = self.clients.invoke_json(
            system_prompt=(
                "你是论文研究助手的通用证据覆盖验证器。"
                "不要按 relation 写分支规则。"
                "给定 query_contract、required_claims、claims 和 evidence，"
                "判断 claims 是否被 evidence 覆盖。"
                f"{DOCUMENT_SAFETY_INSTRUCTION}"
                "只输出 JSON：status(pass|retry|clarify), missing_fields, recommended_action, contradictions。"
                "retry 表示可以通过更多检索补足；clarify 表示必须让用户消歧或补槽。"
            ),
            human_prompt=json.dumps(
                {
                    "query": contract.clean_query,
                    "contract": contract.model_dump(),
                    "required_claims": plan.required_claims,
                    "claims": [item.model_dump() for item in claims],
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
                            "snippet": wrap_untrusted_document_text(
                                item.snippet,
                                doc_id=item.doc_id,
                                title=item.title,
                                source=item.block_type or "pdf",
                                max_chars=1000,
                            ),
                        }
                        for item in evidence[:40]
                    ],
                },
                ensure_ascii=False,
            ),
            fallback={},
        )
        if not isinstance(payload, dict) or not payload:
            return None
        status = str(payload.get("status", "") or "").strip().lower()
        if status not in {"pass", "retry", "clarify"}:
            return None
        missing = payload.get("missing_fields", [])
        if isinstance(missing, str):
            missing_fields = [missing]
        elif isinstance(missing, list):
            missing_fields = [str(item).strip() for item in missing if str(item).strip()]
        else:
            missing_fields = []
        contradictions = payload.get("contradictions", [])
        contradictory_claims: list[str] = []
        if isinstance(contradictions, list):
            contradictory_claims = [str(item).strip() for item in contradictions if str(item).strip()]
        return VerificationReport(
            status=status,  # type: ignore[arg-type]
            missing_fields=missing_fields,
            recommended_action=str(payload.get("recommended_action", "") or f"schema_verifier_{status}"),
            contradictory_claims=contradictory_claims,
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
        checks = []
        if goals & {"paper_title", "year", "origin"}:
            checks.append(self._verify_origin_lookup_claims)
        entity_like = bool(goals & {"entity_type", "role_in_context"})
        if entity_like:
            checks.append(self._verify_entity_definition_claims)
        if goals & {"followup_papers", "candidate_relationship", "strict_followup"}:
            checks.append(self._verify_followup_research_claims)
        if "recommended_papers" in goals:
            checks.append(self._verify_paper_recommendation_claims)
        if goals & {"best_topology", "langgraph_recommendation"}:
            checks.append(self._verify_topology_recommendation_claims)
        if "figure_conclusion" in goals:
            checks.append(self._verify_figure_question_claims)
        if "metric_value" in goals:
            checks.append(self._verify_metric_value_lookup_claims)
        if "formula" in goals:
            checks.append(self._verify_formula_lookup_claims)
        if "reward_model_requirement" in goals:
            checks.append(self._verify_general_question_claims)
        elif not entity_like and goals & {"definition", "examples"}:
            checks.append(self._verify_concept_definition_claims)
        if not checks:
            checks.append(self._verify_general_question_claims)
        for check in checks:
            report = check(contract=contract, plan=plan, claims=claims, papers=papers, evidence=evidence)
            if report is not None:
                return report
        return None

    @staticmethod
    def _verification_goals(*, contract: QueryContract, plan: ResearchPlan) -> set[str]:
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
        for modality in contract.required_modalities:
            if modality == "figure":
                goals.add("figure_conclusion")
            elif modality in {"table", "caption"} and ClaimVerifierMixin._looks_like_metric_verification_goal(contract.clean_query, goals):
                goals.add("metric_value")
        return goals

    @staticmethod
    def _looks_like_metric_verification_goal(query: str, goals: set[str]) -> bool:
        if goals & {"metric_value", "setting"}:
            return True
        normalized = " ".join(str(query or "").lower().split())
        return any(token in normalized for token in ["多少", "数值", "准确率", "得分", "score", "accuracy", "metric", "win rate"])

    def _verify_origin_lookup_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        claim = next((item for item in claims if item.claim_type == "origin"), None)
        if claim is None or not claim.evidence_ids or not claim.paper_ids:
            return VerificationReport(status="retry", missing_fields=["paper_title"], recommended_action="retry_origin")
        if not self._origin_claim_has_intro_support(contract=contract, claim=claim, papers=papers, evidence=evidence):
            return VerificationReport(
                status="retry",
                missing_fields=["origin_evidence"],
                unsupported_claims=["origin claim lacks an introduction/proposal cue near the requested target"],
                recommended_action="retry_origin",
            )
        return None

    def _origin_claim_has_intro_support(
        self,
        *,
        contract: QueryContract,
        claim: Claim,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> bool:
        targets = list(contract.targets or [])
        if claim.entity:
            targets.append(claim.entity)
        aliases = origin_helpers.origin_target_aliases(targets)
        if not aliases:
            return bool(claim.evidence_ids and claim.paper_ids)
        claim_paper_ids = {str(item) for item in claim.paper_ids if str(item)}
        claim_evidence_ids = {str(item) for item in claim.evidence_ids if str(item)}
        paper_by_id = {item.paper_id: item for item in papers}
        for paper_id in list(claim_paper_ids):
            paper = paper_by_id.get(paper_id) or self._candidate_from_paper_id(paper_id)
            if paper is not None and origin_helpers.origin_target_intro_score(origin_helpers.origin_paper_text(paper), aliases) > 0:
                return True
            paper_doc = self.retriever.paper_doc_by_id(paper_id)
            if paper_doc is not None and origin_helpers.origin_target_intro_score(str(paper_doc.page_content or ""), aliases) > 0:
                return True
        for item in evidence:
            if item.doc_id not in claim_evidence_ids and item.paper_id not in claim_paper_ids:
                continue
            text = "\n".join([item.title, item.caption, item.snippet])
            if origin_helpers.origin_target_intro_score(text, aliases) > 0:
                return True
        return False

    def _verify_entity_definition_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        claim = next((item for item in claims if item.claim_type == "entity_definition"), None)
        if claim is None or not str(claim.value or "").strip() or not claim.paper_ids or not claim.evidence_ids:
            return VerificationReport(status="retry", missing_fields=["entity_type"], recommended_action="retry_entity")
        claim_paper_ids = set(claim.paper_ids)
        claim_evidence_ids = set(claim.evidence_ids)
        claim_papers = [item for item in papers if item.paper_id in claim_paper_ids]
        claim_evidence = [item for item in evidence if item.doc_id in claim_evidence_ids or item.paper_id in claim_paper_ids]
        if contract.targets and not self._targets_supported(targets=contract.targets, papers=claim_papers, evidence=claim_evidence):
            if self._targets_supported(targets=contract.targets, papers=papers, evidence=evidence):
                return VerificationReport(status="retry", missing_fields=["supporting_paper"], recommended_action="retry_entity")
            return VerificationReport(status="clarify", missing_fields=["relevant_evidence"], recommended_action="clarify_target")
        return None

    def _verify_followup_research_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        claim = next((item for item in claims if item.claim_type == "followup_research"), None)
        followup_titles = list(claim.structured_data.get("followup_titles", [])) if claim else []
        seed_ids = {str(item.get("paper_id", "")) for item in list(claim.structured_data.get("seed_papers", []))} if claim else set()
        if len(followup_titles) < 1:
            return VerificationReport(status="retry", missing_fields=["followup_papers"], recommended_action="broaden_followup")
        if any(str(item.get("paper_id", "")) in seed_ids for item in followup_titles):
            return VerificationReport(status="retry", missing_fields=["followup_papers"], recommended_action="exclude_seed_paper")
        return None

    def _verify_paper_recommendation_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        claim = next((item for item in claims if item.claim_type == "paper_recommendation"), None)
        recommended = list(claim.structured_data.get("recommended_papers", [])) if claim else []
        if len(recommended) < 1:
            return VerificationReport(status="retry", missing_fields=["recommended_papers"], recommended_action="broaden_recommendation")
        return None

    def _verify_topology_recommendation_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        claim = next((item for item in claims if item.claim_type == "topology_recommendation"), None)
        if claim is None:
            return VerificationReport(
                status="retry",
                missing_fields=["best_topology", "langgraph_recommendation"],
                recommended_action="retry_topology_recommendation",
            )
        structured = dict(claim.structured_data or {})
        topology_terms = [str(item).strip() for item in structured.get("topology_terms", []) if str(item).strip()]
        if not topology_terms and hasattr(self, "_extract_topology_terms"):
            topology_terms = self._extract_topology_terms(evidence)
        if not claim.evidence_ids and not topology_terms:
            return VerificationReport(
                status="retry",
                missing_fields=["topology_evidence"],
                recommended_action="retry_topology_recommendation",
            )
        return None

    def _verify_figure_question_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        claim = next((item for item in claims if item.claim_type == "figure_conclusion"), None)
        if claim is None or not claim.evidence_ids:
            return VerificationReport(status="retry", missing_fields=["figure_conclusion"], recommended_action="retry_figure")
        claim_paper_ids = set(claim.paper_ids)
        target_papers = [paper for paper in papers if not claim_paper_ids or paper.paper_id in claim_paper_ids]
        if contract.targets and target_papers and not any(
            self._paper_identity_matches_targets(paper=paper, targets=contract.targets)
            for paper in target_papers
        ):
            return VerificationReport(status="clarify", missing_fields=["target_paper"], recommended_action="clarify_target")
        return None

    def _verify_metric_value_lookup_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        claim = next((item for item in claims if item.claim_type == "metric_value"), None)
        if claim is None:
            return VerificationReport(status="retry", missing_fields=["metric_value"], recommended_action="retry_table")
        return None

    def _verify_formula_lookup_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        formula_claims = [item for item in claims if item.claim_type == "formula"]
        if not formula_claims:
            return VerificationReport(status="retry", missing_fields=["formula"], recommended_action="retry_formula")
        invalid_values = [
            str(claim.entity or claim.value or "formula")
            for claim in formula_claims
            if not str(claim.value or "").strip()
            or str(claim.value).startswith("已定位到")
            or not self._claim_value_looks_like_formula(str(claim.value or ""))
        ]
        if invalid_values:
            return VerificationReport(status="retry", missing_fields=["formula"], unsupported_claims=invalid_values, recommended_action="retry_formula")
        llm_report = self._verify_formula_claims_with_llm(
            contract=contract,
            claims=formula_claims,
            papers=papers,
            evidence=evidence,
        )
        if llm_report is not None:
            return llm_report
        if contract.targets:
            misaligned = [
                str(dict(claim.structured_data or {}).get("paper_title") or claim.entity or claim.value)
                for claim in formula_claims
                if not self._formula_claim_matches_target(contract=contract, claim=claim, papers=papers, evidence=evidence)
            ]
            if misaligned:
                return VerificationReport(
                    status="retry",
                    missing_fields=["target_aligned_formula"],
                    unsupported_claims=misaligned,
                    recommended_action="retry_formula_target_alignment",
                )
        return None

    def _verify_formula_claims_with_llm(
        self,
        *,
        contract: QueryContract,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        if self.clients.chat is None or not claims:
            return None
        claim_evidence_ids = {doc_id for claim in claims for doc_id in claim.evidence_ids}
        claim_paper_ids = {paper_id for claim in claims for paper_id in claim.paper_ids}
        relevant_evidence = [
            item
            for item in evidence
            if item.doc_id in claim_evidence_ids or item.paper_id in claim_paper_ids
        ]
        if not relevant_evidence:
            relevant_evidence = evidence[:20]
        payload = self.clients.invoke_json(
            system_prompt=(
                "你是论文公式 claim verifier。给定用户目标、formula claims 和 evidence，"
                "判断每条公式是否被 evidence 支撑，且是否对应用户 targets。"
                "不要因为 evidence 同时讨论 PPO/DPO/其他算法就否决；只看 claim.value 的公式、"
                "claim.entity/paper_title 和 evidence 中的明确指代是否一致。"
                "如果公式是从常识模板补出来、目标不一致或证据不足，返回 retry。"
                "只输出 JSON：status(pass|retry|clarify), missing_fields, unsupported_claims, "
                "contradictory_claims, recommended_action。"
            ),
            human_prompt=json.dumps(
                {
                    "query": contract.clean_query,
                    "targets": contract.targets,
                    "answer_slots": list(getattr(contract, "answer_slots", []) or []),
                    "claims": [claim.model_dump() for claim in claims],
                    "papers": [
                        {
                            "paper_id": item.paper_id,
                            "title": item.title,
                            "year": item.year,
                        }
                        for item in papers[:12]
                    ],
                    "evidence": [
                        {
                            "doc_id": item.doc_id,
                            "paper_id": item.paper_id,
                            "title": item.title,
                            "page": item.page,
                            "block_type": item.block_type,
                            "caption": item.caption,
                            "snippet": item.snippet[:1400],
                        }
                        for item in relevant_evidence[:24]
                    ],
                },
                ensure_ascii=False,
            ),
            fallback={},
        )
        if not isinstance(payload, dict) or not payload:
            return None
        raw_status = str(payload.get("status", "") or "").strip().lower()
        status_map = {
            "pass": "pass",
            "supported": "pass",
            "retry": "retry",
            "insufficient": "retry",
            "unsupported": "retry",
            "contradicted": "retry",
            "contradiction": "retry",
            "clarify": "clarify",
            "ambiguous": "clarify",
        }
        status = status_map.get(raw_status)
        if status is None:
            return None
        if status == "pass":
            return None
        missing_fields = self._coerce_verifier_string_list(payload.get("missing_fields"))
        unsupported_claims = self._coerce_verifier_string_list(payload.get("unsupported_claims"))
        contradictory_claims = self._coerce_verifier_string_list(
            payload.get("contradictory_claims") or payload.get("contradictions")
        )
        if not missing_fields:
            missing_fields = ["formula_evidence"]
        return VerificationReport(
            status=status,  # type: ignore[arg-type]
            missing_fields=missing_fields,
            unsupported_claims=unsupported_claims,
            contradictory_claims=contradictory_claims,
            recommended_action=str(payload.get("recommended_action") or "retry_formula_evidence"),
        )

    @staticmethod
    def _coerce_verifier_string_list(value: object) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _claim_value_looks_like_formula(value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        lowered = text.lower()
        formula_markers = ["=", "\\frac", "\\sum", "\\mathbb", "\\mathcal", "∑", "π", "sigma", "loss", "objective"]
        return any(marker in lowered or marker in text for marker in formula_markers)

    def _formula_claim_matches_target(
        self,
        *,
        contract: QueryContract,
        claim: Claim,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> bool:
        claim_evidence_ids = set(claim.evidence_ids)
        claim_paper_ids = set(claim.paper_ids)
        claim_evidence = [item for item in evidence if item.doc_id in claim_evidence_ids or item.paper_id in claim_paper_ids]
        claim_papers = [item for item in papers if item.paper_id in claim_paper_ids]
        context = "\n".join(
            [
                str(claim.value or ""),
                str(claim.entity or ""),
                *[item.title for item in claim_papers],
                *[item.snippet for item in claim_evidence[:6]],
            ]
        )
        normalized_context = self._normalize_lookup_text(context)
        entity_targets = [
            part.strip()
            for part in re.split(r"[/,;、]", str(claim.entity or ""))
            if part.strip()
        ]
        candidate_targets = list(dict.fromkeys([*entity_targets, *list(contract.targets or [])]))
        if not candidate_targets:
            return True
        for target in candidate_targets:
            target_key = self._normalize_lookup_text(target)
            if not target_key:
                continue
            formula_text = f"{claim.value}\n{claim.entity}"
            if self._matches_target(formula_text, target):
                return True
            if self._formula_evidence_supports_target(target=target, evidence=claim_evidence):
                return True
            if not self._is_short_acronym(target) and self._matches_target(normalized_context, target):
                return True
        return False

    def _formula_evidence_supports_target(self, *, target: str, evidence: list[EvidenceBlock]) -> bool:
        formula_markers = ["=", "formula", "objective", "loss", "目标函数", "公式", "∑", "π", "\\pi", "sigma", "σ"]
        for item in evidence[:8]:
            text = "\n".join([item.title, item.caption, item.snippet])
            if not self._matches_target(text, target):
                continue
            lowered = text.lower()
            if any(marker in lowered or marker in text for marker in formula_markers):
                return True
        return False

    def _verify_concept_definition_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        claim = next((item for item in claims if item.claim_type == "concept_definition"), None)
        if claim is None or not claim.evidence_ids:
            return VerificationReport(status="retry", missing_fields=["definition"], recommended_action="retry_definition")
        if contract.targets and not self._targets_supported(targets=contract.targets, papers=papers, evidence=evidence):
            if papers or evidence:
                return VerificationReport(status="retry", missing_fields=["relevant_evidence"], recommended_action="retry_definition")
            return VerificationReport(status="clarify", missing_fields=["relevant_evidence"], recommended_action="clarify_target")
        return None

    def _verify_general_question_claims(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        claims: list[Claim],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> VerificationReport | None:
        if contract.targets and not self._targets_supported(targets=contract.targets, papers=papers, evidence=evidence):
            if papers or evidence:
                return VerificationReport(status="retry", missing_fields=["relevant_evidence"], recommended_action="expand_recall")
            return VerificationReport(status="clarify", missing_fields=["relevant_evidence"], recommended_action="clarify_target")
        return None

    def _targets_supported(
        self,
        *,
        targets: list[str],
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> bool:
        normalized_targets = [self._normalize_lookup_text(item) for item in targets if item]
        if not normalized_targets:
            return True
        haystacks = [self._normalize_lookup_text(item.snippet) for item in evidence[:8]]
        for paper in papers[:4]:
            haystacks.append(self._normalize_lookup_text(paper.title))
            haystacks.append(self._normalize_lookup_text(str(paper.metadata.get("paper_card_text", ""))))
        return any(
            self._matches_target(haystack, target)
            for target in normalized_targets
            for haystack in haystacks
            if haystack
        )

    def _paper_identity_matches_targets(self, *, paper: CandidatePaper, targets: list[str]) -> bool:
        normalized_targets = [str(item).strip() for item in targets if str(item).strip() and not self._is_structural_target_reference(item)]
        if not normalized_targets:
            return True
        aliases = [alias.strip() for alias in str(paper.metadata.get("aliases", "")).split("||") if alias.strip()]
        candidate_names = [paper.title, *aliases]
        if paper.title:
            for separator in [":", " - ", " — ", " – "]:
                if separator in paper.title:
                    head = paper.title.split(separator, 1)[0].strip()
                    if head and head not in candidate_names:
                        candidate_names.append(head)
        for target in normalized_targets:
            canonical_target = self.retriever.canonicalize_target(target)
            normalized_target = self.retriever._normalize_entity_text(canonical_target)
            if not normalized_target:
                continue
            for candidate_name in candidate_names:
                if self._is_initialism_alias_match(candidate_name=candidate_name, target=canonical_target):
                    return True
                candidate = self.retriever._normalize_entity_text(candidate_name)
                if not candidate:
                    continue
                if self._is_identity_alias_match(candidate=candidate, target=normalized_target):
                    return True
        return False

    @staticmethod
    def _is_identity_alias_match(*, candidate: str, target: str) -> bool:
        if not candidate or not target:
            return False
        if candidate == target:
            return True
        if candidate.startswith(target) and len(target) >= 4:
            remainder = candidate[len(target) :]
            if remainder and remainder[0] in {"-", " ", "/", ":"}:
                return True
        if target.startswith(candidate) and len(candidate) >= 4:
            remainder = target[len(candidate) :]
            if remainder and remainder[0] in {"-", " ", "/", ":"}:
                return True
        return False

    @staticmethod
    def _is_initialism_alias_match(*, candidate_name: str, target: str) -> bool:
        normalized_target = re.sub(r"[^A-Za-z0-9]", "", str(target or "")).upper()
        if not (2 <= len(normalized_target) <= 10) or not normalized_target.isupper():
            return False
        words = re.findall(r"[A-Za-z][A-Za-z0-9]*", str(candidate_name or ""))
        if len(words) < 2:
            return False
        stopwords = {"a", "an", "and", "are", "for", "in", "is", "of", "on", "the", "to", "via", "with", "your"}
        initials = "".join(word[0].upper() for word in words if word.lower() not in stopwords)
        return initials == normalized_target
