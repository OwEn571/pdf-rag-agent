from __future__ import annotations

import json
import re
from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services import origin_selection_helpers as origin_helpers
from app.services.confidence import coerce_confidence_value
from app.services.contract_normalization import normalize_lookup_text
from app.services.evidence_presentation import safe_year
from app.services.query_shaping import matches_target


class EntityDefinitionMixin:
    def _candidate_from_paper_id(self, paper_id: str) -> CandidatePaper | None:
        doc = self.retriever.paper_doc_by_id(paper_id)
        if doc is None:
            return None
        meta = dict(doc.metadata or {})
        return CandidatePaper(
            paper_id=paper_id,
            title=str(meta.get("title", "")),
            year=str(meta.get("year", "")),
            score=0.0,
            match_reason="paper_doc_lookup",
            doc_ids=[str(meta.get("doc_id", ""))] if meta.get("doc_id") else [],
            metadata=meta,
        )

    def _ground_entity_papers(
        self,
        *,
        candidates: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        limit: int,
    ) -> list[CandidatePaper]:
        if not evidence:
            return candidates[: max(1, limit)]
        by_id = {item.paper_id: item for item in candidates}
        aggregated: dict[str, dict[str, Any]] = {}
        for item in evidence:
            bucket = aggregated.setdefault(item.paper_id, {"score": 0.0, "doc_ids": []})
            bucket["score"] += float(item.score)
            if item.doc_id not in bucket["doc_ids"]:
                bucket["doc_ids"].append(item.doc_id)
        grounded: list[CandidatePaper] = []
        for paper_id, payload in aggregated.items():
            paper = by_id.get(paper_id) or self._candidate_from_paper_id(paper_id)
            if paper is None:
                continue
            grounded.append(
                paper.model_copy(
                    update={
                        "score": paper.score + float(payload["score"]) + (len(payload["doc_ids"]) * 0.25),
                        "match_reason": "entity_evidence_grounded",
                        "doc_ids": list(dict.fromkeys([*paper.doc_ids, *payload["doc_ids"]]))[:6],
                    }
                )
            )
        grounded.sort(key=lambda item: (-item.score, safe_year(item.year), item.title))
        return grounded[: max(1, limit)] or candidates[: max(1, limit)]

    def _select_entity_supporting_paper(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> tuple[CandidatePaper | None, list[EvidenceBlock]]:
        if not papers:
            return None, []
        target = contract.targets[0] if contract.targets else ""
        if not target:
            paper = self._best_entity_fallback_paper(papers=papers, evidence=evidence)
            return paper, self._prune_entity_supporting_evidence([item for item in evidence if item.paper_id == paper.paper_id])
        context_targets = [item for item in contract.targets[1:] if str(item).strip()]

        matching_evidence = [
            item
            for item in evidence
            if any(
                matches_target(haystack, target)
                for haystack in [item.snippet, item.caption, item.title]
                if haystack
            )
        ]
        if context_targets:
            identity_contextual_evidence = [
                item
                for item in matching_evidence
                if self._entity_context_identity_matches(item=item, context_targets=context_targets)
            ]
            contextual_evidence = [
                item
                for item in matching_evidence
                if self._entity_context_matches(item=item, context_targets=context_targets)
            ]
            if identity_contextual_evidence:
                matching_evidence = identity_contextual_evidence
            elif contextual_evidence:
                matching_evidence = contextual_evidence
        paper_rank = {item.paper_id: idx for idx, item in enumerate(papers)}

        if matching_evidence:
            llm_paper, llm_evidence = self._llm_select_entity_supporting_paper(
                contract=contract,
                papers=papers,
                matching_evidence=matching_evidence,
            ) if not context_targets else (None, [])
            if llm_paper is not None:
                return llm_paper, llm_evidence
            scored: list[tuple[float, CandidatePaper]] = []
            for paper in papers:
                support = [item for item in matching_evidence if item.paper_id == paper.paper_id]
                definition_bonus = sum(float(item.metadata.get("definition_score", 0) or 0) for item in support)
                mechanism_bonus = sum(float(item.metadata.get("mechanism_score", 0) or 0) for item in support)
                application_bonus = sum(float(item.metadata.get("application_score", 0) or 0) for item in support)
                paper_text = "\n".join(
                    [
                        paper.title,
                        str(paper.metadata.get("aliases", "")),
                        str(paper.metadata.get("paper_card_text", "")),
                        str(paper.metadata.get("generated_summary", "")),
                        str(paper.metadata.get("abstract_note", "")),
                        "\n".join(item.snippet for item in support[:3]),
                    ]
                )
                score = sum(item.score for item in support)
                if support:
                    score += 3.0
                score += definition_bonus * 2.5
                score += mechanism_bonus * 1.0
                score += application_bonus * 0.3
                if matches_target(paper_text, target):
                    score += 1.2
                if context_targets:
                    if self._paper_identity_matches_targets(paper=paper, targets=context_targets) or self._paper_introduces_context_target(paper=paper, context_targets=context_targets):
                        score += 12.0
                    elif any(matches_target(paper_text, context_target) for context_target in context_targets):
                        score += 4.0
                if definition_bonus > 0:
                    score += 1.6
                elif any(self._entity_definition_score(item.snippet) > 0 for item in support):
                    score += 0.8
                scored.append((score, paper))
            scored.sort(key=lambda item: (-item[0], paper_rank.get(item[1].paper_id, 999), item[1].title))
            best_paper = scored[0][1]
            best_evidence = [item for item in matching_evidence if item.paper_id == best_paper.paper_id]
            best_evidence.sort(
                key=lambda item: (
                    -self._entity_definition_score(item.snippet),
                    -item.score,
                    item.page,
                    item.doc_id,
                )
            )
            return best_paper, self._prune_entity_supporting_evidence(best_evidence)

        for paper in papers:
            paper_text = "\n".join(
                [
                    paper.title,
                    str(paper.metadata.get("aliases", "")),
                    str(paper.metadata.get("paper_card_text", "")),
                    str(paper.metadata.get("generated_summary", "")),
                    str(paper.metadata.get("abstract_note", "")),
                ]
            )
            if matches_target(paper_text, target) and (
                not context_targets or any(matches_target(paper_text, context_target) for context_target in context_targets)
            ):
                fallback_evidence = [item for item in evidence if item.paper_id == paper.paper_id]
                fallback_evidence.sort(
                    key=lambda item: (
                        -self._entity_definition_score(item.snippet),
                        -item.score,
                        item.page,
                        item.doc_id,
                    )
                )
                return paper, self._prune_entity_supporting_evidence(fallback_evidence)

        paper = self._best_entity_fallback_paper(papers=papers, evidence=evidence)
        fallback_evidence = [item for item in evidence if item.paper_id == paper.paper_id]
        fallback_evidence.sort(
            key=lambda item: (
                -self._entity_definition_score(item.snippet),
                -item.score,
                item.page,
                item.doc_id,
            )
        )
        return paper, self._prune_entity_supporting_evidence(fallback_evidence)

    @staticmethod
    def _best_entity_fallback_paper(*, papers: list[CandidatePaper], evidence: list[EvidenceBlock]) -> CandidatePaper:
        evidence_score_by_paper: dict[str, float] = {}
        for item in evidence:
            evidence_score_by_paper[item.paper_id] = evidence_score_by_paper.get(item.paper_id, 0.0) + float(item.score)
        return max(
            papers,
            key=lambda paper: (
                evidence_score_by_paper.get(paper.paper_id, 0.0),
                float(paper.score),
                -len(paper.title),
            ),
        )

    def _entity_context_identity_matches(self, *, item: EvidenceBlock, context_targets: list[str]) -> bool:
        paper = self._candidate_from_paper_id(item.paper_id)
        if paper is None:
            return False
        return self._paper_identity_matches_targets(paper=paper, targets=context_targets) or self._paper_introduces_context_target(
            paper=paper,
            context_targets=context_targets,
        )

    def _paper_introduces_context_target(self, *, paper: CandidatePaper, context_targets: list[str]) -> bool:
        paper_text = "\n".join(
            [
                paper.title,
                str(paper.metadata.get("aliases", "")),
                str(paper.metadata.get("paper_card_text", "")),
                str(paper.metadata.get("generated_summary", "")),
                str(paper.metadata.get("abstract_note", "")),
            ]
        )
        return origin_helpers.origin_target_definition_score(paper_text, [str(item) for item in context_targets]) >= 4.0

    def _entity_context_matches(self, *, item: EvidenceBlock, context_targets: list[str]) -> bool:
        paper = self._candidate_from_paper_id(item.paper_id)
        paper_text = ""
        if paper is not None:
            paper_text = "\n".join(
                [
                    paper.title,
                    str(paper.metadata.get("aliases", "")),
                    str(paper.metadata.get("paper_card_text", "")),
                    str(paper.metadata.get("generated_summary", "")),
                    str(paper.metadata.get("abstract_note", "")),
                ]
            )
        haystack = "\n".join([item.title, item.caption, item.snippet, paper_text])
        return any(matches_target(haystack, context_target) for context_target in context_targets if str(context_target).strip())

    def _llm_select_entity_supporting_paper(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        matching_evidence: list[EvidenceBlock],
    ) -> tuple[CandidatePaper | None, list[EvidenceBlock]]:
        if self.clients.chat is None or not contract.targets or not matching_evidence:
            return None, []
        target = contract.targets[0]
        grouped: dict[str, list[EvidenceBlock]] = {}
        for item in matching_evidence:
            grouped.setdefault(item.paper_id, []).append(item)
        candidates_payload: list[dict[str, Any]] = []
        by_id = {item.paper_id: item for item in papers}
        for paper in papers[:6]:
            support = grouped.get(paper.paper_id, [])
            if not support:
                continue
            ordered_support = sorted(
                support,
                key=lambda item: (
                    -float(item.metadata.get("definition_score", 0) or 0),
                    -float(item.metadata.get("mechanism_score", 0) or 0),
                    -float(item.score),
                    item.page,
                    item.doc_id,
                ),
            )[:2]
            candidates_payload.append(
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "summary": self._paper_summary_text(paper.paper_id),
                    "evidence": [
                        {
                            "doc_id": item.doc_id,
                            "page": item.page,
                            "snippet": item.snippet[:260],
                        }
                        for item in ordered_support
                    ],
                }
            )
        if not candidates_payload:
            return None, []
        payload = self.clients.invoke_json(
            system_prompt=(
                "你是论文实体 grounding 裁判器。"
                "你的任务是在多个候选论文里，挑出最适合回答“某个实体/术语是什么”的来源论文。"
                "优先级："
                "1. 直接定义或首次引入该实体的论文；"
                "2. 明确解释该实体机制/组成的论文；"
                "3. 只是使用、对比或顺带提到该实体的论文不要优先。"
                "只输出 JSON，字段为 paper_id, evidence_doc_ids, relation_to_target, confidence, reason。"
                "relation_to_target 只能是 [origin, direct_definition, mechanism_explanation, usage_only, incidental_mention]。"
            ),
            human_prompt=json.dumps(
                {
                    "query": contract.clean_query,
                    "target": target,
                    "requested_fields": contract.requested_fields,
                    "candidates": candidates_payload,
                },
                ensure_ascii=False,
            ),
            fallback={},
        )
        if not isinstance(payload, dict) or not payload:
            return None, []
        paper_id = str(payload.get("paper_id", "")).strip()
        relation_to_target = str(payload.get("relation_to_target", "")).strip().lower()
        if paper_id not in by_id:
            return None, []
        if relation_to_target not in {
            "origin",
            "direct_definition",
            "mechanism_explanation",
            "usage_only",
            "incidental_mention",
        }:
            return None, []
        if relation_to_target in {"usage_only", "incidental_mention"}:
            return None, []
        confidence = coerce_confidence_value(
            payload.get("confidence", 0),
            default=0.0,
            label_scores={"high": 0.88, "medium": 0.72, "low": 0.45},
        )
        if confidence < 0.55:
            return None, []
        raw_doc_ids = payload.get("evidence_doc_ids", [])
        evidence_doc_ids = [str(item).strip() for item in raw_doc_ids if str(item).strip()] if isinstance(raw_doc_ids, list) else []
        selected_evidence = [item for item in grouped.get(paper_id, []) if item.doc_id in evidence_doc_ids]
        if not selected_evidence:
            selected_evidence = grouped.get(paper_id, [])
        return by_id[paper_id], self._prune_entity_supporting_evidence(selected_evidence)

    def _prune_entity_supporting_evidence(self, evidence: list[EvidenceBlock]) -> list[EvidenceBlock]:
        if not evidence:
            return []
        cleaned = [item for item in evidence if not self._is_noisy_entity_line(item.snippet)]
        pool = cleaned or evidence
        deduped: list[EvidenceBlock] = []
        seen: set[str] = set()
        for item in pool:
            key = f"{item.paper_id}:{item.page}:{item.block_type}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:4]

    def _entity_supporting_lines(self, evidence: list[EvidenceBlock], *, kind: str) -> list[str]:
        scored: list[tuple[float, str]] = []
        for item in evidence:
            snippet = " ".join(item.snippet.split())
            if self._is_noisy_entity_line(snippet):
                continue
            definition_score = float(item.metadata.get("definition_score", 0) or 0)
            mechanism_score = float(item.metadata.get("mechanism_score", 0) or 0)
            application_score = float(item.metadata.get("application_score", 0) or 0)
            score = 0.0
            if kind == "definition":
                score = definition_score
            elif kind == "mechanism":
                score = mechanism_score
            elif kind == "application":
                score = application_score
            if score <= 0:
                continue
            scored.append((score, snippet[:220]))
        lines: list[str] = []
        seen: set[str] = set()
        for _, line in sorted(scored, key=lambda item: (-item[0], item[1])):
            normalized = " ".join(line.lower().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            lines.append(line)
        return lines[:3]

    @staticmethod
    def _entity_definition_score(text: str) -> int:
        haystack = " ".join(str(text or "").lower().split())
        if not haystack:
            return 0
        score = 0
        if any(
            token in haystack
            for token in [
                "algorithm",
                "framework",
                "method",
                "model",
                "system",
                "dataset",
                "benchmark",
                "objective",
                "loss",
                "reinforcement learning",
                "policy optimization",
                "reward",
                "算法",
                "方法",
                "模型",
                "系统",
                "框架",
                "数据集",
                "基准",
                "目标函数",
                "优化",
            ]
        ):
            score += 1
        if any(
            token in haystack
            for token in [
                " is a ",
                " is an ",
                " refers to ",
                " stands for ",
                " denotes ",
                " employ the ",
                " uses the ",
                " introduce ",
                " propose ",
            ]
        ):
            score += 1
        return score

    def _infer_entity_type(self, *, contract: QueryContract, papers: list[CandidatePaper], evidence: list[EvidenceBlock]) -> str:
        llm_label = self._llm_infer_entity_type(contract=contract, papers=papers, evidence=evidence)
        if llm_label:
            return llm_label
        text_parts = [item.snippet for item in evidence[:6]]
        for paper in papers[:2]:
            text_parts.extend(
                [
                    paper.title,
                    str(paper.metadata.get("paper_card_text", "")),
                    str(paper.metadata.get("generated_summary", "")),
                    str(paper.metadata.get("abstract_note", "")),
                ]
            )
        text = "\n".join(part for part in text_parts if part).lower()
        algorithm_score = 0
        dataset_score = 0
        framework_score = 0
        model_score = 0
        if any(
            token in text
            for token in [
                "algorithm",
                "policy optimization",
                "reinforcement learning",
                "reward model",
                "rule-based rewards",
                "rubric-based rewards",
                "objective",
                "loss",
                "training objective",
                "ppo",
                "grpo",
                "advantage",
                "critic",
                "group relative",
                "算法",
                "优化",
                "训练目标",
                "目标函数",
            ]
        ):
            algorithm_score += 3
        if any(token in text for token in ["dataset", "benchmark", "corpus", "leaderboard", "数据集", "基准"]):
            dataset_score += 2
        if any(token in text for token in ["framework", "system", "platform", "agent", "架构", "系统", "框架"]):
            framework_score += 2
        if any(token in text for token in ["model", "llm", "language model", "simulator", "classifier", "policy", "模型"]):
            model_score += 2
        scores = {
            "优化算法/训练方法": algorithm_score,
            "数据集/benchmark": dataset_score,
            "框架/系统": framework_score,
            "模型/方法": model_score,
        }
        best_label, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score > 0:
            return best_label
        return "方法/框架"

    def _llm_infer_entity_type(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> str:
        if self.clients.chat is None:
            return ""
        payload = self.clients.invoke_json(
            system_prompt=(
                "你是论文实体类型判别器。"
                "请基于目标术语、问题和局部证据，判断这个目标最像什么类型。"
                "优先围绕目标术语本身判断，不要被整篇论文的大主题带偏。"
                "只输出 JSON，字段为 entity_type, confidence, rationale。"
                "entity_type 尽量归一到这些类型之一："
                "[优化算法/训练方法, 数据集/benchmark, 框架/系统, 模型/方法, 评测任务/设置]。"
            ),
            human_prompt=json.dumps(
                {
                    "query": contract.clean_query,
                    "target": contract.targets[0] if contract.targets else "",
                    "requested_fields": contract.requested_fields,
                    "papers": [
                        {
                            "paper_id": item.paper_id,
                            "title": item.title,
                            "summary": self._paper_summary_text(item.paper_id),
                        }
                        for item in papers[:2]
                    ],
                    "evidence": [item.snippet[:260] for item in evidence[:5]],
                },
                ensure_ascii=False,
            ),
            fallback={},
        )
        if not isinstance(payload, dict) or not payload:
            return ""
        label = self._canonicalize_entity_type_label(str(payload.get("entity_type", "")).strip())
        return label

    @staticmethod
    def _canonicalize_entity_type_label(label: str) -> str:
        normalized = " ".join(str(label or "").lower().split())
        if not normalized:
            return ""
        alias_map = {
            "优化算法/训练方法": [
                "优化算法/训练方法",
                "强化学习算法",
                "优化算法",
                "训练方法",
                "algorithm",
                "optimization method",
                "training method",
            ],
            "数据集/benchmark": [
                "数据集/benchmark",
                "数据集",
                "benchmark",
                "dataset",
                "偏好数据集",
                "评测基准",
            ],
            "框架/系统": [
                "框架/系统",
                "框架",
                "系统",
                "framework",
                "system",
                "platform",
            ],
            "模型/方法": [
                "模型/方法",
                "模型",
                "方法",
                "model",
                "method",
            ],
            "评测任务/设置": [
                "评测任务/设置",
                "任务设置",
                "评测任务",
                "task",
                "evaluation setting",
            ],
        }
        for canonical, aliases in alias_map.items():
            normalized_aliases = {" ".join(item.lower().split()) for item in aliases}
            if normalized in normalized_aliases:
                return canonical
        return ""

    @staticmethod
    def _is_noisy_entity_line(text: str) -> bool:
        compact = " ".join(str(text or "").split())
        if not compact:
            return True
        weird_math_chars = "∑𝜋𝜃𝑜𝑡𝑞𝐴ˆβϵμ"
        if sum(1 for ch in compact if ch in weird_math_chars) >= 2:
            return True
        if compact.count("|") >= 2:
            return True
        if re.search(r"\([0-9]{1,2}\)\s*$", compact):
            return True
        letters = sum(1 for ch in compact if ch.isalpha())
        digits = sum(1 for ch in compact if ch.isdigit())
        symbols = sum(1 for ch in compact if not ch.isalnum() and ch not in " .,;:!?()[]{}-_/")
        if letters < 24 and symbols > max(4, letters):
            return True
        if digits > letters and letters < 18:
            return True
        return False

    def _compose_entity_answer_markdown(
        self,
        *,
        contract: QueryContract,
        claims: list[Any],
        evidence: list[EvidenceBlock],
        citations: list[Any],
    ) -> str:
        claim = next((item for item in claims if item.claim_type == "entity_definition"), None)
        if claim is None:
            return ""
        target = contract.targets[0] if contract.targets else (claim.entity or "该对象")
        label = claim.value or "相关技术"
        structured = dict(claim.structured_data or {})
        local_evidence = [item for item in evidence if item.doc_id in claim.evidence_ids]
        if not local_evidence and claim.paper_ids:
            local_evidence = [item for item in evidence if item.paper_id in claim.paper_ids]
        if not local_evidence:
            local_evidence = evidence
        definition_lines = [str(item).strip() for item in list(structured.get("definition_lines", [])) if str(item).strip()]
        mechanism_lines = [str(item).strip() for item in list(structured.get("mechanism_lines", [])) if str(item).strip()]
        application_lines = [str(item).strip() for item in list(structured.get("application_lines", [])) if str(item).strip()]
        if not definition_lines:
            definition_lines = self._entity_supporting_lines(local_evidence, kind="definition")
        if not mechanism_lines:
            mechanism_lines = self._entity_supporting_lines(local_evidence, kind="mechanism")
        if not application_lines:
            application_lines = self._entity_supporting_lines(local_evidence, kind="application")

        requested_fields = {normalize_lookup_text(item) for item in contract.requested_fields if item}
        detail_requested = contract.continuation_mode == "followup" or bool(
            requested_fields
            & {
                "mechanism",
                "workflow",
                "objective",
                "reward_signal",
                "training_signal",
                "formula",
                "variable_explanation",
            }
        )
        paper_title = str(structured.get("paper_title", "")).strip()
        description = self._sanitize_entity_description(str(structured.get("description", "") or ""))
        if len(description) > 520:
            description = description[:517].rstrip() + "..."
        answer: list[str] = [f"### {target}：机制与流程" if detail_requested else f"### {target} 技术简介", ""]
        if description and not self._is_noisy_entity_line(description):
            answer.append(description)
        else:
            answer.append(
                self._entity_intro_sentence(
                    target=target,
                    label=label,
                    paper_title=paper_title,
                    definition_lines=definition_lines,
                    mechanism_lines=mechanism_lines,
                    application_lines=application_lines,
                    evidence=local_evidence,
                )
            )

        if detail_requested:
            mechanism_bullets = self._entity_mechanism_bullets(
                mechanism_lines=mechanism_lines,
                evidence=local_evidence,
            )
            if mechanism_bullets:
                answer.extend(["", "核心机制："])
                answer.extend([f"- {line}" for line in mechanism_bullets[:4]])
            workflow_steps = self._entity_workflow_steps(evidence=local_evidence)
            if workflow_steps:
                answer.extend(["", "典型流程："])
                answer.extend([f"{index}. {step}" for index, step in enumerate(workflow_steps, start=1)])
            reward_bullets = self._entity_reward_bullets(evidence=local_evidence)
            if reward_bullets:
                answer.extend(["", "目标与奖励信号："])
                answer.extend([f"- {line}" for line in reward_bullets[:3]])
        else:
            summary_bullets = self._entity_summary_bullets(
                definition_lines=definition_lines,
                mechanism_lines=mechanism_lines,
                application_lines=application_lines,
            )
            if summary_bullets and not description:
                answer.extend(["", "核心要点："])
                answer.extend([f"- {line}" for line in summary_bullets[:4]])

        application_bullets = self._entity_clean_lines(application_lines, limit=2)
        if application_bullets and not description:
            heading = "当前语料里的应用：" if detail_requested else "应用场景："
            answer.extend(["", heading])
            answer.extend([f"- {line}" for line in application_bullets])

        if paper_title and not detail_requested:
            answer.extend(["", f"当前最直接的定义证据来自《{paper_title}》。"])
        if citations:
            anchor = citations[0]
            page_label = f"第 {anchor.page} 页" if anchor.page else "相关页"
            answer.extend(["", f"主要依据《{anchor.title}》{page_label} 的证据整理。"])
        return "\n".join(line for line in answer if line is not None).strip()

    @staticmethod
    def _sanitize_entity_description(text: str) -> str:
        compact_lines: list[str] = []
        for raw_line in str(text or "").replace("\r\n", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^#{1,6}\s*", "", line).strip()
            line = re.sub(r"^[-*]\s+", "", line).strip()
            line = re.sub(r"^\d+\.\s+", "", line).strip()
            if not line:
                continue
            if re.fullmatch(r"(定义|目的|关键特性|证据|应用场景|核心要点|机制|流程)[:：]?", line):
                continue
            compact_lines.append(line)
        compact = " ".join(" ".join(compact_lines).split())
        compact = re.sub(r"\[([^\]]{0,80})$", r"\1", compact).strip()
        compact = re.sub(r"\s*#+\s*", " ", compact).strip()
        return compact

    def _entity_intro_sentence(
        self,
        *,
        target: str,
        label: str,
        paper_title: str,
        definition_lines: list[str],
        mechanism_lines: list[str],
        application_lines: list[str],
        evidence: list[EvidenceBlock],
    ) -> str:
        joined = " \n".join([*definition_lines, *mechanism_lines, *application_lines, *[item.snippet for item in evidence[:6]]]).lower()
        if "ppo" in joined and ("variant" in joined or "from ppo to grpo" in joined):
            lead = f"{target} 更接近一种基于 PPO 的 `{label}`。"
        elif any(token in joined for token in ["dataset", "benchmark", "corpus", "数据集", "基准"]):
            lead = f"{target} 更接近一个 `{label}`。"
        elif any(token in joined for token in ["algorithm", "policy optimization", "reinforcement learning", "算法", "优化"]):
            lead = f"{target} 更接近一种 `{label}`。"
        else:
            lead = f"{target} 可以定位为 `{label}`。"
        details: list[str] = []
        if any(token in joined for token in ["group of outputs", "group scores", "relative rewards", "group-based reward", "group relative"]):
            details.append("它会把同一问题的多个候选输出放在一组里比较，并利用组内相对 reward 计算 advantage")
        if any(token in joined for token in ["critic", "value model", "value function", "foregoes the critic", "obviates the need"]):
            details.append("这样可以不再依赖单独的 value model / critic")
        if any(token in joined for token in ["training resources", "memory usage", "computational burden", "overhead", "sample-efficient", "reduce training resources"]):
            details.append("设计动机之一是降低 PPO 类方法的训练资源开销")
        if any(token in joined for token in ["reasoning", "alignment", "instruction following", "mathematical reasoning", "推理"]):
            details.append("常见目标是提升推理或对齐表现")
        if details:
            return lead + " " + "；".join(details[:3]) + "。"
        if definition_lines:
            return lead + " 当前最直接的证据把它描述为：" + self._entity_clean_lines(definition_lines, limit=1)[0]
        if paper_title:
            return lead + f" 当前主要依据《{paper_title}》中的相关描述。"
        return lead

    def _entity_clean_lines(self, lines: list[str], *, limit: int) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for line in lines:
            compact = " ".join(str(line or "").split()).strip(" -")
            if not compact:
                continue
            if self._is_noisy_entity_line(compact):
                continue
            normalized = compact.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            if len(compact) > 220:
                compact = compact[:217].rstrip() + "..."
            cleaned.append(compact)
            if len(cleaned) >= limit:
                break
        return cleaned

    def _entity_mechanism_bullets(self, *, mechanism_lines: list[str], evidence: list[EvidenceBlock]) -> list[str]:
        cleaned = self._entity_clean_lines(mechanism_lines, limit=4)
        if cleaned:
            return cleaned
        return self._entity_focus_lines(
            evidence=evidence,
            keywords=["group", "relative", "advantage", "critic", "value model", "objective", "clip", "kl"],
            limit=4,
        )

    def _entity_workflow_steps(self, *, evidence: list[EvidenceBlock]) -> list[str]:
        joined = " \n".join(item.snippet for item in evidence[:8]).lower()
        steps: list[str] = []
        if any(token in joined for token in ["sample a group", "sampled output", "sample 64 outputs", "group of outputs", "sample g outputs"]):
            steps.append("先对同一个问题采样一组候选输出。")
        if any(
            token in joined
            for token in [
                "compute rewards",
                "reward model",
                "score the outputs",
                "rule refers",
                "rule judgment",
                "average reward",
                "group scores",
                "baseline",
            ]
        ):
            steps.append("再根据组内 reward 或 baseline 信息，为每个输出构造训练信号。")
        if any(
            token in joined
            for token in [
                "group average",
                "group standard deviation",
                "relative rewards",
                "mean(r)",
                "std(r)",
                "advantage",
                "advantage estimation",
                "group relative",
            ]
        ):
            steps.append("随后把组内 reward 做相对化或归一化，构造 advantage，而不是依赖单独的 value critic。")
        if any(token in joined for token in ["update the policy", "policy model", "objective", "clip", "kl penalty", "maximizing the grpo objective"]):
            steps.append("最后在 clipping / KL 约束下更新 policy model。")
        return steps[:4]

    def _entity_reward_bullets(self, *, evidence: list[EvidenceBlock]) -> list[str]:
        bullets: list[str] = []
        focus = self._entity_focus_lines(
            evidence=evidence,
            keywords=["reward", "objective", "gradient", "rule", "model", "kl", "clip", "advantage"],
            limit=4,
        )
        for line in focus:
            if line not in bullets:
                bullets.append(line)
        return bullets[:3]

    def _entity_summary_bullets(
        self,
        *,
        definition_lines: list[str],
        mechanism_lines: list[str],
        application_lines: list[str],
    ) -> list[str]:
        joined = " \n".join([*definition_lines, *mechanism_lines, *application_lines]).lower()
        bullets: list[str] = []
        if "ppo" in joined and "variant" in joined:
            bullets.append("它可以看作 PPO 的一个变体。")
        if any(token in joined for token in ["group scores", "relative rewards", "group relative", "advantage"]):
            bullets.append("它通过组内相对 reward / group scores 来估计 baseline 或 advantage。")
        if any(token in joined for token in ["critic", "value model", "value function"]):
            bullets.append("它的关键区别是不再依赖单独的 value critic。")
        if any(token in joined for token in ["training resources", "memory", "computational burden", "less memory", "reduce"]):
            bullets.append("这样做的直接收益是减少训练资源和内存开销。")
        if any(token in joined for token in ["reasoning", "alignment", "mathematical reasoning", "推理"]):
            bullets.append("常见用途是提升推理或对齐表现。")
        if not bullets:
            bullets = self._entity_clean_lines(definition_lines, limit=2)
            for line in self._entity_clean_lines(mechanism_lines, limit=2):
                if line not in bullets:
                    bullets.append(line)
        for line in self._entity_clean_lines(application_lines, limit=1):
            if line not in bullets:
                bullets.append(line)
        return bullets[:4]

    def _entity_focus_lines(self, *, evidence: list[EvidenceBlock], keywords: list[str], limit: int) -> list[str]:
        scored: list[tuple[float, str]] = []
        for item in evidence:
            compact = " ".join(item.snippet.split())
            if self._is_noisy_entity_line(compact):
                continue
            lowered = compact.lower()
            score = float(item.score)
            for token in keywords:
                if token in lowered:
                    score += 1.0
            if score <= 0:
                continue
            if any(token in lowered for token in keywords):
                scored.append((score, compact))
        lines: list[str] = []
        seen: set[str] = set()
        for _, line in sorted(scored, key=lambda item: (-item[0], item[1])):
            normalized = line.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            if len(line) > 220:
                line = line[:217].rstrip() + "..."
            lines.append(line)
            if len(lines) >= limit:
                break
        return lines

    def _compose_entity_description(
        self,
        *,
        contract: QueryContract,
        target: str,
        label: str,
        paper: CandidatePaper,
        evidence: list[EvidenceBlock],
    ) -> str:
        summary = self._paper_summary_text(paper.paper_id)
        requested_fields = list(contract.requested_fields)
        prompt = (
            f"target={target}\n"
            f"entity_type={label}\n"
            f"continuation_mode={contract.continuation_mode}\n"
            f"requested_fields={json.dumps(requested_fields, ensure_ascii=False)}\n"
            f"paper_title={paper.title}\n"
            f"summary={summary}\n"
            f"evidence={json.dumps([item.snippet[:240] for item in evidence[:4]], ensure_ascii=False)}"
        )
        llm_text = self.clients.invoke_text(
            system_prompt=(
                "你是论文实体解释器。请根据给定论文摘要、requested_fields 和证据，用一段简洁中文回答。"
                "不要输出 Markdown 标题、列表、链接或引用，不要使用 #、####、项目符号。"
                "控制在 2-4 句内。"
                "如果 continuation_mode 是 followup，或者 requested_fields 包含 mechanism/workflow/objective/reward_signal，"
                "优先解释它如何工作、优化什么、依赖什么奖励或训练信号，不要只重复泛泛定义。"
                "如果证据同时包含“某篇论文使用它”和“它本身的定义/机制”，优先解释技术本身，再补充应用场景。"
                "不要编造。"
            ),
            human_prompt=prompt,
            fallback="",
        )
        if llm_text:
            return llm_text
        mechanism_lines = self._entity_supporting_lines(evidence, kind="mechanism")
        application_lines = self._entity_supporting_lines(evidence, kind="application")
        if any(field in {"mechanism", "workflow", "objective", "reward_signal"} for field in requested_fields):
            parts = [f"{target} 可以定位为 `{label}`。"]
            if mechanism_lines:
                parts.append("它的工作机制可以概括为：")
                parts.extend([f"- {line}" for line in mechanism_lines[:2]])
            if application_lines:
                parts.append("在当前语料里，它被应用在：")
                parts.extend([f"- {line}" for line in application_lines[:1]])
            return "\n".join(parts).strip()
        if summary:
            snippet = " ".join(summary.split())[:180]
            return f"{target} 可以定位为 `{label}`。相关论文《{paper.title}》提到：{snippet}"
        return f"{target} 可以定位为 `{label}`，相关论文是《{paper.title}》。"
