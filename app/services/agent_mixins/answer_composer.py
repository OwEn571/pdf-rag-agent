from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Callable

from app.domain.models import (
    AssistantCitation,
    CandidatePaper,
    Claim,
    EvidenceBlock,
    QueryContract,
    SessionContext,
    VerificationReport,
)
from app.services.contracts.context import (
    contract_answer_slots,
    contract_note_value,
    contract_notes,
)
from app.services.clarification.questions import build_agent_clarification_question
from app.services.answers.evidence_presentation import claim_evidence_ids, citations_from_doc_ids
from app.services.answers.library_recommendations import (
    clean_library_recommendation_criteria_note,
    compose_library_status_markdown,
    diversify_library_recommendations,
    library_paper_preview_lines,
    library_recommendation_reason,
    library_status_query_wants_listing,
    library_status_query_wants_recommendation,
    library_unique_paper_metadata,
    llm_select_library_recommendations,
    rank_library_papers_for_recommendation,
    recent_library_recommendation_titles,
    select_library_recommendations,
    split_library_authors,
)
from app.services.library.metadata_sql import (
    execute_library_metadata_sql,
    fallback_library_metadata_sql_answer,
    library_metadata_rows,
    library_metadata_sql_schema_description,
    sqlite_row_to_payload,
    validate_library_metadata_sql,
)
from app.services.answers.formula import (
    auto_resolved_candidate_notice,
    compose_formula_answer,
    format_formula_description,
    format_formula_symbol,
    formula_term_lines,
    formula_variable_lines,
    normalize_markdown_math_artifacts,
)
from app.services.answers.followup import compose_followup_research_answer, followup_public_reason
from app.services.answers.paper import (
    compose_metric_value_answer,
    compose_paper_summary_results_answer,
    metric_lines_from_claims,
    paper_result_core_points,
)
from app.services.infra.prompt_safety import DOCUMENT_SAFETY_INSTRUCTION, wrap_untrusted_document_text
from app.services.planning.research import research_plan_context_from_contract
from app.services.contracts.session_context import agent_session_conversation_context, session_llm_history_messages
from app.services.answers.topology import clean_topology_public_text, compose_topology_recommendation_answer
from app.services.library.zotero_sqlite import ZoteroSQLiteReader


@lru_cache(maxsize=1)
def _load_assistant_self_knowledge() -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "assistant_self.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


class AnswerComposerMixin:
    def _compose_answer(
        self,
        *,
        contract: QueryContract,
        claims: list[Claim],
        evidence: list[EvidenceBlock],
        papers: list[CandidatePaper],
        verification: VerificationReport,
        session: SessionContext | None = None,
        stream_callback: Callable[[str], None] | None = None,
        logprob_callback: Callable[[list[float]], None] | None = None,
        request_logprobs: bool = False,
    ) -> tuple[str, list[AssistantCitation]]:
        if verification.status == "clarify":
            return build_agent_clarification_question(
                contract=contract,
                session=session or SessionContext(session_id="clarify"),
                clients=self.clients,
                settings=self.settings,
            ), []
        if verification.status == "retry":
            return self._compose_retry_answer(contract=contract, verification=verification, claims=claims), []
        _screened_ids = {p.paper_id for p in papers if p.paper_id}
        citations = citations_from_doc_ids(
            claim_evidence_ids(claims),
            evidence,
            block_doc_lookup=self.retriever.block_doc_by_id,
            paper_doc_lookup=self.retriever.paper_doc_by_id,
            screened_paper_ids=_screened_ids,
        )
        if not citations and papers:
            fallback_doc_ids = [doc_id for paper in papers[:2] for doc_id in paper.doc_ids[:1]]
            citations = citations_from_doc_ids(
                fallback_doc_ids,
                evidence,
                block_doc_lookup=self.retriever.block_doc_by_id,
                paper_doc_lookup=self.retriever.paper_doc_by_id,
                screened_paper_ids=_screened_ids,
            )
        if contract.allow_web_search:
            citations = self._prioritize_web_citations(claims=claims, evidence=evidence, citations=citations)
        # Direct origin answer: when contract is origin_lookup and papers exist,
        # produce answer from top-ranked paper without requiring LLM composition.
        if contract.relation == "origin_lookup" and not claims and papers:
            return self._compose_origin_answer_from_papers(contract=contract, papers=papers), citations
        if any(claim.claim_type == "topology_recommendation" for claim in claims):
            structured_answer = self._compose_structured_research_answer(
                contract=contract,
                claims=claims,
                evidence=evidence,
            )
            if structured_answer:
                return structured_answer, citations
        if self._should_try_llm_research_composer(claims):
            try:
                llm_answer = self._compose_research_answer_markdown(
                    contract=contract,
                    claims=claims,
                    evidence=evidence,
                    papers=papers,
                    citations=citations,
                    verification=verification,
                    session=session,
                    stream_callback=stream_callback,
                    logprob_callback=logprob_callback,
                    request_logprobs=request_logprobs,
                )
            except RuntimeError:
                llm_answer = ""
            if llm_answer:
                return llm_answer, citations
        if any(claim.claim_type == "followup_research" for claim in claims):
            structured_answer = self._compose_structured_research_answer(
                contract=contract,
                claims=claims,
                evidence=evidence,
            )
            if structured_answer:
                return structured_answer, citations
        if self._should_try_llm_research_composer(claims, include_deterministic=True):
            try:
                llm_answer = self._compose_research_answer_markdown(
                    contract=contract,
                    claims=claims,
                    evidence=evidence,
                    papers=papers,
                    citations=citations,
                    verification=verification,
                    session=session,
                    stream_callback=stream_callback,
                    logprob_callback=logprob_callback,
                    request_logprobs=request_logprobs,
                )
            except RuntimeError:
                llm_answer = ""
            if llm_answer:
                return llm_answer, citations
            if any(claim.claim_type == "entity_definition" for claim in claims):
                fallback_answer = self._compose_entity_answer_markdown(
                    contract=contract,
                    claims=claims,
                    evidence=evidence,
                    citations=citations,
                )
                if fallback_answer:
                    return fallback_answer, citations
        structured_answer = self._compose_structured_research_answer(
            contract=contract,
            claims=claims,
            evidence=evidence,
        )
        if structured_answer:
            return structured_answer, citations
        # Last-resort fallback: if papers exist, produce a paper-backed answer
        # without requiring claims (e.g., when solver returned nothing useful).
        if not claims and papers:
            return self._compose_papers_fallback(contract=contract, papers=papers, evidence=evidence), citations
        llm_answer = self._compose_research_answer_markdown(
            contract=contract,
            claims=claims,
            evidence=evidence,
            papers=papers,
            citations=citations,
            verification=verification,
            session=session,
            stream_callback=stream_callback,
            logprob_callback=logprob_callback,
            request_logprobs=request_logprobs,
        )
        return llm_answer, citations

    @staticmethod
    def _claims_from_schema_solver(claims: list[Claim]) -> bool:
        return any(dict(claim.structured_data or {}).get("source") == "schema_claim_solver" for claim in claims)

    @classmethod
    def _should_try_llm_research_composer(cls, claims: list[Claim], *, include_deterministic: bool = False) -> bool:
        if cls._claims_from_schema_solver(claims):
            return True
        if not include_deterministic:
            return False
        llm_friendly_claims = {
            "entity_definition",
            "paper_summary",
            "metric_value",
            "metric_context",
            "text_answer",
        }
        return any(claim.claim_type in llm_friendly_claims for claim in claims)

    def _prioritize_web_citations(
        self,
        *,
        claims: list[Claim],
        evidence: list[EvidenceBlock],
        citations: list[AssistantCitation],
    ) -> list[AssistantCitation]:
        web_doc_ids: list[str] = []
        for claim in claims:
            if claim.claim_type == "web_research":
                web_doc_ids.extend(claim.evidence_ids)
        if not web_doc_ids:
            return citations
        web_citations = citations_from_doc_ids(
            web_doc_ids,
            evidence,
            block_doc_lookup=self.retriever.block_doc_by_id,
            paper_doc_lookup=self.retriever.paper_doc_by_id,
        )
        seen = {(item.title, item.file_path, item.page) for item in web_citations}
        ordered = list(web_citations)
        for item in citations:
            key = (item.title, item.file_path, item.page)
            if key not in seen:
                seen.add(key)
                ordered.append(item)
        return ordered

    def _compose_retry_answer(
        self,
        *,
        contract: QueryContract,
        verification: VerificationReport,
        claims: list[Claim],
    ) -> str:
        missing = {str(item) for item in verification.missing_fields}
        target = contract.targets[0] if contract.targets else "当前目标"
        if "target_aligned_formula" in missing or "formula" in missing:
            rejected_formula = next((str(claim.value or "").strip() for claim in claims if claim.claim_type == "formula"), "")
            selected_title = contract_note_value(contract, prefix="memory_title=")
            unsupported = [str(item).strip() for item in verification.unsupported_claims if str(item).strip()]
            lines = [
                "## 不能确认公式",
                "",
                f"我不能把刚才检索到的公式当作 `{target}` 的公式来回答，因为证据校验没有通过目标对齐。",
            ]
            if unsupported:
                lines.append("校验器标出的不支持项：" + "；".join(unsupported[:3]) + "。")
            elif rejected_formula:
                lines.append("我会保留这条候选公式作为待复查线索，但不会把它写成已确认结论。")
            if selected_title:
                lines.append(f"我已经限定在《{selected_title}》里查找；如果该 PDF 中没有目标对齐的公式，我会明确说未找到。")
            lines.extend(
                [
                    "",
                    f"我需要重新定位包含 `{target}` 明确定义和公式推导的段落；如果本地 PDF 里只有文字说明而没有公式，我应该明确说“未找到公式”，而不是套用其它方法的公式。",
                ]
            )
            return "\n".join(lines)
        missing_text = "、".join(verification.missing_fields) or "关键证据"
        return f"## 证据不足\n\n当前检索结果还不能可靠覆盖：{missing_text}。我不会把未通过 grounding 的 claim 写成结论。"

    def _compose_research_answer_markdown(
        self,
        *,
        contract: QueryContract,
        claims: list[Claim],
        evidence: list[EvidenceBlock],
        papers: list[CandidatePaper],
        citations: list[AssistantCitation],
        verification: VerificationReport,
        session: SessionContext | None = None,
        stream_callback: Callable[[str], None] | None = None,
        logprob_callback: Callable[[list[float]], None] | None = None,
        request_logprobs: bool = False,
    ) -> str:
        if self.clients.chat is None:
            raise RuntimeError("Research answer generation requires an available chat model.")
        if not claims:
            raise RuntimeError("Research answer generation requires verified claims.")
        evidence_context = self._llm_evidence_context(contract=contract, evidence=evidence)
        prompt = json.dumps(
            {
                "query": contract.clean_query,
                "relation": contract.relation,
                "continuation_mode": contract.continuation_mode,
                "answer_shape": contract.answer_shape,
                "targets": contract.targets,
                "requested_fields": contract.requested_fields,
                "required_modalities": contract.required_modalities,
                "answer_slots": contract_answer_slots(contract),
                "claim_types": list(dict.fromkeys(item.claim_type for item in claims)),
                "notes": contract_notes(contract),
                "verification": verification.model_dump(),
                "claims": [item.model_dump() for item in claims],
                "papers": [
                    {"title": item.title, "year": item.year, "paper_id": item.paper_id}
                    for item in papers[:4]
                ],
                "evidence": [
                    {
                        "doc_id": item.doc_id,
                        "paper_id": item.paper_id,
                        "title": item.title,
                        "page": item.page,
                        "block_type": item.block_type,
                        "caption": item.caption,
                        "snippet": self._llm_evidence_snippet(item=item, contract=contract),
                    }
                    for item in evidence_context
                ],
                "citations": [
                    {
                        "doc_id": item.doc_id,
                        "paper_id": item.paper_id,
                        "title": item.title,
                        "authors": item.authors,
                        "year": item.year,
                        "page": item.page,
                        "block_type": item.block_type,
                    }
                    for item in citations[:4]
                ],
                "conversation_context": agent_session_conversation_context(
                    session,
                    settings=self.settings,
                    max_chars=16000,
                )
                if session is not None
                else {},
            },
            ensure_ascii=False,
        )
        system_prompt = (
            "你是论文研究助手的最终回答整理器。"
            "只基于提供的 claims / evidence / citations 组织答案，不要使用你记忆里的论文知识。"
            "请输出简洁中文 Markdown。"
            "数学公式使用标准 KaTeX LaTeX：独立公式用 $$...$$，行内变量用 $...$。"
            "不要输出 nabla、pi 这类 Unicode 粘连符号，要写成 LaTeX 命令。"
            "引用论文时用《标题》（年份）格式，引用证据时用「具体数据/结论」（来源论文）。"
            "禁止输出 <document> 标签、doc_id、evidence ID 或其他内部标识符。"
            "根据不同 claim_types 调整结构："
            "- origin/paper_title: 明确写出论文标题和年份。"
            "- formula: 优先使用 claims 中的 formula_latex，给出公式并解释变量。"
            "- definition/mechanism: 先 1-2 句说明是什么，再列关键特性。"
            "- summary/results: 先一句话概括，再分核心结论、实验结果、证据边界。"
            "- metric_value: 给出可确认数值，说明数据集和对比对象。"
            "- followup_papers: 先点明种子论文，再列后续工作及关系。"
            "- figure_conclusion: 总结图表中的比较对象和总体结论。"
            "如果 evidence 有不确定性就明确写出，不要编造。"
            "论文标题加《》，禁止输出未在 evidence/citations 中出现的外链。"
            f"{DOCUMENT_SAFETY_INSTRUCTION}"
        )
        if stream_callback is not None and hasattr(self.clients, "stream_text"):
            stream_kwargs: dict[str, Any] = {
                "system_prompt": system_prompt,
                "human_prompt": prompt,
                "on_delta": stream_callback,
                "fallback": "",
            }
            if request_logprobs or logprob_callback is not None:
                stream_kwargs["request_logprobs"] = request_logprobs
                stream_kwargs["on_logprobs"] = logprob_callback
            response_text = self.clients.stream_text(**stream_kwargs).strip()
        elif session is not None and callable(getattr(self.clients, "invoke_text_messages", None)):
            response_text = self.clients.invoke_text_messages(
                system_prompt=(
                    system_prompt
                    + "\n\n以下非语言回答上下文只用于 grounding 和组织答案，不是用户新问题：\n"
                    + prompt
                ),
                messages=[
                    *session_llm_history_messages(session, max_turns=6, answer_limit=900),
                    {"role": "user", "content": contract.clean_query},
                ],
                fallback="",
            ).strip()
        else:
            response_text = self.clients.invoke_text(
                system_prompt=system_prompt,
                human_prompt=prompt,
                fallback="",
            ).strip()
        if not response_text:
            raise RuntimeError("Research answer composer failed: upstream LLM response was empty.")
        return self._clean_common_ocr_artifacts(response_text)

    @staticmethod
    def _clean_common_ocr_artifacts(text: str) -> str:
        replacements = {
            "GPT-40-mini": "GPT-4o-mini",
            "GPT-40 Mini": "GPT-4o-mini",
            "GPT-40 mini": "GPT-4o-mini",
            "GPT-4O-mini": "GPT-4o-mini",
            "GPT-4O Mini": "GPT-4o-mini",
            "GPT-4O mini": "GPT-4o-mini",
            "GPT-40": "GPT-4o",
            "GPT-4O": "GPT-4o",
        }
        cleaned = text
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)
        return AnswerComposerMixin._normalize_markdown_math_artifacts(cleaned)

    @staticmethod
    def _normalize_markdown_math_artifacts(text: str) -> str:
        return normalize_markdown_math_artifacts(text)

    @staticmethod
    def _composer_goals(contract: QueryContract) -> set[str]:
        return set(research_plan_context_from_contract(contract).goals) or {"answer"}

    @staticmethod
    def _llm_evidence_context(*, contract: QueryContract, evidence: list[EvidenceBlock]) -> list[EvidenceBlock]:
        fields = {str(item) for item in contract.requested_fields}
        modalities = {str(item) for item in contract.required_modalities}
        if fields & {"summary", "results", "metric_value"} or modalities & {"table", "caption"}:
            limit = 36
        elif fields & {"followup_papers", "candidate_relationship", "best_topology", "relevant_papers"}:
            limit = 28
        elif fields & {"formula", "definition", "mechanism", "role_in_context"} or modalities & {"figure", "page_text"}:
            limit = 24
        else:
            limit = 18
        selected: list[EvidenceBlock] = []
        seen: set[str] = set()
        for item in evidence:
            if item.doc_id in seen:
                continue
            seen.add(item.doc_id)
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _llm_evidence_snippet(*, item: EvidenceBlock, contract: QueryContract) -> str:
        fields = {str(field) for field in contract.requested_fields}
        if item.block_type in {"table", "caption", "figure"}:
            limit = 760
        elif fields & {"summary", "results", "metric_value", "figure_conclusion"}:
            limit = 620
        else:
            limit = 460
        return wrap_untrusted_document_text(
            item.snippet,
            doc_id=item.doc_id,
            title=item.title,
            source=item.block_type or "pdf",
            max_chars=limit,
        )

    def _compose_structured_research_answer(
        self,
        *,
        contract: QueryContract,
        claims: list[Claim],
        evidence: list[EvidenceBlock],
    ) -> str:
        web_answer = self._compose_web_research_answer_if_needed(contract=contract, claims=claims, evidence=evidence)
        if web_answer:
            return web_answer
        claim_types = {claim.claim_type for claim in claims}
        if "origin" in claim_types:
            return self._compose_origin_answer(claims=claims)
        if "formula" in claim_types:
            return self._compose_formula_answer(claims=claims, contract=contract)
        if claim_types & {"metric_value", "metric_context"}:
            return self._compose_metric_value_answer(contract=contract, claims=claims)
        if "followup_research" in claim_types:
            return self._compose_followup_research_answer(claims=claims)
        if "paper_summary" in claim_types:
            return self._compose_paper_summary_results_answer(contract=contract, claims=claims)
        if "topology_recommendation" in claim_types:
            return self._compose_topology_recommendation_answer(contract=contract, claims=claims, evidence=evidence)
        return ""

    def _compose_web_research_answer_if_needed(
        self,
        *,
        contract: QueryContract,
        claims: list[Claim],
        evidence: list[EvidenceBlock],
    ) -> str:
        web_claim = next((item for item in claims if item.claim_type == "web_research"), None)
        if web_claim is None:
            return ""
        non_web_claims = [item for item in claims if item.claim_type != "web_research"]
        goals = self._composer_goals(contract)
        if non_web_claims and not goals & {"answer", "general_answer", "recommended_papers", "followup_papers"}:
            return ""
        web_evidence = [item for item in evidence if item.doc_id in set(web_claim.evidence_ids)]
        if not web_evidence:
            return ""
        if self.clients.chat is None:
            body = "\n".join(f"- {item.title}: {item.snippet[:220]} ({item.file_path})" for item in web_evidence[:6])
            return f"## Web 检索结果\n\n{body}"
        payload = json.dumps(
            {
                "query": contract.clean_query,
                "relation": contract.relation,
                "targets": contract.targets,
                "web_results": [
                    {
                        "title": item.title,
                        "url": item.file_path,
                        "snippet": wrap_untrusted_document_text(
                            item.snippet,
                            doc_id=item.doc_id,
                            title=item.title,
                            source="web",
                            max_chars=900,
                        ),
                    }
                    for item in web_evidence[:10]
                ],
            },
            ensure_ascii=False,
        )
        answer = self.clients.invoke_text(
            system_prompt=(
                "你是论文研究助手的 Web 证据整理器。"
                "只基于输入的 web_results 回答，不要使用记忆补充事实。"
                f"{DOCUMENT_SAFETY_INSTRUCTION}"
                "如果是最新/新闻/新论文问题，要明确这是基于当前 Web 检索结果。"
                "优先用简洁中文 Markdown，总结要点并在每条后写来源标题。"
                "如果证据不足，就说明还需要继续搜索，而不是编造。"
            ),
            human_prompt=payload,
            fallback="",
        ).strip()
        if answer:
            return answer
        body = "\n".join(f"- {item.title}: {item.snippet[:220]} ({item.file_path})" for item in web_evidence[:6])
        return f"## Web 检索结果\n\n{body}"

    def _compose_metric_value_answer(self, *, contract: QueryContract, claims: list[Claim]) -> str:
        return compose_metric_value_answer(contract=contract, claims=claims)

    def _compose_paper_summary_results_answer(self, *, contract: QueryContract, claims: list[Claim]) -> str:
        return compose_paper_summary_results_answer(contract=contract, claims=claims)

    @staticmethod
    def _compose_origin_answer(*, claims: list[Claim]) -> str:
        if not claims:
            return ""
        claim = next((item for item in claims if item.claim_type == "origin"), claims[0])
        title = str(dict(claim.structured_data or {}).get("paper_title") or claim.value or "").strip()
        if not title:
            return ""
        year = str(dict(claim.structured_data or {}).get("year") or "").strip()
        entity = str(claim.entity or "该对象").strip()
        suffix = f"（{year}）" if year else ""
        return f"## 结论\n\n{entity} 最早由论文《{title}》{suffix}提出。"

    @staticmethod
    def _compose_origin_answer_from_papers(
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
    ) -> str:
        if not papers:
            return ""
        top = papers[0]
        entity = str(contract.targets[0] if contract.targets else "该技术").strip()
        year_suffix = f"（{top.year}）" if top.year else ""
        lines = [
            "## 结论",
            "",
            f"{entity} 最早由论文《{top.title}》{year_suffix}提出。",
        ]
        if len(papers) > 1:
            lines.append("")
            lines.append("相关论文还包括：")
            for paper in papers[1:4]:
                py = f"（{paper.year}）" if paper.year else ""
                lines.append(f"- 《{paper.title}》{py}")
        return "\n".join(lines)

    @staticmethod
    def _compose_papers_fallback(
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> str:
        """Generate a conservative answer from top papers when claims are empty."""
        if contract.relation == "origin_lookup":
            return AnswerComposerMixin._compose_origin_answer_from_papers(
                contract=contract, papers=papers
            )
        top = papers[0]
        target = str(contract.targets[0] if contract.targets else contract.clean_query).strip()
        year = f"（{top.year}）" if top.year else ""
        evidence_snippets = [
            item.snippet[:200] for item in evidence[:3] if item.paper_id == top.paper_id
        ]
        lines = [
            f"## {target}",
            "",
            f"根据检索结果，最相关的论文是《{top.title}》{year}。",
        ]
        if evidence_snippets:
            lines.append("")
            lines.append("该论文中的相关段落：")
            for i, snippet in enumerate(evidence_snippets[:2], 1):
                lines.append(f"\n> {snippet}")
            lines.append("")
            lines.append("如需更详细的解释，可以追问具体方面（如工作机制、实验结果等）。")
        return "\n".join(lines)

    @classmethod
    def _compose_followup_research_answer(cls, *, claims: list[Claim]) -> str:
        _ = cls
        return compose_followup_research_answer(claims=claims)

    @staticmethod
    def _followup_public_reason(item: dict[str, object]) -> str:
        return followup_public_reason(item)

    def _compose_topology_recommendation_answer(
        self,
        *,
        contract: QueryContract,
        claims: list[Claim],
        evidence: list[EvidenceBlock],
    ) -> str:
        _ = contract
        return compose_topology_recommendation_answer(claims=claims, evidence=evidence)

    @staticmethod
    def _clean_topology_public_text(text: str) -> str:
        return clean_topology_public_text(text)

    @staticmethod
    def _compose_formula_answer(*, claims: list[Claim], contract: QueryContract | None = None) -> str:
        return compose_formula_answer(claims=claims, contract=contract)

    @staticmethod
    def _auto_resolved_candidate_notice(contract: QueryContract | None) -> str:
        return auto_resolved_candidate_notice(contract)

    @staticmethod
    def _formula_term_lines(claim: Claim) -> list[str]:
        return formula_term_lines(claim)

    @staticmethod
    def _formula_variable_lines(value: object) -> list[str]:
        return formula_variable_lines(value)

    @staticmethod
    def _format_formula_symbol(symbol: str) -> str:
        return format_formula_symbol(symbol)

    @staticmethod
    def _format_formula_description(description: str) -> str:
        return format_formula_description(description)

    @staticmethod
    def _metric_lines_from_claims(claims: list[Claim]) -> list[str]:
        return metric_lines_from_claims(claims)

    @staticmethod
    def _paper_result_core_points(*, target: str, support_text: str) -> list[str]:
        return paper_result_core_points(target=target, support_text=support_text)

    def _compose_conversation_response(
        self,
        *,
        contract: QueryContract,
        query: str,
        session: SessionContext,
    ) -> str:
        # Dispatch table: relation → handler returning a string.
        # Add new static or composed responses here; unknown relations
        # fall through to the LLM-based generic composer below.
        _conversation_composers: dict[str, Any] = {
            "self_identity": lambda: (
                "我是你的论文研究助手，可以围绕 Zotero 论文库做检索、总结、对比、术语解释、公式/图表/表格解读和后续工作追踪。"
            ),
            "capability": lambda: (
                "我可以帮你做这些事：\n\n"
                "- 论文总结：提炼核心贡献、方法和实验结果。\n"
                "- 对比分析：比较多篇论文的方法、指标、优缺点和适用场景。\n"
                "- 概念/术语/定义：解释论文里的技术名词、模型、数据集和 benchmark。\n"
                "- 公式、表格、图表：定位并解释公式变量、表格指标、figure 结论。\n"
                "- 后续工作追踪：从一篇论文、数据集或方法出发，找相关扩展研究。"
            ),
            "library_status": lambda: self._compose_library_status_response(query=query),
            "library_recommendation": lambda: self._compose_library_recommendation_response(query=query, session=session),
        }
        handler = _conversation_composers.get(contract.relation)
        if handler is not None:
            return handler()
        if self.clients.chat is None:
            raise RuntimeError("Conversation response generation requires an available chat model.")
        response_text = self.clients.invoke_text(
            system_prompt=(
                "你是论文研究助手的对话回复整理器。"
                "请基于当前 intent、用户问题和最近会话上下文，用简洁中文 Markdown 回复。"
                "不要编造论文事实，不要输出研究结论，只处理元对话。"
                "根据 intent 类型选择回复风格：能力介绍用 3-5 条要点，库状态用 assistant_self_knowledge 规则，"
                "澄清类问题帮助用户把问题说清楚，一般问题就自然回答。"
            ),
            human_prompt=json.dumps(
                {
                    "query": query,
                    "intent": str(contract.relation),
                    "assistant_self_knowledge": _load_assistant_self_knowledge(),
                    "conversation_context": agent_session_conversation_context(
                        session,
                        settings=self.settings,
                    ),
                },
                ensure_ascii=False,
            ),
            fallback="",
        ).strip()
        if not response_text:
            raise RuntimeError("Conversation response generation failed: upstream LLM response was empty.")
        return response_text

    def _compose_library_status_response(self, *, query: str = "") -> str:
        docs = library_unique_paper_metadata(paper_documents=list(self.retriever.paper_documents()))
        collection_paths: dict[str, list[str]] = {}
        try:
            collection_paths = ZoteroSQLiteReader(self.settings).read_attachment_collection_paths()
        except Exception:  # noqa: BLE001
            collection_paths = {}
        return compose_library_status_markdown(
            query=query,
            docs=docs,
            collection_paths=collection_paths,
        )

    def _compose_library_metadata_query_response(self, *, query: str = "") -> dict[str, Any]:
        result = self._run_library_metadata_sql_query(query=query)
        if result.get("error"):
            result["answer"] = ""
            return result
        answer = self._compose_library_metadata_sql_answer(query=query, result=result)
        result["answer"] = answer
        return result

    def _run_library_metadata_sql_query(self, *, query: str) -> dict[str, Any]:
        rows = self._library_metadata_rows()
        if not rows:
            return {
                "sql": "",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "error": "empty_library_metadata",
            }
        payload = self._plan_library_metadata_sql(query=query, sample_rows=rows[:5])
        raw_sql = str(payload.get("sql", "") if isinstance(payload, dict) else "").strip()
        try:
            sql = self._validate_library_metadata_sql(raw_sql)
            query_result = self._execute_library_metadata_sql(sql=sql, paper_rows=rows, max_rows=80)
        except Exception as exc:  # noqa: BLE001
            return {
                "sql": raw_sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "error": str(exc),
            }
        query_result["planner_reason"] = str(payload.get("reason", "") if isinstance(payload, dict) else "").strip()
        return query_result

    def _plan_library_metadata_sql(self, *, query: str, sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if self.clients.chat is None:
            return {}
        schema = self._library_metadata_sql_schema_description()
        sample_payload = [
            {
                "paper_id": row.get("paper_id", ""),
                "title": row.get("title", ""),
                "authors": row.get("authors", ""),
                "year": row.get("year", ""),
                "categories": row.get("categories", ""),
                "tags": row.get("tags", ""),
                "has_pdf": row.get("has_pdf", 0),
            }
            for row in sample_rows
        ]
        return self.clients.invoke_json(
            system_prompt=(
                "你是论文库元信息 SQL 规划器，只负责把用户问题转换成只读 SQLite SELECT 查询。"
                "你不能直接回答用户；最终回答由另一个步骤根据 SQL 结果生成。"
                "只能使用给定 schema 中的表和字段。"
                "只允许生成一条 SELECT 或 WITH ... SELECT 语句；不要生成 PRAGMA、DDL、DML、注释或多语句。"
                "如果用户问是否存在、有哪些、某作者/某年份/某标签/某分类论文，优先返回匹配论文行，字段包含 title、year、authors、paper_id。"
                "如果用户明确问数量、总数或范围，可以返回 COUNT/MIN/MAX 等聚合。"
                "文本匹配使用 LOWER(column) LIKE LOWER('%关键词%')；精确年份使用 year_int。"
                "对于可能返回很多论文的查询，加 LIMIT 20 到 50。"
                "只输出 JSON：sql, reason。"
            ),
            human_prompt=json.dumps(
                {
                    "user_query": query,
                    "schema": schema,
                    "sample_rows": sample_payload,
                },
                ensure_ascii=False,
            ),
            fallback={},
        )

    @staticmethod
    def _library_metadata_sql_schema_description() -> dict[str, Any]:
        return library_metadata_sql_schema_description()

    def _library_metadata_rows(self) -> list[dict[str, Any]]:
        collection_paths: dict[str, list[str]] = {}
        try:
            collection_paths = ZoteroSQLiteReader(self.settings).read_attachment_collection_paths()
        except Exception:  # noqa: BLE001
            collection_paths = {}
        return library_metadata_rows(
            paper_documents=list(self.retriever.paper_documents()),
            collection_paths=collection_paths,
        )

    @staticmethod
    def _split_library_authors(authors: str) -> list[str]:
        return split_library_authors(authors)

    @staticmethod
    def _validate_library_metadata_sql(sql: str) -> str:
        return validate_library_metadata_sql(sql)

    def _execute_library_metadata_sql(
        self,
        *,
        sql: str,
        paper_rows: list[dict[str, Any]],
        max_rows: int,
    ) -> dict[str, Any]:
        return execute_library_metadata_sql(sql=sql, paper_rows=paper_rows, max_rows=max_rows)

    @staticmethod
    def _sqlite_row_to_payload(row: Any) -> dict[str, Any]:
        return sqlite_row_to_payload(row)

    def _compose_library_metadata_sql_answer(self, *, query: str, result: dict[str, Any]) -> str:
        if self.clients.chat is not None:
            answer = self.clients.invoke_text(
                system_prompt=(
                    "你是论文库元信息查询结果解释器。"
                    "只能基于 SQL_result 回答，不要使用外部常识，不要根据现实年份判断库中有没有论文。"
                    "如果 SQL_result.rows 为空，就明确说当前本地索引没有查到匹配记录。"
                    "如果 rows 是论文列表，直接回答是否查到，并必须说明 SQL_result.row_count；"
                    "再按 SQL_result.rows 的原始顺序列出最多 5 篇论文的标题、年份、作者或 paper_id；不要重排。"
                    "如果 SQL_result.truncated=true，说明展示的是前若干条。"
                    "如果 rows 是聚合结果，用自然语言解释这些数值。"
                    "回答保持简洁中文 Markdown；可以说明“我查的是当前本地 paper index 元信息”。"
                ),
                human_prompt=json.dumps(
                    {
                        "user_query": query,
                        "SQL_result": {
                            "sql": result.get("sql", ""),
                            "columns": result.get("columns", []),
                            "rows": result.get("rows", [])[:40],
                            "row_count": result.get("row_count", 0),
                            "truncated": result.get("truncated", False),
                            "planner_reason": result.get("planner_reason", ""),
                        },
                    },
                    ensure_ascii=False,
                ),
                fallback="",
            ).strip()
            if answer:
                return answer
        return self._fallback_library_metadata_sql_answer(query=query, result=result)

    @staticmethod
    def _fallback_library_metadata_sql_answer(*, query: str, result: dict[str, Any]) -> str:
        return fallback_library_metadata_sql_answer(query=query, result=result)

    def _compose_library_recommendation_response(self, *, query: str = "", session: SessionContext | None = None) -> str:
        docs = library_unique_paper_metadata(paper_documents=list(self.retriever.paper_documents()))
        candidate_pool = self._rank_library_papers_for_recommendation(docs=docs, query=query, limit=18)
        recommendations, criteria_note = self._select_library_recommendations(
            query=query,
            candidates=candidate_pool,
            session=session,
            limit=5,
        )
        if not recommendations:
            return "当前索引里没有足够的论文元数据支撑推荐。"
        primary = recommendations[0]
        primary_year = f"（{primary['year']}）" if primary.get("year") else ""
        intro = self._clean_library_recommendation_criteria_note(
            criteria_note,
            has_recent_recommendations=bool(self._recent_library_recommendation_titles(session)),
        )
        lines = [
            "## 我会优先看这几篇",
            "",
            f"你库里现在有 **{len(docs)} 篇**论文；不全铺开，{intro}",
            "",
            f"**首选**：《{primary['title']}》{primary_year}。{primary['reason']}",
        ]
        if len(recommendations) > 1:
            lines.extend(["", "## 接着看", ""])
            for item in recommendations[1:]:
                year_suffix = f"（{item['year']}）" if item.get("year") else ""
                lines.append(f"- 《{item['title']}》{year_suffix}：{item['reason']}")
        lines.extend(
            [
                "",
                "如果你想按某条线读，我建议下一步直接问我“按 RAG / alignment / personalization / RLHF 分一条阅读路线”。",
            ]
        )
        return "\n".join(lines)

    def _select_library_recommendations(
        self,
        *,
        query: str,
        candidates: list[dict[str, str]],
        session: SessionContext | None,
        limit: int,
    ) -> tuple[list[dict[str, str]], str]:
        return select_library_recommendations(
            query=query,
            candidates=candidates,
            session=session,
            limit=limit,
            clients=self.clients,
            settings=self.settings,
        )

    def _llm_select_library_recommendations(
        self,
        *,
        query: str,
        candidates: list[dict[str, str]],
        session: SessionContext | None,
        recent_titles: list[str],
        limit: int,
    ) -> tuple[list[dict[str, str]], str]:
        return llm_select_library_recommendations(
            query=query,
            candidates=candidates,
            session=session,
            recent_titles=recent_titles,
            limit=limit,
            clients=self.clients,
            settings=self.settings,
        )

    @staticmethod
    def _clean_library_recommendation_criteria_note(note: str, *, has_recent_recommendations: bool) -> str:
        return clean_library_recommendation_criteria_note(
            note,
            has_recent_recommendations=has_recent_recommendations,
        )

    def _diversify_library_recommendations(
        self,
        *,
        candidates: list[dict[str, str]],
        recent_titles: list[str],
        query: str,
        limit: int,
    ) -> list[dict[str, str]]:
        return diversify_library_recommendations(
            candidates=candidates,
            recent_titles=recent_titles,
            query=query,
            limit=limit,
        )

    def _recent_library_recommendation_titles(self, session: SessionContext | None) -> list[str]:
        return recent_library_recommendation_titles(session)

    @staticmethod
    def _library_status_query_wants_recommendation(query: str) -> bool:
        return library_status_query_wants_recommendation(query)

    @staticmethod
    def _library_status_query_wants_listing(query: str) -> bool:
        return library_status_query_wants_listing(query)

    @staticmethod
    def _library_paper_preview_lines(
        *,
        docs: list[dict[str, object]],
        collection_paths: dict[str, list[str]],
        limit: int,
    ) -> list[str]:
        return library_paper_preview_lines(docs=docs, collection_paths=collection_paths, limit=limit)

    def _rank_library_papers_for_recommendation(
        self,
        *,
        docs: list[dict[str, object]],
        query: str,
        limit: int = 3,
    ) -> list[dict[str, str]]:
        return rank_library_papers_for_recommendation(docs=docs, query=query, limit=limit)

    @staticmethod
    def _library_recommendation_reason(*, title: str, year: str, summary: str, tags: list[str]) -> str:
        return library_recommendation_reason(title=title, year=year, summary=summary, tags=tags)
