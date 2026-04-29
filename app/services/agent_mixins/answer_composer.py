from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import sqlite3
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
from app.services.evidence_presentation import extract_topology_terms
from app.services.prompt_safety import DOCUMENT_SAFETY_INSTRUCTION, wrap_untrusted_document_text
from app.services.zotero_sqlite import ZoteroSQLiteReader


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
    ) -> tuple[str, list[AssistantCitation]]:
        if verification.status == "clarify":
            return self._clarification_question(contract, session or SessionContext(session_id="clarify")), []
        if verification.status == "retry":
            return self._compose_retry_answer(contract=contract, verification=verification, claims=claims), []
        citations = self._citations_from_doc_ids(self._claim_evidence_ids(claims), evidence)
        if not citations and papers:
            fallback_doc_ids = [doc_id for paper in papers[:2] for doc_id in paper.doc_ids[:1]]
            citations = self._citations_from_doc_ids(fallback_doc_ids, evidence)
        if contract.allow_web_search:
            citations = self._prioritize_web_citations(claims=claims, evidence=evidence, citations=citations)
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
        if self._should_try_llm_research_composer(claims, include_legacy=True):
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
        llm_answer = self._compose_research_answer_markdown(
            contract=contract,
            claims=claims,
            evidence=evidence,
            papers=papers,
            citations=citations,
            verification=verification,
            session=session,
            stream_callback=stream_callback,
        )
        return llm_answer, citations

    @staticmethod
    def _claims_from_schema_solver(claims: list[Claim]) -> bool:
        return any(dict(claim.structured_data or {}).get("source") == "schema_claim_solver" for claim in claims)

    @classmethod
    def _should_try_llm_research_composer(cls, claims: list[Claim], *, include_legacy: bool = False) -> bool:
        if cls._claims_from_schema_solver(claims):
            return True
        if not include_legacy:
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
        web_citations = self._citations_from_doc_ids(web_doc_ids, evidence)
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
            selected_title = self._contract_note_value(contract.notes, "memory_title=")
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

    @staticmethod
    def _contract_note_value(notes: list[str], prefix: str) -> str:
        for note in notes:
            raw = str(note or "")
            if raw.startswith(prefix):
                return raw[len(prefix) :].strip()
        return ""

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
                "answer_slots": self._answer_slots(contract),
                "claim_types": list(dict.fromkeys(item.claim_type for item in claims)),
                "notes": contract.notes,
                "verification": verification.model_dump(),
                "claims": [item.model_dump() for item in claims],
                "papers": [
                    {"title": item.title, "year": item.year, "paper_id": item.paper_id}
                    for item in papers[:4]
                ],
                "evidence": [
                    {
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
                        "title": item.title,
                        "authors": item.authors,
                        "year": item.year,
                        "page": item.page,
                        "block_type": item.block_type,
                    }
                    for item in citations[:4]
                ],
                "conversation_context": self._session_conversation_context(session, max_chars=16000) if session is not None else {},
            },
            ensure_ascii=False,
        )
        system_prompt = (
                "你是论文研究助手的最终回答整理器。"
                "你的主要依据是输入中的 query_contract / claims / evidence / citations。"
                "conversation_context 是上一轮真实对话历史，只用于解析指代、继承上一轮种子/比较对象、避免重复回答和说明当前问题相对上一轮任务的关系。"
                "如果当前问题是在确认上一轮候选是否属于后续/扩展/相关工作，不要写成普通论文摘要；要直接说明它相对上一轮 seed 的关系、证据强弱和不确定边界。"
                "不要使用你记忆里的论文知识，不要套固定答案模板，不要补充未给出的事实。"
                "请输出简洁中文 Markdown。"
                "除公式符号、论文标题、模型名、数据集名和不可翻译专有名词外，解释文字必须使用中文；"
                "不要把变量解释写成英文长句。"
                "根据 requested_fields、answer_slots、claim_types 和 required_modalities 决定答案结构，不要按 relation 套模板。"
                "如果 claim_types/requested_fields 包含 formula，优先使用 claim.structured_data.formula_latex 或 claim.value 中已经给出的公式；"
                "数学公式必须是 Markdown + KaTeX 可渲染的标准 LaTeX，独立公式单独成段并使用 $$...$$，行内变量用 $...$。"
                "不要输出 ∇θLDPO、πθ、πref、logσ 这类半 Unicode 半文本粘连符号；应写成 \\nabla_{\\theta}\\mathcal{L}_{\\mathrm{DPO}}、"
                "\\pi_{\\theta}、\\pi_{\\mathrm{ref}}、\\log \\sigma 这类 LaTeX。"
                "不要使用 \\(...\\) 或 \\[...\\]，也不要把 LaTeX 只包在普通括号里。"
                "如果传入的 claims 来自不同 paper_id，请按论文分小节输出；每节标题使用论文标题，"
                "并明确标注该节公式、数值或结论仅来自该论文，禁止把不同论文的定义揉成同一句话。"
                "如果 claim_types/requested_fields 包含 origin/paper_title/year，明确写出论文标题和年份。"
                "如果包含 recommended_papers，按推荐顺序列出论文，并给出每篇为什么值得读的一句话理由。"
                "如果包含 followup_papers/candidate_relationship，先点明种子/原始论文，再列出后续工作，并说明每篇为什么算后续或扩展。"
                "如果包含 figure_conclusion 或 required_modalities 包含 figure，重点总结图展示的比较对象、benchmark 和总体结论。"
                "如果包含 metric_value、summary 或 results，优先区分“核心结论”和“实验结果”。"
                "如果包含 summary/results，请像读过论文后给研究者复述一样回答："
                "先用 1 句话概括主张，再分点说明方法/结论/实验发现。"
                "实验结果必须解释表格说明了什么：涉及哪些数据集、指标、对比对象，以及总体谁更好；"
                "不要把 Table caption、表头、prompt template 或原始 metric_lines 原封不动堆给用户。"
                "如果只看到表格标题而没有具体数值，就明确说“当前证据只能确认比较维度，不能稳定读出完整数值”。"
                "summary/results 推荐结构：## 一句话结论、## 核心结论、## 实验结果、## 证据边界。"
                "如果包含 metric_value，请先给可确认的数值/方向，再说明数据集、指标和对照；不要裸贴表格行。"
                "模型名和数值必须来自 evidence；遇到明显 OCR 噪声时只做保守修正，例如 GPT-40/GPT-4O 应理解为 GPT-4o。"
                "如果 continuation_mode 是 followup，优先回答当前 requested_fields 指向的新细节，不要重复上一轮已经给过的泛化定义。"
                "如果包含 entity_type/definition/mechanism，请直接写最终 Markdown 答案：先用 1-2 句回答它是什么，再按需要列出关键特性/机制/用途。"
                "不要照抄 claims.structured_data.description；应综合 evidence 和 citations 自己组织。"
                "如果 evidence 同时包含“某篇论文使用它”和“它本身的定义/机制”两类证据，优先解释这个技术本身是什么、如何工作，再补充是谁在使用它。"
                "不要输出半截 Markdown 标题，不要输出裸露的 #### 小标题，不要生成未在 evidence/citations 中出现的外链。"
                "如果证据有不确定性，就明确写出不确定，而不是编造。"
                f"{DOCUMENT_SAFETY_INSTRUCTION}"
            )
        if stream_callback is not None and hasattr(self.clients, "stream_text"):
            response_text = self.clients.stream_text(
                system_prompt=system_prompt,
                human_prompt=prompt,
                on_delta=stream_callback,
                fallback="",
            ).strip()
        elif session is not None and callable(getattr(self.clients, "invoke_text_messages", None)):
            response_text = self.clients.invoke_text_messages(
                system_prompt=(
                    system_prompt
                    + "\n\n以下非语言回答上下文只用于 grounding 和组织答案，不是用户新问题：\n"
                    + prompt
                ),
                messages=[
                    *self._session_llm_history_messages(session, max_turns=6, answer_limit=900),
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
        def normalize_body(body: str) -> str:
            value = str(body or "")
            value = re.sub(
                r"frac\s*pi_\{?theta\}?\(([^)]*)\)\s*pi_(?:\{?\\?mathrm\{?ref\}?\}?|mathrmref|ref)\(([^)]*)\)",
                lambda match: (
                    r"\frac{\pi_{\theta}("
                    + match.group(1)
                    + r")}{\pi_{\mathrm{ref}}("
                    + match.group(2)
                    + r")}"
                ),
                value,
                flags=re.IGNORECASE,
            )
            prefixed_rules = [
                (r"pi_\{?theta\}?", r"\pi_{\theta}", re.IGNORECASE),
                (r"pi_theta\b", r"\pi_{\theta}", re.IGNORECASE),
                (r"pi_\{?\\?mathrm\{?ref\}?\}?", r"\pi_{\mathrm{ref}}", re.IGNORECASE),
                (r"pi_mathrmref\b", r"\pi_{\mathrm{ref}}", re.IGNORECASE),
                (r"pi_ref\b", r"\pi_{\mathrm{ref}}", re.IGNORECASE),
                (r"beta\b", r"\beta", 0),
                (r"sigma\b", r"\sigma", 0),
                (r"theta\b", r"\theta", 0),
                (r"log(?=\s*(?:\\pi|\\sigma|[A-Za-z{]))", r"\log", 0),
                (r"frac(?=\s*\{)", r"\frac", 0),
                (r"mathbb(?=\s*\{)", r"\mathbb", 0),
                (r"mathcal(?=\s*\{)", r"\mathcal", 0),
                (r"mathrm(?=\s*\{)", r"\mathrm", 0),
            ]
            for pattern, replacement, flags in prefixed_rules:
                value = re.sub(
                    rf"(^|[^\\A-Za-z]){pattern}",
                    lambda match, repl=replacement: match.group(1) + repl,
                    value,
                    flags=flags,
                )
            return value

        normalized = re.sub(
            r"\$\$([\s\S]+?)\$\$",
            lambda match: "$$" + normalize_body(match.group(1)) + "$$",
            str(text or ""),
        )
        normalized = re.sub(
            r"(?<!\$)\$([^$\n]+?)\$(?!\$)",
            lambda match: "$" + normalize_body(match.group(1)) + "$",
            normalized,
        )
        return normalized

    @staticmethod
    def _answer_slots_from_notes(notes: list[str]) -> list[str]:
        slots: list[str] = []
        for note in notes:
            text = str(note or "")
            if text.startswith("answer_slot=") and "=" in text:
                slot = text.split("=", 1)[1].strip()
                if slot:
                    slots.append(slot)
        return list(dict.fromkeys(slots))

    @staticmethod
    def _answer_slots(contract: QueryContract) -> list[str]:
        slots = [str(item).strip() for item in list(getattr(contract, "answer_slots", []) or []) if str(item).strip()]
        if slots:
            return list(dict.fromkeys(slots))
        return AnswerComposerMixin._answer_slots_from_notes(contract.notes)

    @staticmethod
    def _composer_goals(contract: QueryContract) -> set[str]:
        goals: set[str] = set()
        for value in [
            *list(contract.requested_fields or []),
            *AnswerComposerMixin._answer_slots(contract),
        ]:
            key = "_".join(str(value or "").strip().lower().replace("-", "_").split())
            aliases = {
                "general_answer": {"answer"},
                "paper_recommendation": {"recommended_papers"},
                "followup_research": {"followup_papers", "candidate_relationship"},
                "paper_summary": {"summary", "results"},
                "metric_value": {"metric_value", "setting"},
                "figure": {"figure_conclusion"},
                "origin": {"origin", "paper_title", "year"},
                "entity_definition": {"entity_type", "definition", "mechanism"},
                "concept_definition": {"definition", "mechanism", "examples"},
            }
            goals.update(aliases.get(key, {key} if key else set()))
        return goals or {"answer"}

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
        lines = self._metric_lines_from_claims(claims)
        if not lines:
            return ""
        target = " / ".join(contract.targets) if contract.targets else (claims[0].entity or "目标对象")
        body = "\n".join(f"- {line}" for line in lines[:5])
        return (
            "## 结论\n\n"
            f"{target} 的表现需要按表格证据来读；我保留原始指标行，避免把 win rate、accuracy/ACC、"
            "p-soups、Llama/baseline 这类对照信息压缩丢失。\n\n"
            "## 实验结果\n\n"
            f"{body}"
        )

    def _compose_paper_summary_results_answer(self, *, contract: QueryContract, claims: list[Claim]) -> str:
        if not claims:
            return ""
        claim = claims[0]
        lines = self._metric_lines_from_claims(claims)
        target = contract.targets[0] if contract.targets else (claim.entity or "该论文")
        summary = " ".join(str(claim.value or "").split())
        if not summary:
            summary = f"{target} 的核心结论需要结合论文摘要与实验表格理解。"
        core_points = self._paper_result_core_points(target=target, support_text="\n".join([summary] + lines))
        core_body = "\n".join(f"- {item}" for item in core_points)
        metric_body = "\n".join(f"- {line}" for line in lines[:6]) if lines else "- 当前证据没有稳定抽出独立的指标行。"
        return (
            "## 核心结论\n\n"
            f"{core_body}\n\n"
            "## 实验结果\n\n"
            f"{metric_body}"
        )

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

    @classmethod
    def _compose_followup_research_answer(cls, *, claims: list[Claim]) -> str:
        if not claims:
            return ""
        claim = claims[0]
        structured = dict(claim.structured_data or {})
        seeds = list(structured.get("seed_papers", []) or [])
        rows = list(structured.get("followup_titles", []) or [])
        if not rows:
            return ""
        entity = str(claim.entity or "").strip()
        selected_candidate_title = str(structured.get("selected_candidate_title", "") or "").strip()
        seed_text = ""
        if seeds:
            seed = dict(seeds[0] or {})
            seed_title = str(seed.get("title", "")).strip()
            seed_year = str(seed.get("year", "")).strip()
            if seed_title:
                prefix = f"围绕 {entity} 追踪后续工作，" if entity else ""
                seed_text = prefix + f"种子论文是《{seed_title}》" + (f"（{seed_year}）" if seed_year else "") + "。"
        direct_body: list[str] = []
        strong_body: list[str] = []
        weak_body: list[str] = []
        for row in rows[:10]:
            item = dict(row or {})
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            year = str(item.get("year", "")).strip()
            relation_type = str(item.get("relation_type", "")).strip() or "后续/扩展"
            strength = str(item.get("relationship_strength", "")).strip().lower()
            reason = cls._followup_public_reason(item)
            line = f"- 《{title}》" + (f"（{year}）" if year else "") + f"：{relation_type}"
            if reason:
                line += f"，{reason}"
            if strength == "direct":
                direct_body.append(line)
            elif strength == "strong_related":
                strong_body.append(line)
            else:
                weak_body.append(line)
        if not direct_body and not strong_body and not weak_body:
            return ""
        if selected_candidate_title:
            selected_key = " ".join(selected_candidate_title.lower().split())
            selected_rows = [
                dict(row or {})
                for row in rows
                if selected_key
                and (
                    selected_key in " ".join(str(row.get("title", "")).lower().split())
                    or " ".join(str(row.get("title", "")).lower().split()) in selected_key
                )
            ]
            selected = selected_rows[0] if selected_rows else dict(rows[0] or {})
            strength = str(selected.get("relationship_strength", "")).strip().lower()
            relation_type = str(selected.get("relation_type", "") or "相关延续候选")
            classification = str(selected.get("classification", "") or "").strip()
            evidence_ids = [str(item) for item in list(selected.get("evidence_ids", []) or []) if str(item)]
            title = str(selected.get("title", selected_candidate_title) or selected_candidate_title)
            year = str(selected.get("year", "") or "")
            reason = cls._followup_public_reason(selected)
            strict_followup = bool(selected.get("strict_followup", False))
            if strict_followup:
                verdict = "可以写成严格后续工作。"
            elif strength == "direct":
                verdict = "有直接使用/评测/扩展类证据，但当前验证器没有把它判成严格后续工作。"
            elif strength == "strong_related":
                verdict = "更适合写成强相关延续候选，暂时不要写成严格后续工作。"
            elif strength == "unrelated":
                verdict = "当前证据不支持把它写成后续工作。"
            else:
                verdict = "当前证据不足，不能写成严格后续工作。"
            parts = ["## 判断", "", f"《{title}》" + (f"（{year}）" if year else "") + f"相对 {entity or '种子论文'}：{verdict}"]
            if seed_text:
                parts.extend(["", "## 种子论文", "", seed_text])
            parts.extend(["", "## 关系证据", "", f"- 关系类型：{relation_type}"])
            parts.append(f"- 严格后续：{'是' if strict_followup else '否/证据不足'}")
            if classification:
                parts.append(f"- 验证分类：{classification}")
            if evidence_ids:
                parts.append(f"- 证据范围：已对 seed/candidate 两侧论文证据块做关系验证（{len(evidence_ids)} 个 evidence block）。")
            if reason:
                parts.append(f"- 依据：{reason}")
            parts.extend([
                "",
                "## 写法建议",
                "",
                "- 如果正文需要严谨表述，优先写成 related work / strong continuation candidate。",
                "- 只有看到明确使用、继承、引用或评测种子论文/数据集的证据后，再写成严格后续工作。",
            ])
            return "\n".join(parts).strip()
        parts: list[str] = ["## 检索结论", ""]
        if seed_text:
            parts.extend([seed_text, ""])
        if direct_body:
            parts.extend(["我能确认到一些直接后续/使用证据；其余论文需要按相关候选阅读。", "", "## 直接后续/使用证据", "", *direct_body[:5]])
        else:
            parts.extend([
                "本地库当前没有足够证据确认严格意义上的后续工作，也就是没有稳定看到“明确使用、继承、引用或评测该种子论文/数据集”的关系证据。",
                "",
            ])
        if strong_body:
            parts.extend(["", "## 强相关延续候选", "", *strong_body[:6]])
        if weak_body:
            parts.extend(["", "## 同主题但待确认", "", *weak_body[:4]])
        parts.extend([
            "",
            "## 读法建议",
            "",
            "- 如果你要找“严格后续工作”，优先看第一组；如果第一组为空，就需要继续做引用链或 Web 验证。",
            "- 第二、三组更适合当作 related work 线索，不宜直接写成“使用了 AlignX/继承了 AlignX”。",
        ])
        return "\n".join(parts).strip()

    @staticmethod
    def _followup_public_reason(item: dict[str, object]) -> str:
        reason = " ".join(str(item.get("reason", "") or "").split())
        strength = str(item.get("relationship_strength", "") or "").strip().lower()
        if not reason:
            return ""
        ascii_letters = sum(1 for char in reason if ("a" <= char.lower() <= "z"))
        chinese_chars = sum(1 for char in reason if "\u4e00" <= char <= "\u9fff")
        if ascii_letters > chinese_chars * 2 and "；" not in reason:
            if strength == "direct":
                return "结构化证据显示它与种子论文/数据集存在直接使用、评测或扩展关系。"
            if strength == "strong_related":
                return "主题和任务设置接近，但当前证据还不足以确认严格继承或使用关系。"
            return "仅能确认属于相邻研究方向，仍需引用链或全文证据复核。"
        return reason

    def _compose_topology_recommendation_answer(
        self,
        *,
        contract: QueryContract,
        claims: list[Claim],
        evidence: list[EvidenceBlock],
    ) -> str:
        if not claims:
            return ""
        claim = claims[0]
        structured = dict(claim.structured_data or {})
        terms = [str(item) for item in structured.get("topology_terms", []) if str(item).strip()]
        if not terms:
            terms = extract_topology_terms(evidence)
        terms_text = "、".join(terms or ["DAG", "irregular/random", "chain", "tree", "mesh"])
        summary = self._clean_topology_public_text(str(claim.value or structured.get("summary", "") or "").strip())
        rationale = self._clean_topology_public_text(str(structured.get("rationale", "") or "").strip())
        return (
            "## 结论\n\n"
            "我会用 **DAG 作为主干拓扑** 来组织 PDF-Agent 的 multi-agent 系统；"
            "局部可以嵌入 chain 或 tree，但不建议把整套系统做成全 mesh。\n\n"
            "## 证据边界\n\n"
            f"- 当前论文证据主要覆盖 {terms_text} 等 multi-agent topology。"
            f"{summary if summary else '这些证据没有证明存在一个脱离任务的绝对最优拓扑。'}\n"
            "- 因此下面是“论文证据 + PDF-RAG 工程约束”的设计建议，不是某篇论文直接给出的定理。\n\n"
            "## 组织建议\n\n"
            "- **DAG 主干**：把意图识别、论文召回、证据扩展、公式/表格/图像解析、claim verification、answer compose 做成有依赖的节点；可并行的检索和解析分支并行跑，最后汇总。\n"
            "- **局部 chain**：单篇 PDF 精读、OCR 清洗、公式抽取这类严格顺序步骤可以保持 chain，便于调试和重试。\n"
            "- **局部 tree**：多篇论文比较、多个候选含义消歧、多个公式定义聚合时，用 tree 做分解-汇总会更自然。\n"
            "- **少用 mesh / irregular**：它们适合探索式协作，但对 PDF-Agent 来说更容易带来状态漂移、重复检索和不可控成本。\n"
            f"- 选择时优先看任务约束：{rationale if rationale else '证据质量、节点成本、是否需要并行验证，以及最终回答是否要可追溯。'}"
        )

    @staticmethod
    def _clean_topology_public_text(text: str) -> str:
        compact = " ".join(str(text or "").split())
        if not compact:
            return ""
        lowered = compact.lower()
        blocked_markers = [
            "does not address",
            "does not contain",
            "impossible to determine",
            "no direct analysis",
            "not provide specific",
            "cannot determine",
            "无法确定",
            "不能确定",
            "没有覆盖",
            "不包含",
        ]
        if any(marker in lowered for marker in blocked_markers):
            return ""
        ascii_letters = sum(1 for char in compact if ("a" <= char.lower() <= "z"))
        chinese_chars = sum(1 for char in compact if "\u4e00" <= char <= "\u9fff")
        if ascii_letters > max(80, chinese_chars * 3):
            return ""
        if compact and compact[-1] not in "。.!?！？":
            compact += "。"
        return compact

    @staticmethod
    def _compose_formula_answer(*, claims: list[Claim], contract: QueryContract | None = None) -> str:
        if not claims:
            return ""
        notice = AnswerComposerMixin._auto_resolved_candidate_notice(contract)
        formula_claims = [claim for claim in claims if claim.claim_type == "formula"] or [claims[0]]
        if len(formula_claims) > 1:
            sections = ["## 核心公式"]
            all_term_lines: list[str] = []
            for index, claim in enumerate(formula_claims, start=1):
                formula_text = str(claim.value or "").strip()
                if not formula_text:
                    continue
                structured = dict(claim.structured_data or {})
                paper_title = str(structured.get("paper_title", "") or "").strip()
                heading = f"### {index}. 《{paper_title}》" if paper_title else f"### {index}. {claim.entity or '公式'}"
                formula_format = str(structured.get("formula_format", "")).lower()
                if formula_format == "latex":
                    sections.extend(["", heading, "", "$$\n" + formula_text + "\n$$"])
                else:
                    sections.extend(["", heading, "", "```text\n" + formula_text + "\n```"])
                all_term_lines.extend(AnswerComposerMixin._formula_term_lines(claim))
            term_lines = list(dict.fromkeys(all_term_lines))
            if term_lines:
                sections.extend(["", "## 变量", "", *term_lines])
            answer = "\n".join(sections).strip()
            return f"{notice}\n\n{answer}".strip() if notice else answer
        claim = formula_claims[0]
        formula_text = str(claim.value or "").strip()
        if not formula_text:
            return ""
        term_lines = AnswerComposerMixin._formula_term_lines(claim)
        formula_format = str(dict(claim.structured_data or {}).get("formula_format", "")).lower()
        if formula_format == "latex":
            answer = "## 核心公式\n\n$$\n" + formula_text + "\n$$"
        else:
            answer = "## 核心公式\n\n```text\n" + formula_text + "\n```"
        if term_lines:
            answer += "\n\n## 变量\n\n" + "\n".join(term_lines)
        if notice:
            answer = f"{notice}\n\n{answer}"
        return answer

    @staticmethod
    def _auto_resolved_candidate_notice(contract: QueryContract | None) -> str:
        if contract is None or "auto_resolved_by_llm_judge" not in set(contract.notes or []):
            return ""
        selected: dict[str, object] = {}
        for note in contract.notes:
            raw = str(note or "")
            if not raw.startswith("selected_ambiguity_option="):
                continue
            try:
                payload = json.loads(raw.split("=", 1)[1])
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                selected = payload
                break
        title = str(selected.get("display_title") or selected.get("title") or "").strip()
        label = str(selected.get("display_label") or selected.get("label") or selected.get("meaning") or "").strip()
        if title:
            return f"我按最匹配的候选《{title}》来回答。"
        if label:
            return f"我按最匹配的候选“{label}”来回答。"
        return "我按当前候选中最匹配的一项来回答。"

    @staticmethod
    def _formula_term_lines(claim: Claim) -> list[str]:
        structured = dict(claim.structured_data or {})
        variable_lines = AnswerComposerMixin._formula_variable_lines(structured.get("variables"))
        if variable_lines:
            return variable_lines
        terms = {str(item).lower() for item in list(structured.get("terms", []) or [])}
        term_lines: list[str] = []
        if "pi_theta" in terms:
            term_lines.append("- $\\pi_\\theta$：当前策略（policy）。")
        if "pi_phi" in terms:
            term_lines.append("- $\\pi_\\phi$：PBA 中条件化在显式偏好方向上的生成策略。")
        if "pi_ref" in terms:
            term_lines.append("- $\\pi_{ref}$：参考策略（reference policy）。")
        if "p_tilde" in terms:
            term_lines.append("- $\\tilde{P}$：由 persona 聚合得到的显式 preference direction vector。")
        if "beta" in terms:
            term_lines.append("- $\\beta$：控制偏好约束强度的系数。")
        if "log_sigma" in terms:
            term_lines.append("- $\\log \\sigma$：sigmoid 偏好概率项的对数形式。")
        if "preferred" in terms:
            term_lines.append("- $y_w$：preferred response，偏好样本。")
        if "rejected" in terms:
            term_lines.append("- $y_l$：rejected response，劣选样本。")
        if "ratio" in terms:
            term_lines.append("- $r_t(\\theta)$：新旧策略在同一动作上的概率比。")
        if "advantage" in terms:
            term_lines.append("- $\\hat{A}_t$：优势估计，表示该动作相对当前价值基线的好坏。")
        if "epsilon" in terms:
            term_lines.append("- $\\epsilon$：clip 范围，用来限制单步策略更新幅度。")
        if "clip" in terms:
            term_lines.append("- $\\operatorname{clip}$：把概率比裁剪到 $[1-\\epsilon, 1+\\epsilon]$。")
        return term_lines

    @staticmethod
    def _formula_variable_lines(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        lines: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or item.get("name") or "").strip()
            description = str(
                item.get("description")
                or item.get("meaning")
                or item.get("role")
                or item.get("definition")
                or ""
            ).strip()
            if not symbol or not description:
                continue
            symbol_markdown = AnswerComposerMixin._format_formula_symbol(symbol)
            description_markdown = AnswerComposerMixin._format_formula_description(description)
            line = f"- {symbol_markdown}：{description_markdown}"
            key = " ".join(line.lower().split())
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
        return lines

    @staticmethod
    def _format_formula_symbol(symbol: str) -> str:
        compact = " ".join(str(symbol or "").strip().split())
        if not compact:
            return "`?`"
        if any(token in compact for token in ["\\", "_", "^", "{", "}", "(", ")", "π", "σ", "β", "θ", "∇", "ϕ", "φ", "ϵ", "ε"]):
            return f"${compact}$"
        if re.fullmatch(r"[A-Za-z](?:_[A-Za-z0-9]+)?", compact):
            return f"${compact}$"
        return f"`{compact}`"

    @staticmethod
    def _format_formula_description(description: str) -> str:
        text = str(description or "").strip()
        if not text:
            return ""
        text = AnswerComposerMixin._normalize_markdown_math_artifacts(text)

        def wrap_plain_segment(segment: str) -> str:
            patterns = [
                r"\\log\s+\\sigma",
                r"\\pi_\{\\mathrm\{ref\}\}",
                r"\\pi_\{\\theta\}",
                r"\\pi_\{[^{}]+\}",
                r"\\pi_[A-Za-z0-9]+",
                r"\\(?:beta|sigma|theta|phi|varphi|epsilon|nabla)",
                r"y_[wl]",
                r"r\(x,\s*y\)",
                r"\bD\b",
            ]
            combined = re.compile("|".join(f"(?:{pattern})" for pattern in patterns))
            return combined.sub(lambda match: f"${match.group(0)}$", segment)

        parts = re.split(r"(\$[^$\n]+\$)", text)
        return "".join(part if part.startswith("$") and part.endswith("$") else wrap_plain_segment(part) for part in parts)

    @staticmethod
    def _metric_lines_from_claims(claims: list[Claim]) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for claim in claims:
            for line in list(dict(claim.structured_data or {}).get("metric_lines", []) or []):
                compact = " ".join(str(line).split())
                if not compact:
                    continue
                normalized = compact.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                lines.append(compact)
        return lines

    @staticmethod
    def _paper_result_core_points(*, target: str, support_text: str) -> list[str]:
        lower = support_text.lower()
        points: list[str] = []
        if "preference inference" in lower or "偏好推断" in lower:
            points.append(f"{target} 的核心问题包含 preference inference（偏好推断）。")
        if "conditioned generation" in lower or "generator" in lower or "条件生成" in lower:
            points.append("方法把 conditioned generation / generator 作为生成侧能力来建模。")
        if "modular" in lower or "模块化" in lower:
            points.append("整体结论强调 modular（模块化）拆分，而不是把偏好建模与生成能力混成单一黑盒。")
        if not points:
            compact = " ".join(support_text.split())
            points.append(compact[:260] if compact else f"{target} 的核心结论见下方证据。")
        return points[:4]

    def _compose_conversation_response(
        self,
        *,
        contract: QueryContract,
        query: str,
        session: SessionContext,
    ) -> str:
        if contract.relation == "self_identity":
            return "我是你的论文研究助手，可以围绕 Zotero 论文库做检索、总结、对比、术语解释、公式/图表/表格解读和后续工作追踪。"
        if contract.relation == "capability":
            return (
                "我可以帮你做这些事：\n\n"
                "- 论文总结：提炼核心贡献、方法和实验结果。\n"
                "- 对比分析：比较多篇论文的方法、指标、优缺点和适用场景。\n"
                "- 概念/术语/定义：解释论文里的技术名词、模型、数据集和 benchmark。\n"
                "- 公式、表格、图表：定位并解释公式变量、表格指标、figure 结论。\n"
                "- 后续工作追踪：从一篇论文、数据集或方法出发，找相关扩展研究。"
            )
        if contract.relation == "library_status":
            return self._compose_library_status_response(query=query)
        if contract.relation == "library_recommendation":
            return self._compose_library_recommendation_response(query=query, session=session)
        if self.clients.chat is None:
            raise RuntimeError("Conversation response generation requires an available chat model.")
        response_text = self.clients.invoke_text(
            system_prompt=(
                "你是论文研究助手的对话回复整理器。"
                "请基于当前 relation、用户问题和最近会话上下文，用简洁中文 Markdown 回复。"
                "不要编造论文事实，不要输出研究结论，只处理元对话。"
                "如果 relation 是 capability，就用 3-5 条要点说明能力边界。"
                "如果 relation 是 library_status，必须使用 assistant_self_knowledge 的规则，不要把检索候选数量当成总论文数。"
                "如果 relation 是 library_recommendation，从当前本地库状态出发推荐，不要请求用户反复澄清。"
                "如果 relation 是 clarify_user_intent 或 correction_without_context，就帮助用户把问题说清楚。"
            ),
            human_prompt=json.dumps(
                {
                    "query": query,
                    "relation": contract.relation,
                    "assistant_self_knowledge": _load_assistant_self_knowledge(),
                    "conversation_context": self._session_conversation_context(session),
                },
                ensure_ascii=False,
            ),
            fallback="",
        ).strip()
        if not response_text:
            raise RuntimeError("Conversation response generation failed: upstream LLM response was empty.")
        return response_text

    def _compose_library_status_response(self, *, query: str = "") -> str:
        docs = []
        seen_paper_ids: set[str] = set()
        for doc in self.retriever.paper_documents():
            meta = dict(doc.metadata or {})
            paper_id = str(meta.get("paper_id", "")).strip()
            if not paper_id or paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper_id)
            docs.append(meta)

        collection_paths: dict[str, list[str]] = {}
        try:
            collection_paths = ZoteroSQLiteReader(self.settings).read_attachment_collection_paths()
        except Exception:  # noqa: BLE001
            collection_paths = {}

        category_counter: Counter[str] = Counter()
        year_values: list[int] = []
        pdf_paths = 0
        for meta in docs:
            paper_id = str(meta.get("paper_id", "")).strip()
            tags = [tag for tag in str(meta.get("tags", "")).split("||") if tag]
            categories = collection_paths.get(paper_id) or tags[:3] or ["未分类"]
            for category in dict.fromkeys(str(item or "未分类") for item in categories):
                category_counter[category] += 1
            year = str(meta.get("year", "")).strip()
            if year.isdigit():
                year_values.append(int(year))
            if str(meta.get("file_path", "")).lower().endswith(".pdf"):
                pdf_paths += 1

        total = len(docs)
        lines = [
            "## 当前论文库",
            "",
            f"我当前索引的本地 Zotero/PDF 论文库共有 **{total} 篇论文**。",
        ]
        if pdf_paths:
            lines.append(f"其中有 PDF 路径的记录是 **{pdf_paths} 篇**。")
        if year_values:
            lines.append(f"年份范围大约是 **{min(year_values)}–{max(year_values)}**。")
        if category_counter:
            top_categories = "、".join(f"{name}（{count}）" for name, count in category_counter.most_common(6))
            lines.extend(["", "## 分类概览", "", f"主要分类：{top_categories}。"])
        if self._library_status_query_wants_listing(query):
            preview_limit = 18 if total <= 24 else 12
            preview_lines = self._library_paper_preview_lines(
                docs=docs,
                collection_paths=collection_paths,
                limit=preview_limit,
            )
            if preview_lines:
                lines.extend(
                    [
                        "",
                        "## 文章预览",
                        "",
                        f"你问“有哪些文章”时，我先按年份较新、元数据较完整列出 **{len(preview_lines)} 篇预览**；完整列表可以在左侧 Zotero 分类栏继续浏览。",
                        "",
                        *preview_lines,
                    ]
                )
        if self._library_status_query_wants_recommendation(query):
            recommendations = self._rank_library_papers_for_recommendation(docs=docs, query=query, limit=3)
            if recommendations:
                primary = recommendations[0]
                primary_year = f"（{primary['year']}）" if primary.get("year") else ""
                lines.extend(
                    [
                        "",
                        "## 默认推荐",
                        "",
                        "如果你没有指定“值得一读”的标准，我默认按基础性、覆盖面和本地摘要可支撑性来选。",
                        f"我会先读《{primary['title']}》{primary_year}：{primary['reason']}",
                    ]
                )
                if len(recommendations) > 1:
                    runners_up_items = []
                    for item in recommendations[1:]:
                        year_suffix = f"（{item['year']}）" if item.get("year") else ""
                        runners_up_items.append(f"《{item['title']}》{year_suffix}")
                    runners_up = "；".join(runners_up_items)
                    lines.append(f"备选：{runners_up}。")
        lines.extend(
            [
                "",
                "这个数字来自当前索引的 paper store，不是一次检索召回的 top-k 候选数；候选论文数量不能当成总论文数。",
            ]
        )
        return "\n".join(lines)

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
        return {
            "dialect": "SQLite",
            "tables": {
                "papers": {
                    "description": "One row per indexed local paper.",
                    "columns": {
                        "paper_id": "Stable local paper id.",
                        "title": "Paper title.",
                        "authors": "Comma-separated author names.",
                        "year": "Original year string from metadata.",
                        "year_int": "Integer publication year when parseable, otherwise NULL.",
                        "tags": "Tags joined by ||.",
                        "categories": "Zotero collection/category paths joined by ||.",
                        "aliases": "Known aliases joined by ||.",
                        "abstract": "Abstract text when available.",
                        "summary": "Generated summary when available.",
                        "has_pdf": "1 if the indexed file path is a PDF, else 0.",
                        "file_path": "Local PDF path when available.",
                        "searchable_text": "Concatenated title/authors/year/tags/categories/aliases/abstract/summary for broad LIKE matching.",
                    },
                },
                "paper_authors": {
                    "description": "One row per parsed author name.",
                    "columns": {
                        "paper_id": "Stable local paper id.",
                        "title": "Paper title copied from papers.",
                        "author": "Single author name.",
                        "year_int": "Integer publication year when parseable.",
                    },
                },
                "paper_tags": {
                    "description": "One row per tag.",
                    "columns": {
                        "paper_id": "Stable local paper id.",
                        "title": "Paper title copied from papers.",
                        "tag": "Single tag.",
                        "year_int": "Integer publication year when parseable.",
                    },
                },
                "paper_categories": {
                    "description": "One row per Zotero collection/category path.",
                    "columns": {
                        "paper_id": "Stable local paper id.",
                        "title": "Paper title copied from papers.",
                        "category": "Single collection/category path.",
                        "year_int": "Integer publication year when parseable.",
                    },
                },
            },
        }

    def _library_metadata_rows(self) -> list[dict[str, Any]]:
        collection_paths: dict[str, list[str]] = {}
        try:
            collection_paths = ZoteroSQLiteReader(self.settings).read_attachment_collection_paths()
        except Exception:  # noqa: BLE001
            collection_paths = {}

        rows: list[dict[str, Any]] = []
        seen_paper_ids: set[str] = set()
        for doc in self.retriever.paper_documents():
            meta = dict(doc.metadata or {})
            paper_id = str(meta.get("paper_id", "")).strip()
            if not paper_id or paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper_id)
            title = " ".join(str(meta.get("title", "") or "").split())
            authors = " ".join(str(meta.get("authors", "") or "").split())
            year = str(meta.get("year", "") or "").strip()
            year_int = int(year) if year.isdigit() else None
            tags = [tag.strip() for tag in str(meta.get("tags", "") or "").split("||") if tag.strip()]
            categories = [str(item or "未分类").strip() for item in (collection_paths.get(paper_id) or tags[:3] or ["未分类"])]
            aliases = [alias.strip() for alias in str(meta.get("aliases", "") or "").split("||") if alias.strip()]
            abstract = " ".join(str(meta.get("abstract_note", "") or "").split())
            summary = " ".join(str(meta.get("generated_summary", "") or "").split())
            file_path = str(meta.get("file_path", "") or "").strip()
            searchable_text = " ".join(
                item
                for item in [
                    title,
                    authors,
                    year,
                    " ".join(tags),
                    " ".join(categories),
                    " ".join(aliases),
                    abstract,
                    summary,
                    " ".join(str(doc.page_content or "").split())[:1200],
                ]
                if item
            )
            rows.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "year_int": year_int,
                    "tags": "||".join(tags),
                    "categories": "||".join(categories),
                    "aliases": "||".join(aliases),
                    "abstract": abstract[:3000],
                    "summary": summary[:3000],
                    "has_pdf": 1 if file_path.lower().endswith(".pdf") else 0,
                    "file_path": file_path,
                    "searchable_text": searchable_text[:6000],
                    "_author_list": self._split_library_authors(authors),
                    "_tag_list": tags,
                    "_category_list": categories,
                }
            )
        return rows

    @staticmethod
    def _split_library_authors(authors: str) -> list[str]:
        normalized = str(authors or "").replace(" and ", ",")
        names = [" ".join(item.split()) for item in normalized.split(",")]
        deduped: list[str] = []
        seen: set[str] = set()
        for name in names:
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                deduped.append(name)
        return deduped

    @staticmethod
    def _validate_library_metadata_sql(sql: str) -> str:
        normalized = " ".join(str(sql or "").strip().split())
        if normalized.endswith(";"):
            normalized = normalized[:-1].strip()
        if not normalized:
            raise ValueError("empty_sql")
        lowered = normalized.lower()
        if ";" in normalized:
            raise ValueError("multiple_sql_statements_are_not_allowed")
        if any(token in lowered for token in ["--", "/*", "*/"]):
            raise ValueError("sql_comments_are_not_allowed")
        if not (lowered.startswith("select ") or lowered.startswith("with ")):
            raise ValueError("only_select_sql_is_allowed")
        forbidden = {
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "create",
            "replace",
            "attach",
            "detach",
            "pragma",
            "vacuum",
            "reindex",
            "analyze",
            "load_extension",
        }
        tokens = set(re.findall(r"[a-z_]+", lowered))
        blocked = sorted(tokens & forbidden)
        if blocked:
            raise ValueError(f"forbidden_sql_keyword={blocked[0]}")
        if re.search(r"\b(sqlite_master|sqlite_schema|sqlite_temp_master)\b", lowered):
            raise ValueError("sqlite_internal_tables_are_not_allowed")
        allowed_tables = {"papers", "paper_authors", "paper_tags", "paper_categories"}
        table_refs = re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
        disallowed_tables = [table for table in table_refs if table not in allowed_tables]
        if disallowed_tables:
            raise ValueError(f"unknown_table={disallowed_tables[0]}")
        return normalized

    def _execute_library_metadata_sql(
        self,
        *,
        sql: str,
        paper_rows: list[dict[str, Any]],
        max_rows: int,
    ) -> dict[str, Any]:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(
                """
                CREATE TABLE papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    year TEXT,
                    year_int INTEGER,
                    tags TEXT,
                    categories TEXT,
                    aliases TEXT,
                    abstract TEXT,
                    summary TEXT,
                    has_pdf INTEGER,
                    file_path TEXT,
                    searchable_text TEXT
                );
                CREATE TABLE paper_authors (
                    paper_id TEXT,
                    title TEXT,
                    author TEXT,
                    year_int INTEGER
                );
                CREATE TABLE paper_tags (
                    paper_id TEXT,
                    title TEXT,
                    tag TEXT,
                    year_int INTEGER
                );
                CREATE TABLE paper_categories (
                    paper_id TEXT,
                    title TEXT,
                    category TEXT,
                    year_int INTEGER
                );
                CREATE INDEX idx_papers_year ON papers(year_int);
                CREATE INDEX idx_papers_title ON papers(title);
                CREATE INDEX idx_paper_authors_author ON paper_authors(author);
                CREATE INDEX idx_paper_tags_tag ON paper_tags(tag);
                CREATE INDEX idx_paper_categories_category ON paper_categories(category);
                """
            )
            paper_insert_rows = [
                (
                    row["paper_id"],
                    row["title"],
                    row["authors"],
                    row["year"],
                    row["year_int"],
                    row["tags"],
                    row["categories"],
                    row["aliases"],
                    row["abstract"],
                    row["summary"],
                    row["has_pdf"],
                    row["file_path"],
                    row["searchable_text"],
                )
                for row in paper_rows
            ]
            conn.executemany(
                """
                INSERT INTO papers (
                    paper_id, title, authors, year, year_int, tags, categories, aliases,
                    abstract, summary, has_pdf, file_path, searchable_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                paper_insert_rows,
            )
            author_rows = [
                (row["paper_id"], row["title"], author, row["year_int"])
                for row in paper_rows
                for author in list(row.get("_author_list", []) or [])
            ]
            tag_rows = [
                (row["paper_id"], row["title"], tag, row["year_int"])
                for row in paper_rows
                for tag in list(row.get("_tag_list", []) or [])
            ]
            category_rows = [
                (row["paper_id"], row["title"], category, row["year_int"])
                for row in paper_rows
                for category in list(row.get("_category_list", []) or [])
            ]
            conn.executemany("INSERT INTO paper_authors (paper_id, title, author, year_int) VALUES (?, ?, ?, ?)", author_rows)
            conn.executemany("INSERT INTO paper_tags (paper_id, title, tag, year_int) VALUES (?, ?, ?, ?)", tag_rows)
            conn.executemany("INSERT INTO paper_categories (paper_id, title, category, year_int) VALUES (?, ?, ?, ?)", category_rows)
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.execute(sql)
            columns = [str(item[0]) for item in (cursor.description or [])]
            fetched = cursor.fetchmany(max_rows + 1)
            truncated = len(fetched) > max_rows
            result_rows = [self._sqlite_row_to_payload(row) for row in fetched[:max_rows]]
            return {
                "sql": sql,
                "columns": columns,
                "rows": result_rows,
                "row_count": len(result_rows),
                "truncated": truncated,
                "error": "",
            }
        finally:
            conn.close()

    @staticmethod
    def _sqlite_row_to_payload(row: sqlite3.Row) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in row.keys():
            value = row[key]
            if value is None or isinstance(value, (str, int, float)):
                payload[str(key)] = value
            else:
                payload[str(key)] = str(value)
        return payload

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
        rows = list(result.get("rows", []) or [])
        columns = [str(item) for item in list(result.get("columns", []) or [])]
        if not rows:
            return "当前本地 paper index 的元信息查询没有返回匹配记录。"
        if len(rows) == 1 and "title" not in {column.lower() for column in columns}:
            values = "，".join(f"{key}={value}" for key, value in rows[0].items())
            return f"我查的是当前本地 paper index 元信息，结果是：{values}。"
        lines = [
            "我查的是当前本地 paper index 元信息，SQL 查询返回这些记录：",
            "",
        ]
        for row in rows[:12]:
            title = str(row.get("title", "") or row.get("paper_title", "") or "").strip()
            year = str(row.get("year", "") or row.get("year_int", "") or "").strip()
            authors = str(row.get("authors", "") or row.get("author", "") or "").strip()
            paper_id = str(row.get("paper_id", "") or "").strip()
            parts = [f"《{title}》" if title else paper_id or "未命名论文"]
            if year:
                parts.append(f"年份：{year}")
            if authors:
                parts.append(f"作者：{authors}")
            if paper_id and title:
                parts.append(f"paper_id：{paper_id}")
            lines.append("- " + "；".join(parts))
        if bool(result.get("truncated")):
            lines.append("")
            lines.append("结果较多，以上只展示前几条。")
        return "\n".join(lines)

    def _compose_library_recommendation_response(self, *, query: str = "", session: SessionContext | None = None) -> str:
        docs = []
        seen_paper_ids: set[str] = set()
        for doc in self.retriever.paper_documents():
            meta = dict(doc.metadata or {})
            paper_id = str(meta.get("paper_id", "")).strip()
            if not paper_id or paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper_id)
            docs.append(meta)
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
        recent_titles = self._recent_library_recommendation_titles(session)
        llm_selected, llm_note = self._llm_select_library_recommendations(
            query=query,
            candidates=candidates,
            session=session,
            recent_titles=recent_titles,
            limit=limit,
        )
        if llm_selected:
            return llm_selected[:limit], llm_note
        diversified = self._diversify_library_recommendations(
            candidates=candidates,
            recent_titles=recent_titles,
            query=query,
            limit=limit,
        )
        note = "这次我会避开刚刚已经推荐过的论文，换几个不同入口。" if recent_titles else "我会按主题覆盖、论文类型和摘要证据强度挑几个不同入口。"
        return diversified, note

    def _llm_select_library_recommendations(
        self,
        *,
        query: str,
        candidates: list[dict[str, str]],
        session: SessionContext | None,
        recent_titles: list[str],
        limit: int,
    ) -> tuple[list[dict[str, str]], str]:
        if self.clients.chat is None or not candidates:
            return [], ""
        try:
            context = self._session_conversation_context(session) if session is not None else {}
        except Exception:  # noqa: BLE001
            context = {}
        payload = self.clients.invoke_json(
            system_prompt=(
                "你是库内论文推荐重排器。"
                "你的任务不是直接回答用户，而是从 candidate_papers 中选择 3-5 篇最适合当前问题的论文。"
                "必须结合 current_query、conversation_context 和 recently_recommended_titles。"
                "不要固定偏好某一篇；如果最近已经推荐过某篇，除非用户明确追问它，否则优先换不同方向。"
                "推荐理由只能基于候选的 title/year/tags/summary，不要编造引用数或未提供事实。"
                "criteria_note 和 reason 必须使用中文，criteria_note 不超过 60 个汉字，直接说明本轮选择标准。"
                "只输出 JSON：criteria_note, recommendations，其中 recommendations 每项包含 title, reason。"
            ),
            human_prompt=json.dumps(
                {
                    "current_query": query,
                    "recently_recommended_titles": recent_titles,
                    "candidate_papers": candidates[:18],
                    "conversation_context": context,
                },
                ensure_ascii=False,
            ),
            fallback={},
        )
        if not isinstance(payload, dict):
            return [], ""
        by_title = {self._normalize_title_key(item.get("title", "")): item for item in candidates}
        selected: list[dict[str, str]] = []
        raw_recommendations = payload.get("recommendations", [])
        if isinstance(raw_recommendations, list):
            for raw in raw_recommendations:
                if not isinstance(raw, dict):
                    continue
                title = str(raw.get("title", "") or "").strip()
                candidate = by_title.get(self._normalize_title_key(title))
                if candidate is None:
                    continue
                reason = str(raw.get("reason", "") or "").strip() or candidate.get("reason", "")
                selected.append({**candidate, "reason": reason})
                if len(selected) >= limit:
                    break
        note = str(payload.get("criteria_note", "") or "").strip()
        return selected, note

    @staticmethod
    def _clean_library_recommendation_criteria_note(note: str, *, has_recent_recommendations: bool) -> str:
        fallback = (
            "这次我会避开刚刚已经推荐过的论文，换几个不同入口。"
            if has_recent_recommendations
            else "我会按当前问题、主题覆盖、论文类型和摘要证据强度综合挑选。"
        )
        compact = " ".join(str(note or "").split())
        if not compact:
            return fallback
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", compact))
        ascii_letters = len(re.findall(r"[A-Za-z]", compact))
        if ascii_letters > max(12, chinese_chars * 2):
            return fallback
        return compact[:120]

    def _diversify_library_recommendations(
        self,
        *,
        candidates: list[dict[str, str]],
        recent_titles: list[str],
        query: str,
        limit: int,
    ) -> list[dict[str, str]]:
        recent_keys = {self._normalize_title_key(title) for title in recent_titles}
        wants_same_best = any(marker in str(query).lower() for marker in ["最值得", "best", "top", "first"])
        fresh: list[dict[str, str]] = []
        repeated: list[dict[str, str]] = []
        for item in candidates:
            key = self._normalize_title_key(item.get("title", ""))
            if key in recent_keys and not wants_same_best:
                repeated.append(item)
            else:
                fresh.append(item)
        return [*fresh, *repeated][:limit]

    def _recent_library_recommendation_titles(self, session: SessionContext | None) -> list[str]:
        if session is None:
            return []
        titles: list[str] = []
        memory = dict(session.working_memory or {})
        for item in reversed([entry for entry in list(memory.get("tool_results", []) or []) if isinstance(entry, dict)]):
            if item.get("tool") != "get_library_recommendation":
                continue
            titles.extend(re.findall(r"《([^》]{2,220})》", str(item.get("answer_preview", "") or "")))
            if titles:
                break
        if not titles:
            for turn in reversed(session.turns[-4:]):
                if turn.relation not in {"library_recommendation", "compound_query"}:
                    continue
                titles.extend(re.findall(r"《([^》]{2,220})》", turn.answer))
                if titles:
                    break
        deduped: list[str] = []
        seen: set[str] = set()
        for title in titles:
            key = self._normalize_title_key(title)
            if key and key not in seen:
                seen.add(key)
                deduped.append(title)
        return deduped[:8]

    @staticmethod
    def _library_status_query_wants_recommendation(query: str) -> bool:
        compact = " ".join(str(query or "").lower().split())
        markers = [
            "最值得",
            "值得一读",
            "值得读",
            "值得一看",
            "值得看",
            "推荐",
            "哪篇",
            "哪几篇",
            "must read",
            "worth reading",
            "recommend",
        ]
        return any(marker in compact for marker in markers)

    @staticmethod
    def _library_status_query_wants_listing(query: str) -> bool:
        compact = " ".join(str(query or "").lower().split())
        markers = [
            "有哪些",
            "有哪些文章",
            "有哪些论文",
            "文章列表",
            "论文列表",
            "列出",
            "list",
        ]
        return any(marker in compact for marker in markers)

    @staticmethod
    def _library_paper_preview_lines(
        *,
        docs: list[dict[str, object]],
        collection_paths: dict[str, list[str]],
        limit: int,
    ) -> list[str]:
        def year_value(meta: dict[str, object]) -> int:
            year = str(meta.get("year", "") or "").strip()
            return int(year) if year.isdigit() else 0

        ranked = sorted(
            docs,
            key=lambda meta: (-year_value(meta), str(meta.get("title", "") or "").lower()),
        )
        lines: list[str] = []
        for meta in ranked[:limit]:
            title = str(meta.get("title", "") or "").strip()
            if not title:
                continue
            year = str(meta.get("year", "") or "").strip()
            paper_id = str(meta.get("paper_id", "") or "").strip()
            tags = [tag for tag in str(meta.get("tags", "") or "").split("||") if tag]
            categories = collection_paths.get(paper_id) or tags[:1]
            suffix_parts = []
            if year:
                suffix_parts.append(year)
            if categories:
                suffix_parts.append(str(categories[0]))
            suffix = f"（{' · '.join(suffix_parts)}）" if suffix_parts else ""
            lines.append(f"- 《{title}》{suffix}")
        return lines

    def _rank_library_papers_for_recommendation(
        self,
        *,
        docs: list[dict[str, object]],
        query: str,
        limit: int = 3,
    ) -> list[dict[str, str]]:
        lowered_query = str(query or "").lower()
        wants_recent = any(marker in lowered_query for marker in ["最新", "最近", "newest", "latest", "recent"])
        wants_survey = any(marker in lowered_query for marker in ["综述", "survey", "review", "入门", "overview"])
        scored: list[tuple[float, dict[str, str]]] = []
        for meta in docs:
            title = str(meta.get("title", "") or "").strip()
            if not title:
                continue
            year_text = str(meta.get("year", "") or "").strip()
            tags = [tag for tag in str(meta.get("tags", "") or "").split("||") if tag]
            summary = str(meta.get("generated_summary") or meta.get("abstract_note") or "").strip()
            text = " ".join([title, summary, " ".join(tags)]).lower()
            score = 0.0
            if wants_recent and year_text.isdigit():
                score += max(0, int(year_text) - 2000) * 0.2
            if wants_survey and any(token in text for token in ["survey", "review", "overview", "综述"]):
                score += 8.0
            signal_weights = {
                "survey": 2.2,
                "benchmark": 2.0,
                "foundational": 3.0,
                "foundation": 2.4,
                "seminal": 3.0,
                "introduce": 2.0,
                "introduces": 2.0,
                "introduced": 1.8,
                "propose": 1.2,
                "all you need": 3.5,
                "comprehensive": 1.5,
                "framework": 1.0,
            }
            for token, weight in signal_weights.items():
                if token in text:
                    score += weight
            score += min(len(tags), 5) * 0.15
            if year_text.isdigit() and not wants_recent:
                year = int(year_text)
                if 2016 <= year <= 2024:
                    score += 0.8
                elif year > 2024:
                    score += 0.35
            reason = self._library_recommendation_reason(title=title, year=year_text, summary=summary, tags=tags)
            scored.append(
                (
                    score,
                    {
                        "title": title,
                        "year": year_text,
                        "paper_id": str(meta.get("paper_id", "") or "").strip(),
                        "tags": "、".join(tags[:5]),
                        "summary": summary[:360],
                        "reason": reason,
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]["title"].lower()))
        return [item for _score, item in scored[:limit]]

    @staticmethod
    def _library_recommendation_reason(*, title: str, year: str, summary: str, tags: list[str]) -> str:
        compact = " ".join(summary.split())
        if compact:
            if len(compact) > 140:
                compact = compact[:137].rstrip() + "..."
            return compact
        if tags:
            return f"它覆盖 {', '.join(tags[:3])} 等主题，适合作为进入当前库主题的切入口。"
        suffix = f"（{year}）" if year else ""
        return f"这是库中元数据完整的一篇论文{suffix}，适合作为默认起点。"

    def _summarize_figure_text(
        self,
        *,
        contract: QueryContract,
        fallback_text: str,
        evidence: list[EvidenceBlock],
    ) -> str:
        prompt = (
            f"query={contract.clean_query}\n"
            f"fallback={fallback_text}\n"
            f"evidence={json.dumps([item.snippet[:280] for item in evidence[:6]], ensure_ascii=False)}"
        )
        llm_text = self.clients.invoke_text(
            system_prompt=(
                "你是论文 figure 回答整理器。请基于图注和邻近文本，用简洁中文 Markdown 概括 figure 展示的内容。"
                "优先输出一个结论句，再补 2-4 条 bullet，说明比较对象、关键 benchmark、总体结论。"
                "不要逐字转抄原文，不要编造未出现的数值。"
            ),
            human_prompt=prompt,
            fallback="",
        )
        return llm_text.strip()
