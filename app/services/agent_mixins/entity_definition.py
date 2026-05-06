from __future__ import annotations

from typing import Any

from app.domain.models import CandidatePaper, EvidenceBlock, QueryContract
from app.services.answers.entity import (
    compose_entity_answer_markdown,
    compose_entity_description,
    entity_clean_lines,
    entity_focus_lines,
    entity_intro_sentence,
    entity_mechanism_bullets,
    entity_reward_bullets,
    entity_summary_bullets,
    entity_supporting_lines,
    entity_workflow_steps,
    sanitize_entity_description,
)
from app.services.entities.definition_profiles import ENTITY_DEFINITION_MARKERS
from app.services.entities.supporting_paper_selector import (
    best_entity_fallback_paper,
    candidate_from_paper_id,
    entity_context_identity_matches,
    entity_context_matches,
    entity_definition_score,
    ground_entity_papers,
    is_noisy_entity_line,
    llm_select_entity_supporting_paper,
    paper_introduces_context_target,
    prune_entity_supporting_evidence,
    select_entity_supporting_paper,
)
from app.services.entities.type_inference import infer_entity_type


class EntityDefinitionMixin:
    def _candidate_from_paper_id(self, paper_id: str) -> CandidatePaper | None:
        return candidate_from_paper_id(paper_id, paper_doc_lookup=self.retriever.paper_doc_by_id)

    def _ground_entity_papers(
        self,
        *,
        candidates: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        limit: int,
    ) -> list[CandidatePaper]:
        return ground_entity_papers(
            candidates=candidates,
            evidence=evidence,
            limit=limit,
            paper_lookup=self._candidate_from_paper_id,
        )

    def _select_entity_supporting_paper(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
    ) -> tuple[CandidatePaper | None, list[EvidenceBlock]]:
        return select_entity_supporting_paper(
            clients=self.clients,
            paper_doc_lookup=self.retriever.paper_doc_by_id,
            paper_identity_matches_targets=lambda paper, targets: self._paper_identity_matches_targets(
                paper=paper,
                targets=targets,
            ),
            contract=contract,
            papers=papers,
            evidence=evidence,
        )

    @staticmethod
    def _best_entity_fallback_paper(*, papers: list[CandidatePaper], evidence: list[EvidenceBlock]) -> CandidatePaper:
        return best_entity_fallback_paper(papers=papers, evidence=evidence)

    def _entity_context_identity_matches(self, *, item: EvidenceBlock, context_targets: list[str]) -> bool:
        return entity_context_identity_matches(
            item=item,
            context_targets=context_targets,
            paper_lookup=self._candidate_from_paper_id,
            paper_identity_matches_targets=lambda paper, targets: self._paper_identity_matches_targets(
                paper=paper,
                targets=targets,
            ),
        )

    def _paper_introduces_context_target(self, *, paper: CandidatePaper, context_targets: list[str]) -> bool:
        return paper_introduces_context_target(paper=paper, context_targets=context_targets)

    def _entity_context_matches(self, *, item: EvidenceBlock, context_targets: list[str]) -> bool:
        return entity_context_matches(
            item=item,
            context_targets=context_targets,
            paper_lookup=self._candidate_from_paper_id,
        )

    def _llm_select_entity_supporting_paper(
        self,
        *,
        contract: QueryContract,
        papers: list[CandidatePaper],
        matching_evidence: list[EvidenceBlock],
    ) -> tuple[CandidatePaper | None, list[EvidenceBlock]]:
        return llm_select_entity_supporting_paper(
            clients=self.clients,
            paper_doc_lookup=self.retriever.paper_doc_by_id,
            contract=contract,
            papers=papers,
            matching_evidence=matching_evidence,
        )

    def _prune_entity_supporting_evidence(self, evidence: list[EvidenceBlock]) -> list[EvidenceBlock]:
        return prune_entity_supporting_evidence(evidence)

    def _entity_supporting_lines(self, evidence: list[EvidenceBlock], *, kind: str) -> list[str]:
        return entity_supporting_lines(evidence, kind=kind)

    @staticmethod
    def _entity_definition_score(text: str) -> int:
        return entity_definition_score(text)

    def _infer_entity_type(self, *, contract: QueryContract, papers: list[CandidatePaper], evidence: list[EvidenceBlock]) -> str:
        return infer_entity_type(
            clients=self.clients,
            paper_doc_lookup=self.retriever.paper_doc_by_id,
            contract=contract,
            papers=papers,
            evidence=evidence,
        )

    @staticmethod
    def _is_noisy_entity_line(text: str) -> bool:
        return is_noisy_entity_line(text)

    def _compose_entity_answer_markdown(
        self,
        *,
        contract: QueryContract,
        claims: list[Any],
        evidence: list[EvidenceBlock],
        citations: list[Any],
    ) -> str:
        return compose_entity_answer_markdown(
            contract=contract,
            claims=claims,
            evidence=evidence,
            citations=citations,
        )

    @staticmethod
    def _sanitize_entity_description(text: str) -> str:
        return sanitize_entity_description(text)

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
        return entity_intro_sentence(
            target=target,
            label=label,
            paper_title=paper_title,
            definition_lines=definition_lines,
            mechanism_lines=mechanism_lines,
            application_lines=application_lines,
            evidence=evidence,
        )

    def _entity_clean_lines(self, lines: list[str], *, limit: int) -> list[str]:
        return entity_clean_lines(lines, limit=limit)

    def _entity_mechanism_bullets(self, *, mechanism_lines: list[str], evidence: list[EvidenceBlock]) -> list[str]:
        return entity_mechanism_bullets(
            mechanism_lines=mechanism_lines,
            evidence=evidence,
        )

    def _entity_workflow_steps(self, *, evidence: list[EvidenceBlock]) -> list[str]:
        return entity_workflow_steps(evidence=evidence)

    def _entity_reward_bullets(self, *, evidence: list[EvidenceBlock]) -> list[str]:
        return entity_reward_bullets(evidence=evidence)

    def _entity_summary_bullets(
        self,
        *,
        definition_lines: list[str],
        mechanism_lines: list[str],
        application_lines: list[str],
    ) -> list[str]:
        return entity_summary_bullets(
            definition_lines=definition_lines,
            mechanism_lines=mechanism_lines,
            application_lines=application_lines,
        )

    def _entity_focus_lines(self, *, evidence: list[EvidenceBlock], keywords: list[str], limit: int) -> list[str]:
        return entity_focus_lines(evidence=evidence, keywords=keywords, limit=limit)

    def _compose_entity_description(
        self,
        *,
        contract: QueryContract,
        target: str,
        label: str,
        paper: CandidatePaper,
        evidence: list[EvidenceBlock],
    ) -> str:
        return compose_entity_description(
            clients=self.clients,
            paper_doc_lookup=self.retriever.paper_doc_by_id,
            contract=contract,
            target=target,
            label=label,
            paper=paper,
            evidence=evidence,
        )
