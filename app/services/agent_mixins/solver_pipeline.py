from __future__ import annotations

import logging

from app.domain.models import CandidatePaper, Claim, EvidenceBlock, QueryContract, ResearchPlan, SessionContext
from app.services.claims.concept_definition_solver import solve_concept_definition_claims
from app.services.claims.deterministic_solver import solve_claims_with_deterministic_fallback
from app.services.claims.deterministic_runner import DeterministicSolverHandlers
from app.services.claims.entity_definition_solver import solve_entity_definition_claims
from app.services.claims.followup_research_solver import solve_followup_research_claims
from app.services.claims.formula_solver import solve_formula_claims
from app.services.claims.figure_solver import solve_figure_claims
from app.services.claims.generic_solver import solve_claims_with_generic_schema
from app.services.claims.origin_solver import solve_origin_lookup_claims
from app.services.claims.solver_pipeline import run_claim_solver_pipeline
from app.services.claims.table_solver import solve_table_claims
from app.services.claims.text_solver import (
    solve_default_text_claims,
    solve_metric_context_claims,
    solve_paper_recommendation_claims,
    solve_paper_summary_results_claims,
    solve_topology_discovery_claims,
    solve_topology_recommendation_claims,
)
from app.services.contracts.session_context import agent_session_conversation_context
from app.services.planning.schema_claims import should_use_schema_claim_solver

logger = logging.getLogger(__name__)

# P2-3: Removed dead _DETERMINISTIC_SOLVER_REGISTRY (duplicated
# deterministic_runner._STAGE_HANDLER_TABLE + solver_dispatch._DETERMINISTIC_STAGE_TABLE).
# To add a new solver: (1) add stage in _DETERMINISTIC_STAGE_TABLE,
# (2) add handler attr in _STAGE_HANDLER_TABLE,
# (3) wire the lambda in _deterministic_solver_handlers().


class SolverPipelineMixin:
    def _run_solvers(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        self._last_generic_claim_solver_shadow = {}
        agent_settings = getattr(self, "agent_settings", None)
        schema_allowed = should_use_schema_claim_solver(
            contract=contract,
            plan=plan,
            agent_settings=agent_settings,
        )
        generic_enabled = bool(getattr(agent_settings, "generic_claim_solver_enabled", False))
        shadow_enabled = bool(getattr(agent_settings, "generic_claim_solver_shadow_enabled", False))
        result = run_claim_solver_pipeline(
            schema_allowed=schema_allowed,
            generic_enabled=generic_enabled,
            shadow_enabled=shadow_enabled,
            solve_schema=lambda: self._run_schema_claim_solver(
                contract=contract,
                plan=plan,
                papers=papers,
                evidence=evidence,
                session=session,
            ),
            solve_deterministic=lambda: self._run_deterministic_claim_solver(
                contract=contract,
                plan=plan,
                papers=papers,
                evidence=evidence,
                session=session,
            ),
        )
        self._last_generic_claim_solver_shadow = {"selected": result.selected, **result.shadow}
        return result.claims

    def _run_schema_claim_solver(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        return solve_claims_with_generic_schema(
            clients=self.clients,
            contract=contract,
            plan=plan,
            papers=papers,
            evidence=evidence,
            conversation_context=agent_session_conversation_context(
                session,
                settings=self.settings,
                max_chars=12000,
            ),
        )

    def _run_deterministic_claim_solver(
        self,
        *,
        contract: QueryContract,
        plan: ResearchPlan,
        papers: list[CandidatePaper],
        evidence: list[EvidenceBlock],
        session: SessionContext,
    ) -> list[Claim]:
        return solve_claims_with_deterministic_fallback(
            handlers=self._deterministic_solver_handlers(),
            contract=contract,
            plan=plan,
            papers=papers,
            evidence=evidence,
            session=session,
        )

    def _deterministic_solver_handlers(self) -> DeterministicSolverHandlers:
        return DeterministicSolverHandlers(
            origin_lookup=lambda *, contract, papers, evidence, session: solve_origin_lookup_claims(
                contract=contract,
                papers=papers,
                evidence=evidence,
                paper_documents=self.retriever.paper_documents(),
                candidate_from_paper_id=self._candidate_from_paper_id,
                paper_identity_matches_targets=lambda paper, targets: self._paper_identity_matches_targets(
                    paper=paper,
                    targets=targets,
                ),
            ),
            formula=lambda *, contract, papers, evidence: solve_formula_claims(
                clients=self.clients,
                contract=contract,
                papers=papers,
                evidence=evidence,
                retrieval_formula_token_weights=self.settings.retrieval_formula_token_weights,
            ),
            followup_research=lambda *, contract, papers, evidence, session: solve_followup_research_claims(
                clients=self.clients,
                retriever=self.retriever,
                paper_limit_default=int(self.settings.paper_limit_default),
                contract=contract,
                papers=papers,
                evidence=evidence,
                session=session,
            ),
            figure=lambda *, contract, papers, evidence: solve_figure_claims(
                clients=self.clients,
                settings=self.settings,
                rendered_page_data_url_cache=self._rendered_page_data_url_cache,
                contract=contract,
                papers=papers,
                evidence=evidence,
                logger=logger,
            ),
            table=lambda *, contract, papers, evidence: solve_table_claims(
                clients=self.clients,
                settings=self.settings,
                rendered_page_data_url_cache=self._rendered_page_data_url_cache,
                contract=contract,
                papers=papers,
                evidence=evidence,
                paper_identity_matches_targets=lambda paper, targets: self._paper_identity_matches_targets(
                    paper=paper,
                    targets=targets,
                ),
                logger=logger,
            ),
            metric_context=lambda *, contract, papers, evidence, session: solve_metric_context_claims(
                contract=contract,
                papers=papers,
                evidence=evidence,
                solver_metric_token_weights=self.settings.solver_metric_token_weights,
                paper_identity_matches_targets=lambda paper, targets: self._paper_identity_matches_targets(
                    paper=paper,
                    targets=targets,
                ),
            ),
            paper_recommendation=lambda *, contract, papers, evidence, session: solve_paper_recommendation_claims(
                contract=contract,
                papers=papers,
                paper_doc_lookup=self.retriever.paper_doc_by_id,
            ),
            topology_recommendation=lambda *, contract, papers, evidence, session: solve_topology_recommendation_claims(
                clients=self.clients,
                evidence=evidence,
            ),
            topology_discovery=lambda *, contract, papers, evidence, session: solve_topology_discovery_claims(
                papers=papers,
                evidence=evidence,
            ),
            paper_summary_results=lambda *, contract, papers, evidence, session: solve_paper_summary_results_claims(
                contract=contract,
                papers=papers,
                evidence=evidence,
                solver_metric_token_weights=self.settings.solver_metric_token_weights,
                paper_doc_lookup=self.retriever.paper_doc_by_id,
                paper_identity_matches_targets=lambda paper, targets: self._paper_identity_matches_targets(
                    paper=paper,
                    targets=targets,
                ),
            ),
            default_text=lambda *, contract, papers, evidence, session: solve_default_text_claims(
                contract=contract,
                papers=papers,
                evidence=evidence,
                paper_doc_lookup=self.retriever.paper_doc_by_id,
            ),
            entity_definition=lambda *, contract, papers, evidence, session: solve_entity_definition_claims(
                contract=contract,
                papers=papers,
                evidence=evidence,
                select_supporting_paper=lambda current_contract, current_papers, current_evidence: self._select_entity_supporting_paper(
                    contract=current_contract,
                    papers=current_papers,
                    evidence=current_evidence,
                ),
                infer_entity_type=lambda current_contract, current_papers, current_evidence: self._infer_entity_type(
                    contract=current_contract,
                    papers=current_papers,
                    evidence=current_evidence,
                ),
                entity_supporting_lines=lambda current_evidence, kind: self._entity_supporting_lines(
                    current_evidence,
                    kind=kind,
                ),
            ),
            concept_definition=lambda *, contract, papers, evidence, session: solve_concept_definition_claims(
                clients=self.clients,
                paper_doc_lookup=self.retriever.paper_doc_by_id,
                contract=contract,
                papers=papers,
                evidence=evidence,
            ),
        )
