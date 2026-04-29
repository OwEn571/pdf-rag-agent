from __future__ import annotations

from app.domain.models import QueryContract
from app.services.agent_step_messages import agent_step_message


def test_agent_step_message_includes_targets_for_search_tools() -> None:
    contract = QueryContract(clean_query="DPO 公式", relation="formula_lookup", targets=["DPO"])

    assert "DPO" in agent_step_message(action="search_corpus", contract=contract)
    assert "formula_lookup" in agent_step_message(action="understand_user_intent", contract=contract)


def test_agent_step_message_falls_back_to_action_name() -> None:
    contract = QueryContract(clean_query="hello")

    assert agent_step_message(action="custom_tool", contract=contract) == "custom_tool"
