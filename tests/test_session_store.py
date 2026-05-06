from __future__ import annotations

from app.domain.models import QueryContract, SessionTurn
from app.services.memory.session_store import SQLiteSessionStore


def test_sqlite_session_store_persists_context(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite3"
    first = SQLiteSessionStore(db_path, max_turns=3)
    first.append_turn("demo", SessionTurn(query="q1", answer="a1", relation="capability"))

    second = SQLiteSessionStore(db_path, max_turns=3)
    restored = second.get("demo")

    assert restored.session_id == "demo"
    assert len(restored.turns) == 1
    assert restored.turns[0].answer == "a1"


def test_sqlite_session_store_trims_old_turns_into_summary(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3", max_turns=2)

    for index in range(4):
        store.append_turn(
            "demo",
            SessionTurn(
                query=f"q{index}",
                answer=f"a{index}",
                relation="research",
                titles=[f"paper-{index}"],
            ),
        )

    context = store.get("demo")

    assert [turn.query for turn in context.turns] == ["q2", "q3"]
    assert "q0" in context.summary
    assert "q1" in context.summary


def test_sqlite_commit_turn_persists_mutated_context(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3", max_turns=3)
    context = store.get("demo")
    context.last_relation = "entity_definition"
    context.working_memory = {"target_bindings": {"ppo": {"title": "PPO paper"}}}
    context.set_active_research(
        relation="entity_definition",
        targets=["PPO"],
        titles=["PPO paper"],
        requested_fields=["definition"],
        required_modalities=["page_text"],
        answer_shape="narrative",
        precision_requirement="high",
        clean_query="PPO 是什么",
    )
    turn = SessionTurn.from_contract(
        query="PPO 是什么",
        answer="answer",
        contract=QueryContract(
            clean_query="PPO 是什么",
            relation="entity_definition",
            targets=["PPO"],
            requested_fields=["definition"],
            required_modalities=["page_text"],
            precision_requirement="high",
        ),
        titles=["PPO paper"],
    )

    store.commit_turn(context, turn)
    restored = store.get("demo")

    assert restored.last_relation == "entity_definition"
    assert restored.active_research.targets == ["PPO"]
    assert restored.active_targets == ["PPO"]
    assert restored.working_memory["target_bindings"]["ppo"]["title"] == "PPO paper"
    assert restored.turns[0].titles == ["PPO paper"]


def test_sqlite_commit_turn_promotes_legacy_active_fields(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3", max_turns=3)
    context = store.get("demo")
    context.active_research_relation = "formula_lookup"
    context.active_targets = ["PBA"]
    context.active_titles = ["AlignX paper"]
    context.active_requested_fields = ["formula", "variable_explanation"]
    context.active_required_modalities = ["page_text", "table"]
    context.active_answer_shape = "bullets"
    context.active_precision_requirement = "exact"
    context.active_clean_query = "PBA 公式是什么"
    turn = SessionTurn.from_contract(
        query="PBA 公式是什么",
        answer="answer",
        contract=QueryContract(
            clean_query="PBA 公式是什么",
            relation="formula_lookup",
            targets=["PBA"],
            requested_fields=["formula"],
            required_modalities=["page_text"],
            precision_requirement="exact",
        ),
        titles=["AlignX paper"],
    )

    store.commit_turn(context, turn)
    restored = store.get("demo")

    assert restored.active_research.relation == "formula_lookup"
    assert restored.active_research.targets == ["PBA"]
    assert restored.active_research.titles == ["AlignX paper"]
    assert restored.active_research.precision_requirement == "exact"
    assert restored.active_research.last_topic_signature
    assert restored.active_targets == ["PBA"]
