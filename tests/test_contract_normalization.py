from __future__ import annotations

from app.services.contracts.normalization import (
    clean_contract_target_text,
    is_structural_target_reference,
    normalize_contract_targets,
    normalize_lookup_text,
    normalize_modalities,
)


def test_contract_normalization_cleans_task_suffixes_and_acronyms() -> None:
    assert clean_contract_target_text(" PBA 的目标函数 ") == "PBA"
    assert clean_contract_target_text("DPOformula") == "DPO"
    assert clean_contract_target_text("Direct Preference Optimization paper") == "Direct Preference Optimization"


def test_contract_normalization_drops_structural_targets_and_requested_fields() -> None:
    targets = normalize_contract_targets(
        targets=["PBA 公式", "figure1", "PBA", "definition"],
        requested_fields=["definition"],
        canonicalize_targets=lambda values: values,
    )

    assert targets == ["PBA"]
    assert is_structural_target_reference("figure1")
    assert is_structural_target_reference("图 1")
    assert is_structural_target_reference("Table-2a")
    assert not is_structural_target_reference("DPO")


def test_contract_normalization_modalities_use_aliases_and_relation_defaults() -> None:
    assert normalize_modalities(["visual"], relation="general_question") == ["figure", "caption", "page_text"]
    assert normalize_modalities([], relation="formula_lookup") == ["page_text", "table"]
    assert normalize_modalities([], relation="figure_question") == ["figure", "caption", "page_text"]
    assert normalize_modalities([], relation="metric_value_lookup") == ["table", "caption", "page_text"]
    assert normalize_modalities([], relation="general_question") == ["page_text", "paper_card"]


def test_normalize_lookup_text_matches_title_key_semantics() -> None:
    assert normalize_lookup_text("  A   Mixed CASE Title  ") == "a mixed case title"
    assert normalize_lookup_text("") == ""
