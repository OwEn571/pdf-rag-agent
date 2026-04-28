from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any


FormulaTermExtractor = Callable[[str], list[str]]


def formula_payload_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_formulas = payload.get("formulas")
    candidates: list[dict[str, Any]] = []
    if isinstance(raw_formulas, list):
        candidates.extend(item for item in raw_formulas if isinstance(item, dict))
    candidates.append(payload)
    return candidates


def llm_formula_payload_from_response(
    payload: Any,
    *,
    allowed_evidence_ids: set[str],
    term_extractor: FormulaTermExtractor,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for formula_payload in formula_payload_candidates(payload):
        formula_text = normalize_extracted_formula_text(
            str(formula_payload.get("formula_text") or formula_payload.get("formula_latex") or "").strip()
        )
        if not formula_text:
            continue
        raw_evidence_ids = formula_payload.get("evidence_ids", [])
        if isinstance(raw_evidence_ids, str):
            raw_evidence_ids = [raw_evidence_ids]
        evidence_ids = [str(item).strip() for item in raw_evidence_ids if str(item).strip() in allowed_evidence_ids]
        if not evidence_ids:
            continue
        variables = normalize_formula_variables(formula_payload.get("variables"))
        terms = formula_terms_from_variables(variables, term_extractor=term_extractor)
        formula_format = str(formula_payload.get("formula_format") or "").strip().lower()
        if formula_format not in {"latex", "text"}:
            formula_format = "latex" if looks_like_latex_formula(formula_text) else "text"
        return {
            "formula_text": formula_text,
            "formula_latex": formula_text if formula_format == "latex" else "",
            "evidence_ids": evidence_ids,
            "terms": list(dict.fromkeys(terms)),
            "variables": variables,
            "formula_format": formula_format,
            "source": "llm_formula_extractor",
            "confidence": formula_payload.get("confidence", 0.78),
        }
    return {}


def formula_terms_from_variables(
    variables: list[dict[str, str]],
    *,
    term_extractor: FormulaTermExtractor,
) -> list[str]:
    text = "\n".join(
        " ".join([str(item.get("symbol", "")), str(item.get("description", ""))])
        for item in variables
    )
    return term_extractor(text)


def normalize_formula_variables(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    variables: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        symbol = ""
        description = ""
        if isinstance(item, dict):
            symbol = str(item.get("symbol") or item.get("name") or "").strip()
            description = str(
                item.get("description")
                or item.get("meaning")
                or item.get("role")
                or item.get("definition")
                or ""
            ).strip()
        else:
            raw = str(item or "").strip()
            if ":" in raw:
                symbol, description = [part.strip() for part in raw.split(":", 1)]
            elif "：" in raw:
                symbol, description = [part.strip() for part in raw.split("：", 1)]
            else:
                symbol = raw
        symbol = normalize_formula_variable_symbol(symbol)
        description = " ".join(description.split())[:260]
        if not symbol:
            continue
        key = (symbol.lower(), description.lower())
        if key in seen:
            continue
        seen.add(key)
        variables.append({"symbol": symbol, "description": description})
        if len(variables) >= 16:
            break
    return variables


def normalize_formula_variable_symbol(symbol: str) -> str:
    normalized = " ".join(str(symbol or "").strip().split())
    normalized = normalize_latex_like_math(normalized)
    return normalized[:120]


def normalize_formula_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", str(text or ""))
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def normalize_extracted_formula_text(text: str) -> str:
    formula = " ".join(str(text or "").strip().split())
    if not formula:
        return ""
    formula = normalize_formula_label(formula)
    formula = normalize_latex_like_math(formula)
    formula = normalize_formula_label(formula)
    formula = re.sub(r"\s+", " ", formula).strip()
    return formula[:1200]


def latex_symbol_token(value: str) -> str:
    token = str(value or "").strip()
    return {
        "theta": r"\theta",
        "θ": r"\theta",
        "phi": r"\phi",
        "ϕ": r"\phi",
        "φ": r"\phi",
    }.get(token, token)


def normalize_latex_like_math(text: str) -> str:
    normalized = str(text or "")

    def replace_gradient_loss(match: re.Match[str]) -> str:
        var = latex_symbol_token(match.group("var"))
        label = str(match.group("label") or "").strip().upper()
        return rf"\nabla_{{{var}}}\mathcal{{L}}_{{\mathrm{{{label}}}}}"

    def replace_gradient(match: re.Match[str]) -> str:
        var = latex_symbol_token(match.group("var"))
        return rf"\nabla_{{{var}}}"

    normalized = re.sub(
        r"∇\s*_?\s*(?P<var>θ|theta|ϕ|φ|phi|[A-Za-z])\s*L(?P<label>[A-Za-z][A-Za-z0-9\-]{1,12})\b",
        replace_gradient_loss,
        normalized,
    )
    normalized = re.sub(
        r"∇\s*_?\s*(?P<var>θ|theta|ϕ|φ|phi|[A-Za-z])",
        replace_gradient,
        normalized,
    )
    normalized = re.sub(
        r"(?:β|\\beta)\s*log(?=\s*(?:π|\\pi|[A-Za-z{_]))",
        r"\\beta \\log ",
        normalized,
    )
    replacements = {
        "log σ": r"\log \sigma",
        "logσ": r"\log \sigma",
        "πref": r"\pi_{\mathrm{ref}}",
        "π_ref": r"\pi_{\mathrm{ref}}",
        "π_θ": r"\pi_{\theta}",
        "πθ": r"\pi_{\theta}",
        "π_ϕ": r"\pi_{\phi}",
        "πϕ": r"\pi_{\phi}",
        "π_φ": r"\pi_{\phi}",
        "πφ": r"\pi_{\phi}",
        "P̃": r"\tilde{P}",
        "p̃": r"\tilde{p}",
        "𝜎": r"\sigma",
        "σ": r"\sigma",
        "β": r"\beta",
        "θ": r"\theta",
        "ϕ": r"\phi",
        "φ": r"\phi",
        "ϵ": r"\epsilon",
        "ε": r"\epsilon",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\\betalog(?=\s*(?:\\pi|[A-Za-z{_]))", r"\\beta \\log ", normalized)
    normalized = re.sub(r"(?<!\\)\blog(?=\s*(?:\\pi|\\sigma|[A-Za-z{_]))", r"\\log ", normalized)
    normalized = re.sub(r"\byw\b", r"y_w", normalized)
    normalized = re.sub(r"\byl\b", r"y_l", normalized)
    normalized = re.sub(r"\)\s*~\s*([A-Za-z])\s*\[", r") \\sim \1[", normalized)
    return normalized


def normalize_formula_label(text: str) -> str:
    def replace_subscript_label(match: re.Match[str]) -> str:
        label = str(match.group("label") or "").strip("{} ")
        return r"L_{\mathrm{" + label.upper() + "}}"

    normalized = re.sub(
        r"\bL\s*(?:_|\{)\s*\{?\s*(?:\\mathrm\{?)?\s*(?P<label>[A-Za-z][A-Za-z0-9\-]{1,12})\s*\}?\s*\}?",
        replace_subscript_label,
        text,
    )
    normalized = re.sub(
        r"\bL(?P<label>[A-Z][A-Z0-9\-]{1,12})\b",
        replace_subscript_label,
        normalized,
    )
    return normalized


def best_formula_window(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    scored: list[tuple[float, int]] = []
    formula_tokens = ["=", "∇", "sigma", "σ", "mathcal", "frac", "π", " pi", "loss", "objective", "l_"]
    noise_tokens = ["what does", "mechanistic understanding", "theoretical properties", "in section"]
    for index, line in enumerate(lines):
        lowered = line.lower()
        score = sum(1.0 for token in formula_tokens if token in lowered)
        score += min(2.0, lowered.count("="))
        score -= sum(1.2 for token in noise_tokens if token in lowered)
        scored.append((score, index))
    _, best_index = max(scored, key=lambda item: (item[0], -item[1]))
    start = max(0, best_index - 1)
    end = min(len(lines), best_index + 3)
    window = " ".join(lines[start:end])
    window = re.sub(r"\s+", " ", window).strip()
    return window[:900]


def looks_like_latex_formula(text: str) -> bool:
    return any(token in str(text or "") for token in ["\\mathcal", "\\frac", "\\pi_", "\\sigma", "\\nabla"])


def formula_block_score(
    text: str,
    *,
    query: str | None = None,
    token_weights: Mapping[str, float] | None = None,
) -> float:
    haystack = str(text or "").lower()
    score = 0.0
    for token, weight in (token_weights or {}).items():
        if str(token).lower() in haystack:
            score += float(weight)
    if query is not None and not formula_query_wants_gradient(query):
        if any(token in haystack for token in ["objective", "loss", "目标函数", "损失"]):
            score += 2.5
        if any(token in haystack for token in ["\\nabla", "∇", "gradient", "梯度"]):
            score -= 2.5
    if "figure " in haystack or haystack.startswith("figure"):
        score -= 2.0
    if any(token in haystack for token in ["a.4", "appendix", "plackett-luce", "rankings"]):
        score -= 4.0
    return max(0.0, score)


def formula_query_wants_gradient(query: str) -> bool:
    normalized = " ".join(str(query or "").lower().split())
    compact = normalized.replace(" ", "")
    return any(
        marker in normalized or marker in compact
        for marker in ["gradient", "grad", "derivative", "update", "梯度", "导数", "求导", "更新", "推导"]
    )
