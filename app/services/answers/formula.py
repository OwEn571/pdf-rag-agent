from __future__ import annotations

import re

from app.domain.models import Claim, QueryContract
from app.services.contracts.context import contract_has_note, contract_note_json_value
from app.services.intents.marker_matching import MarkerProfile, query_matches_any


FORMULA_ANSWER_MARKERS: dict[str, MarkerProfile] = {
    "formula_math_symbol": (
        "\\",
        "_",
        "^",
        "{",
        "}",
        "(",
        ")",
        "π",
        "σ",
        "β",
        "θ",
        "∇",
        "ϕ",
        "φ",
        "ϵ",
        "ε",
        "phi",
        "theta",
        "sigma",
        "beta",
        "epsilon",
        "pi_",
    ),
}


def normalize_markdown_math_artifacts(text: str) -> str:
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


def compose_formula_answer(*, claims: list[Claim], contract: QueryContract | None = None) -> str:
    if not claims:
        return ""
    notice = auto_resolved_candidate_notice(contract)
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
            all_term_lines.extend(formula_term_lines(claim))
        term_lines = list(dict.fromkeys(all_term_lines))
        if term_lines:
            sections.extend(["", "## 变量", "", *term_lines])
        answer = "\n".join(sections).strip()
        return f"{notice}\n\n{answer}".strip() if notice else answer
    claim = formula_claims[0]
    formula_text = str(claim.value or "").strip()
    if not formula_text:
        return ""
    term_lines = formula_term_lines(claim)
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


def auto_resolved_candidate_notice(contract: QueryContract | None) -> str:
    if contract is None or not contract_has_note(contract, "auto_resolved_by_llm_judge"):
        return ""
    selected = contract_note_json_value(contract, prefix="selected_ambiguity_option=")
    title = str(selected.get("display_title") or selected.get("title") or "").strip()
    label = str(selected.get("display_label") or selected.get("label") or selected.get("meaning") or "").strip()
    if title:
        return f"我按最匹配的候选《{title}》来回答。"
    if label:
        return f"我按最匹配的候选“{label}”来回答。"
    return "我按当前候选中最匹配的一项来回答。"


def formula_term_lines(claim: Claim) -> list[str]:
    structured = dict(claim.structured_data or {})
    variable_lines = formula_variable_lines(structured.get("variables"))
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


def formula_variable_lines(value: object) -> list[str]:
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
        symbol_markdown = format_formula_symbol(symbol)
        description_markdown = format_formula_description(description)
        line = f"- {symbol_markdown}：{description_markdown}"
        key = " ".join(line.lower().split())
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return lines


def format_formula_symbol(symbol: str) -> str:
    compact = " ".join(str(symbol or "").strip().split())
    if not compact:
        return "`?`"
    if query_matches_any(compact, "", FORMULA_ANSWER_MARKERS["formula_math_symbol"]):
        return f"${compact}$"
    if re.fullmatch(r"[A-Za-z](?:_[A-Za-z0-9]+)?", compact):
        return f"${compact}$"
    return f"`{compact}`"


def format_formula_description(description: str) -> str:
    text = str(description or "").strip()
    if not text:
        return ""
    text = normalize_markdown_math_artifacts(text)

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
