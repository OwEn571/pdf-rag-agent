from __future__ import annotations

from app.domain.models import Claim


def compose_followup_research_answer(*, claims: list[Claim]) -> str:
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
        reason = followup_public_reason(item)
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
        return compose_selected_followup_candidate_answer(
            rows=rows,
            selected_candidate_title=selected_candidate_title,
            entity=entity,
            seed_text=seed_text,
        )
    parts: list[str] = ["## 检索结论", ""]
    if seed_text:
        parts.extend([seed_text, ""])
    if direct_body:
        parts.extend(["我能确认到一些直接后续/使用证据；其余论文需要按相关候选阅读。", "", "## 直接后续/使用证据", "", *direct_body[:5]])
    else:
        parts.extend(
            [
                "本地库当前没有足够证据确认严格意义上的后续工作，也就是没有稳定看到“明确使用、继承、引用或评测该种子论文/数据集”的关系证据。",
                "",
            ]
        )
    if strong_body:
        parts.extend(["", "## 强相关延续候选", "", *strong_body[:6]])
    if weak_body:
        parts.extend(["", "## 同主题但待确认", "", *weak_body[:4]])
    parts.extend(
        [
            "",
            "## 读法建议",
            "",
            "- 如果你要找“严格后续工作”，优先看第一组；如果第一组为空，就需要继续做引用链或 Web 验证。",
            "- 第二、三组更适合当作 related work 线索，不宜直接写成“使用了 AlignX/继承了 AlignX”。",
        ]
    )
    return "\n".join(parts).strip()


def compose_selected_followup_candidate_answer(
    *,
    rows: list[object],
    selected_candidate_title: str,
    entity: str,
    seed_text: str,
) -> str:
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
    reason = followup_public_reason(selected)
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
    parts.extend(
        [
            "",
            "## 写法建议",
            "",
            "- 如果正文需要严谨表述，优先写成 related work / strong continuation candidate。",
            "- 只有看到明确使用、继承、引用或评测种子论文/数据集的证据后，再写成严格后续工作。",
        ]
    )
    return "\n".join(parts).strip()


def followup_public_reason(item: dict[str, object]) -> str:
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
