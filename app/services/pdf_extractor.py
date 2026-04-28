from __future__ import annotations

from functools import lru_cache
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from app.core.config import Settings

logger = logging.getLogger(__name__)

CAPTION_ANCHOR_RE = re.compile(r"(?im)^\s*(table|tab\.|figure|fig\.|表|图)\s*\d+")
TABLE_ANCHOR_RE = re.compile(r"(?im)^\s*(table|tab\.|表)\s*\d+")
FIGURE_ANCHOR_RE = re.compile(r"(?im)^\s*(figure|fig\.|图)\s*\d+")
NUMERIC_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?%?")
SEPARATOR_RE = re.compile(r"(?:\s{2,}|\t)")
HTML_TAG_RE = re.compile(r"<[^>]+>")

TABLE_LIKE_THRESHOLD = 2.5
FIGURE_LIKE_THRESHOLD = 2.5
SCANNED_LIKE_THRESHOLD = 2.0
SCANNED_TEXT_CHAR_THRESHOLD = 80
MAX_HI_RES_PAGES_PER_DOC = 6


@dataclass(slots=True)
class PageSignals:
    caption_anchor_count: int = 0
    table_anchor_count: int = 0
    figure_anchor_count: int = 0
    numeric_density: float = 0.0
    short_line_ratio: float = 0.0
    avg_tokens_per_line: float = 0.0
    separator_pattern_score: float = 0.0
    text_chars: int = 0
    image_object_count: int = 0
    table_like_score: float = 0.0
    figure_like_score: float = 0.0
    scanned_like_score: float = 0.0
    selected_reasons: tuple[str, ...] = ()


@dataclass(slots=True)
class ExtractedBlock:
    page: int
    block_type: str
    text: str
    bbox: tuple[float, float, float, float] | None = None
    caption: str = ""
    source_parser: str = "hi_res"


@dataclass(slots=True)
class ExtractedPage:
    page: int
    text: str
    blocks: list[ExtractedBlock] = field(default_factory=list)
    signals: PageSignals = field(default_factory=PageSignals)
    selected_for_hi_res: bool = False


class PDFExtractor:
    def __init__(self, settings: Settings | None = None, prefer_unstructured: bool = True) -> None:
        self.settings = settings
        self.prefer_unstructured = prefer_unstructured

    def extract_pages(self, pdf_path: Path) -> list[ExtractedPage]:
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("pypdf parse failed, fallback to unstructured fast if available: %s", exc)
            return self._extract_with_unstructured_fast(pdf_path)

        pages = self._extract_pages_with_pypdf(reader)
        selected_pages = self._select_hi_res_pages(pages)
        if selected_pages and self.prefer_unstructured:
            hi_res_blocks = self._extract_selected_hi_res_blocks(pdf_path, reader, selected_pages)
            for page in pages:
                page.selected_for_hi_res = page.page in selected_pages
                page.blocks = hi_res_blocks.get(page.page, [])
                if not page.text.strip():
                    page.text = self._fallback_page_text(page.blocks)

        if not any(page.text.strip() or page.blocks for page in pages) and self.prefer_unstructured:
            return self._extract_with_unstructured_fast(pdf_path)

        return [page for page in pages if page.text.strip() or page.blocks]

    def _extract_pages_with_pypdf(self, reader: PdfReader) -> list[ExtractedPage]:
        pages: list[ExtractedPage] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            image_count = self._count_page_images(page)
            signals = self._compute_page_signals(text=text, image_count=image_count)
            pages.append(ExtractedPage(page=idx, text=text, signals=signals))
        return pages

    def _extract_with_unstructured_fast(self, pdf_path: Path) -> list[ExtractedPage]:
        partition_pdf = self._load_partition_pdf()
        if partition_pdf is None:
            return []
        elements = partition_pdf(filename=str(pdf_path), strategy="fast")
        page_map: dict[int, list[str]] = {}
        for element in elements:
            text = (getattr(element, "text", "") or "").strip()
            if not text:
                continue
            metadata = getattr(element, "metadata", None)
            page_number = self._extract_page_number(metadata)
            page_map.setdefault(page_number, []).append(text)
        pages: list[ExtractedPage] = []
        for page_number, chunks in sorted(page_map.items(), key=lambda item: item[0]):
            page_text = "\n".join(chunks).strip()
            signals = self._compute_page_signals(text=page_text, image_count=0)
            pages.append(ExtractedPage(page=page_number, text=page_text, signals=signals))
        return pages

    def _extract_selected_hi_res_blocks(
        self,
        pdf_path: Path,
        reader: PdfReader,
        selected_pages: set[int],
    ) -> dict[int, list[ExtractedBlock]]:
        partition_pdf = self._load_partition_pdf()
        if partition_pdf is None:
            return {}
        blocks_by_page: dict[int, list[ExtractedBlock]] = {}
        with tempfile.TemporaryDirectory(prefix="zprag_v4_hi_res_") as temp_dir:
            for range_start, range_end in self._group_contiguous_pages(selected_pages):
                tmp_pdf = Path(temp_dir) / f"{pdf_path.stem}_pages_{range_start}_{range_end}.pdf"
                writer = PdfWriter()
                for page_number in range(range_start, range_end + 1):
                    writer.add_page(reader.pages[page_number - 1])
                with tmp_pdf.open("wb") as f:
                    writer.write(f)
                try:
                    elements = partition_pdf(
                        filename=str(tmp_pdf),
                        strategy="hi_res",
                        infer_table_structure=True,
                        starting_page_number=range_start,
                    )
                    for block in self._elements_to_blocks(elements):
                        blocks_by_page.setdefault(block.page, []).append(block)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "hi_res parse failed on file=%s pages=%s-%s: %s",
                        pdf_path.name,
                        range_start,
                        range_end,
                        exc,
                    )
        return blocks_by_page

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_partition_pdf() -> object | None:
        try:
            from unstructured.partition.pdf import partition_pdf
        except Exception as exc:  # noqa: BLE001
            logger.warning("unstructured.partition.pdf unavailable: %s", exc)
            return None
        return partition_pdf

    def _elements_to_blocks(self, elements: list[object]) -> list[ExtractedBlock]:
        captions: list[ExtractedBlock] = []
        blocks: list[ExtractedBlock] = []
        for element in elements:
            category = str(getattr(element, "category", "") or "")
            text = (getattr(element, "text", "") or "").strip()
            metadata = getattr(element, "metadata", None)
            page_number = self._extract_page_number(metadata)
            bbox = self._extract_bbox(metadata)
            text_as_html = getattr(metadata, "text_as_html", None)
            if self._is_caption(category=category, text=text):
                captions.append(ExtractedBlock(page=page_number, block_type="caption", text=text, bbox=bbox))
                continue
            if category == "Table":
                normalized_table = self._normalize_table_text(text=text, text_as_html=text_as_html)
                if normalized_table:
                    blocks.append(
                        ExtractedBlock(page=page_number, block_type="table", text=normalized_table, bbox=bbox)
                    )
                continue
            if category in {"Image", "Picture"}:
                blocks.append(ExtractedBlock(page=page_number, block_type="figure", text=text, bbox=bbox))
        for block in blocks:
            nearest_caption = self._match_nearest_caption(block, captions)
            if nearest_caption is not None:
                block.caption = nearest_caption.text
        filtered_blocks: list[ExtractedBlock] = []
        for block in blocks:
            if block.block_type == "figure" and not block.caption and self._looks_like_noisy_ocr(block.text):
                continue
            filtered_blocks.append(block)
        return [*captions, *filtered_blocks]

    def _select_hi_res_pages(self, pages: list[ExtractedPage]) -> set[int]:
        ranked_candidates: list[tuple[float, int, ExtractedPage]] = []
        for page in pages:
            reasons: list[str] = []
            signals = page.signals
            if signals.table_like_score >= TABLE_LIKE_THRESHOLD:
                reasons.append(f"table:{signals.table_like_score:.2f}")
            if signals.figure_like_score >= FIGURE_LIKE_THRESHOLD:
                reasons.append(f"figure:{signals.figure_like_score:.2f}")
            if signals.scanned_like_score >= SCANNED_LIKE_THRESHOLD:
                reasons.append(f"scanned:{signals.scanned_like_score:.2f}")
            if not reasons:
                continue
            page.signals.selected_reasons = tuple(reasons)
            ranked_candidates.append(
                (
                    max(signals.table_like_score, signals.figure_like_score, signals.scanned_like_score),
                    page.page,
                    page,
                )
            )
        if not ranked_candidates:
            return set()
        ranked_candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = ranked_candidates[: self._max_hi_res_pages_per_document()]
        return {page.page for _, _, page in selected}

    def _compute_page_signals(self, text: str, image_count: int) -> PageSignals:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        tokens = text.split()
        token_count = len(tokens)
        numeric_tokens = sum(1 for token in tokens if NUMERIC_TOKEN_RE.search(token))
        short_lines = sum(1 for line in lines if len(line.split()) <= 6)
        separator_lines = sum(1 for line in lines if SEPARATOR_RE.search(line))
        caption_anchor_count = len(CAPTION_ANCHOR_RE.findall(text))
        table_anchor_count = len(TABLE_ANCHOR_RE.findall(text))
        figure_anchor_count = len(FIGURE_ANCHOR_RE.findall(text))
        numeric_density = numeric_tokens / token_count if token_count else 0.0
        short_line_ratio = short_lines / len(lines) if lines else 0.0
        avg_tokens_per_line = token_count / len(lines) if lines else 0.0
        separator_pattern_score = separator_lines / len(lines) if lines else 0.0
        text_chars = len(text)
        table_like_score = 0.0
        if table_anchor_count > 0:
            table_like_score += 2.5
        if numeric_density >= 0.18:
            table_like_score += 1.0
        if short_line_ratio >= 0.45:
            table_like_score += 1.0
        if separator_pattern_score >= 0.12:
            table_like_score += 0.75
        if avg_tokens_per_line <= 7 and len(lines) >= 8:
            table_like_score += 0.5
        figure_like_score = 0.0
        if figure_anchor_count > 0:
            figure_like_score += 2.5
        if image_count > 0:
            figure_like_score += 1.0
        if figure_anchor_count > 0 and text_chars <= 1600:
            figure_like_score += 0.75
        if figure_anchor_count > 0 and numeric_density < 0.15:
            figure_like_score += 0.5
        scanned_like_score = 0.0
        if text_chars < SCANNED_TEXT_CHAR_THRESHOLD:
            scanned_like_score += 1.5
        if text_chars < 20:
            scanned_like_score += 1.0
        if image_count > 0 and text_chars < 160:
            scanned_like_score += 0.5
        return PageSignals(
            caption_anchor_count=caption_anchor_count,
            table_anchor_count=table_anchor_count,
            figure_anchor_count=figure_anchor_count,
            numeric_density=round(numeric_density, 4),
            short_line_ratio=round(short_line_ratio, 4),
            avg_tokens_per_line=round(avg_tokens_per_line, 4),
            separator_pattern_score=round(separator_pattern_score, 4),
            text_chars=text_chars,
            image_object_count=image_count,
            table_like_score=round(table_like_score, 2),
            figure_like_score=round(figure_like_score, 2),
            scanned_like_score=round(scanned_like_score, 2),
        )

    def _max_hi_res_pages_per_document(self) -> int:
        if self.settings is None:
            return MAX_HI_RES_PAGES_PER_DOC
        return max(1, int(self.settings.pdf_hi_res_max_pages_per_document))

    @staticmethod
    def _group_contiguous_pages(pages: set[int]) -> list[tuple[int, int]]:
        ordered = sorted(page for page in pages if page > 0)
        if not ordered:
            return []
        ranges: list[tuple[int, int]] = []
        start = ordered[0]
        end = ordered[0]
        for page in ordered[1:]:
            if page == end + 1:
                end = page
                continue
            ranges.append((start, end))
            start = page
            end = page
        ranges.append((start, end))
        return ranges

    @staticmethod
    def _count_page_images(page: object) -> int:
        images = getattr(page, "images", None)
        if images is None:
            return 0
        try:
            return len(images)
        except Exception:  # noqa: BLE001
            try:
                return sum(1 for _ in images)
            except Exception:  # noqa: BLE001
                return 0

    @staticmethod
    def _is_caption(category: str, text: str) -> bool:
        if not text:
            return False
        return category == "FigureCaption" or bool(CAPTION_ANCHOR_RE.match(text))

    @staticmethod
    def _match_nearest_caption(block: ExtractedBlock, captions: list[ExtractedBlock]) -> ExtractedBlock | None:
        if block.bbox is None:
            return None
        best_match: ExtractedBlock | None = None
        best_distance = float("inf")
        block_center_x = (block.bbox[0] + block.bbox[2]) / 2
        block_top = block.bbox[1]
        block_bottom = block.bbox[3]
        for caption in captions:
            if caption.page != block.page or caption.bbox is None:
                continue
            caption_center_x = (caption.bbox[0] + caption.bbox[2]) / 2
            caption_top = caption.bbox[1]
            caption_bottom = caption.bbox[3]
            vertical_gap = min(abs(block_top - caption_bottom), abs(caption_top - block_bottom))
            horizontal_gap = abs(block_center_x - caption_center_x)
            distance = vertical_gap + horizontal_gap * 0.1
            if distance < best_distance:
                best_distance = distance
                best_match = caption
        return best_match

    @staticmethod
    def _normalize_table_text(text: str, text_as_html: object) -> str:
        html = str(text_as_html or "").strip()
        if html:
            normalized = html
            normalized = re.sub(r"(?i)</tr>", "\n", normalized)
            normalized = re.sub(r"(?i)</t[dh]>", " | ", normalized)
            normalized = re.sub(r"(?i)<br\s*/?>", " ", normalized)
            normalized = HTML_TAG_RE.sub(" ", normalized)
            normalized = re.sub(r"\s*\|\s*(\|\s*)+", " | ", normalized)
            normalized = re.sub(r"[ \t]+", " ", normalized)
            normalized = re.sub(r"\n{2,}", "\n", normalized)
            normalized = normalized.replace(" | \n", "\n").strip(" |\n")
            if normalized:
                return normalized
        return text.strip()

    @staticmethod
    def _looks_like_noisy_ocr(text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return True
        tokens = cleaned.split()
        if len(tokens) < 6:
            return True
        alpha_chars = sum(1 for char in cleaned if char.isalpha())
        if alpha_chars == 0:
            return True
        long_tokens = sum(1 for token in tokens if len(token) >= 4)
        return long_tokens / max(len(tokens), 1) < 0.2

    @staticmethod
    def _fallback_page_text(blocks: list[ExtractedBlock]) -> str:
        parts: list[str] = []
        for block in blocks:
            if block.block_type == "caption":
                parts.append(block.text)
            else:
                if block.caption:
                    parts.append(block.caption)
                if block.text:
                    parts.append(block.text)
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _extract_bbox(metadata: object) -> tuple[float, float, float, float] | None:
        if metadata is None:
            return None
        coordinates = getattr(metadata, "coordinates", None)
        if isinstance(metadata, dict):
            coordinates = coordinates or metadata.get("coordinates")
        if coordinates is None:
            return None
        points = getattr(coordinates, "points", None)
        if isinstance(coordinates, dict):
            points = points or coordinates.get("points")
        if not points:
            return None
        try:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
        except (TypeError, ValueError):
            return None
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _extract_page_number(metadata: object) -> int:
        if metadata is None:
            return 1
        page_number = getattr(metadata, "page_number", None)
        if isinstance(metadata, dict):
            page_number = page_number or metadata.get("page_number")
        if isinstance(page_number, int) and page_number > 0:
            return page_number
        return 1
