from __future__ import annotations

import base64
import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable


ALLOWED_SUBPROCESS_COMMANDS = {"pdftoppm"}
SubprocessRunFn = Callable[..., object]


def subprocess_command_allowed(command: list[str]) -> bool:
    if not command:
        return False
    executable = str(command[0] or "").strip()
    if not executable:
        return False
    return executable == Path(executable).name and executable in ALLOWED_SUBPROCESS_COMMANDS


def render_pdf_page_image_data_url(
    *,
    file_path: str,
    page: int,
    pdf_render_dpi: int,
    timeout_seconds: float,
    cache: dict[tuple[str, int], str],
    run_command: SubprocessRunFn = subprocess.run,
    logger: logging.Logger | None = None,
) -> str:
    pdf_path = str(file_path or "").strip()
    if not pdf_path or page <= 0:
        return ""
    cache_key = (pdf_path, page)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    source = Path(pdf_path)
    if not source.exists():
        return ""
    try:
        with TemporaryDirectory(prefix="zprag_v4_fig_") as temp_dir:
            output_prefix = Path(temp_dir) / "page"
            command = [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "-png",
                "-r",
                str(max(72, int(pdf_render_dpi))),
                str(source),
                str(output_prefix),
            ]
            if not subprocess_command_allowed(command):
                if logger is not None:
                    logger.warning("blocked non-whitelisted subprocess command: %s", command[0] if command else "")
                return ""
            run_command(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(5.0, float(timeout_seconds)),
            )
            image_path = output_prefix.with_suffix(".png")
            if not image_path.exists():
                return ""
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            data_url = f"data:image/png;base64,{encoded}"
            cache[cache_key] = data_url
            return data_url
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warning("failed to render figure page image: file=%s page=%s err=%s", pdf_path, page, exc)
        return ""
