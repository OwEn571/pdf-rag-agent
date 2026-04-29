from __future__ import annotations

from pathlib import Path

from app.services.pdf_rendering import render_pdf_page_image_data_url, subprocess_command_allowed


def test_pdf_rendering_allowlist_rejects_paths_and_unknown_commands() -> None:
    assert subprocess_command_allowed(["pdftoppm", "-png"])
    assert not subprocess_command_allowed(["/usr/bin/pdftoppm", "-png"])
    assert not subprocess_command_allowed(["python", "-c", "print(1)"])
    assert not subprocess_command_allowed([])


def test_render_pdf_page_image_data_url_uses_cache_and_command(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    cache: dict[tuple[str, int], str] = {}
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append({"command": command, **kwargs})
        output_prefix = Path(command[-1])
        output_prefix.with_suffix(".png").write_bytes(b"fake-image")
        return object()

    data_url = render_pdf_page_image_data_url(
        file_path=str(pdf_path),
        page=2,
        pdf_render_dpi=50,
        timeout_seconds=1.0,
        cache=cache,
        run_command=fake_run,
    )
    cached = render_pdf_page_image_data_url(
        file_path=str(pdf_path),
        page=2,
        pdf_render_dpi=300,
        timeout_seconds=10.0,
        cache=cache,
        run_command=fake_run,
    )

    assert data_url == "data:image/png;base64,ZmFrZS1pbWFnZQ=="
    assert cached == data_url
    assert len(calls) == 1
    assert calls[0]["command"][:8] == ["pdftoppm", "-f", "2", "-l", "2", "-singlefile", "-png", "-r"]
    assert calls[0]["command"][8] == "72"
    assert calls[0]["timeout"] == 5.0


def test_render_pdf_page_image_data_url_rejects_missing_or_invalid_input(tmp_path: Path) -> None:
    cache: dict[tuple[str, int], str] = {}

    assert render_pdf_page_image_data_url(
        file_path="",
        page=1,
        pdf_render_dpi=144,
        timeout_seconds=5.0,
        cache=cache,
    ) == ""
    assert render_pdf_page_image_data_url(
        file_path=str(tmp_path / "missing.pdf"),
        page=1,
        pdf_render_dpi=144,
        timeout_seconds=5.0,
        cache=cache,
    ) == ""
    assert render_pdf_page_image_data_url(
        file_path=str(tmp_path),
        page=0,
        pdf_render_dpi=144,
        timeout_seconds=5.0,
        cache=cache,
    ) == ""
