"""Convert PDF files to clean Markdown text."""

from __future__ import annotations

import glob
from pathlib import Path

import pymupdf
import pymupdf4llm

from docmind.log import logger


def pdf_to_markdown(
    pdf_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Convert a single PDF to a Markdown file.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    output_dir:
        Directory where the ``.md`` file will be written.

    Returns
    -------
    Path
        Path of the generated Markdown file.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    ValueError
        If *pdf_path* is not a PDF file.
    """
    src = Path(pdf_path)
    if not src.exists():
        raise FileNotFoundError(f"PDF not found: {src}")
    if src.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got '{src.suffix}'")

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(src))
    try:
        raw_md: str = pymupdf4llm.to_markdown(
            doc,
            header=False,
            footer=False,
            page_separators=True,
            ignore_images=True,
            write_images=False,
            image_path=None,
        )
    finally:
        doc.close()

    clean_md = raw_md.encode("utf-8", errors="surrogatepass").decode(
        "utf-8", errors="ignore"
    )

    out_path = (dest_dir / src.stem).with_suffix(".md")
    out_path.write_bytes(clean_md.encode("utf-8"))

    logger.info("Converted {} → {}", src.name, out_path.name)
    return out_path


def batch_convert(
    pattern: str,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Convert every PDF matching *pattern* to Markdown.

    Parameters
    ----------
    pattern:
        A glob pattern (e.g. ``"docs/*.pdf"``).
    output_dir:
        Target directory for Markdown files.
    overwrite:
        When ``False`` (default), skip files that already
        have a corresponding ``.md`` in *output_dir*.

    Returns
    -------
    list[Path]
        Paths of all generated (or already existing) Markdown
        files.
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []

    for pdf_str in sorted(glob.glob(pattern)):
        pdf = Path(pdf_str)
        md_path = (dest / pdf.stem).with_suffix(".md")
        if md_path.exists() and not overwrite:
            logger.debug("Skipping {} (already converted)", pdf.name)
            results.append(md_path)
            continue
        results.append(pdf_to_markdown(pdf, dest))

    return results
