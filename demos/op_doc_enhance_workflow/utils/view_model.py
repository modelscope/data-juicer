import os
from html import escape
from typing import Any, Dict, List

from .example_ir import ExampleIR


def _paths(assets) -> List[str]:
    """Extract relative paths from asset objects."""
    return [m.rel for m in assets]


def _files_names(assets) -> List[str]:
    """Extract file names from asset objects."""
    return [os.path.basename(m.rel) for m in assets]


def _render_text_block(text: str) -> str:
    """Render text content in a styled pre block."""
    return f'<pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">{escape(text)}</pre>'


def _render_images(paths: List[str]) -> str:
    """Render image paths as HTML img elements in a grid layout."""
    items = "".join(f'<img src="{escape(p)}" width="160" style="margin:4px;"/>' for p in paths)
    return f'<div class="image-grid">{items}</div>'


def _render_videos(paths: List[str]) -> str:
    """Render video paths as HTML video elements with controls."""
    items = "".join(f'<video src="{escape(p)}" controls width="320" style="margin:4px;"></video>' for p in paths)
    return f'<div class="video-grid">{items}</div>'


def _render_audios(paths: List[str]) -> str:
    """Render audio paths as HTML audio elements with controls."""
    items = "".join(f'<audio src="{escape(p)}" controls style="display:block; margin:4px 0;"></audio>' for p in paths)
    return f'<div class="audio-list">{items}</div>'


def _shorten(val: Any, max_len: int = 120) -> str:
    """Truncate string representation of value to max length."""
    s = str(val)
    return (s[:max_len] + "...") if len(s) > max_len else s


def _render_meta(meta: Dict[str, Any]) -> str:
    """Render metadata as collapsible HTML table."""
    if not meta:
        return ""
    rows = "".join(
        f"<tr><td style='padding:4px 8px; color:#555;'>{escape(str(k))}</td>"
        f"<td style='padding:4px 8px;'>{escape(_shorten(v))}</td></tr>"
        for k, v in meta.items()
    )
    table = (
        "<details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary>"
        f"<table style='border-collapse:collapse; margin-top:6px;'>{rows}</table></details>"
    )
    return table


def _render_sample_header(sample_idx: int, sample: Dict[str, Any]) -> str:
    """Render sample header with index and content summary."""
    content_types = []
    if sample.get("text"):
        content_types.append("text")
    if sample.get("answer"):
        content_types.append("answer")
    if sample.get("images"):
        count = len(sample["images"])
        content_types.append(f"{count} image{'s' if count != 1 else ''}")
    if sample.get("videos"):
        count = len(sample["videos"])
        content_types.append(f"{count} video{'s' if count != 1 else ''}")
    if sample.get("audios"):
        count = len(sample["audios"])
        content_types.append(f"{count} audio{'s' if count != 1 else ''}")

    summary = " | ".join(content_types) if content_types else "empty"
    return f'<div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample {sample_idx + 1}:</strong> {summary}</div>'


def _render_sample_card(sample: Dict[str, Any], sample_idx: int) -> str:
    """Render individual sample as HTML card with header and clear content organization."""
    parts: List[str] = []

    # Add sample header
    parts.append(_render_sample_header(sample_idx, sample))

    # Handle Q&A format
    if sample.get("text") and sample.get("answer"):
        q = escape(sample.get("text", "") or "")
        a = escape(sample.get("answer", "") or "")
        parts.append(
            f'<div class="qa" style="margin-bottom:6px;"><div><strong>Q:</strong> {q}</div><div><strong>A:</strong> {a}</div></div>'
        )
    # Handle text-only content
    elif sample.get("text"):
        parts.append(_render_text_block(str(sample["text"])))

    # Handle media content with counts and labels
    if sample.get("images"):
        parts.append(
            f'<div class="media-section" style="margin-bottom:8px;">'
            f'<div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">{"|".join(_files_names(sample["images"]))}:</div>'
            f'{_render_images(_paths(sample["images"]))}</div>'
        )

    if sample.get("videos"):
        parts.append(
            f'<div class="media-section" style="margin-bottom:8px;">'
            f'<div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">{"|".join(_files_names(sample["videos"]))}:</div>'
            f'{_render_videos(_paths(sample["videos"]))}</div>'
        )

    if sample.get("audios"):
        parts.append(
            f'<div class="media-section" style="margin-bottom:8px;">'
            f'<div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">{"|".join(_files_names(sample["audios"]))}:</div>'
            f'{_render_audios(_paths(sample["audios"]))}</div>'
        )

    # Handle metadata
    if sample.get("meta"):
        parts.append(_render_meta(sample["meta"]))

    inner = "".join(parts) if parts else "<div style='color:#999; font-style:italic;'>(empty sample)</div>"
    return f'<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);">{inner}</div>'


def _render_payload(payload: Dict[str, Any]) -> str:
    """Render payload samples with enhanced sample-aware layout."""
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        return ""

    # Always use sample cards to maintain sample boundaries
    return "".join(_render_sample_card(s, i) for i, s in enumerate(samples))


def to_legacy_view(ex: ExampleIR) -> Dict:
    """Convert ExampleIR to legacy view format with rendered HTML content."""
    return {
        "input": _render_payload(ex.input or {}),
        "output": _render_payload(ex.output or {}),
        # "explanation": "",
    }
