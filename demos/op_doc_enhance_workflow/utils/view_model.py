import os
from html import escape
from typing import Any, Dict, List

from .example_ir import ExampleIR

TEXT_COLLAPSE_THRESHOLD = 300
LIST_COLLAPSE_THRESHOLD = 600
IMAGES_INLINE_LIMIT = 6
VIDEOS_INLINE_LIMIT = 1
AUDIOS_INLINE_LIMIT = 3


def _paths(assets) -> List[str]:
    """Extract relative paths from asset objects."""
    return [m.rel for m in assets]


def _files_names(assets) -> List[str]:
    """Extract file names from asset objects."""
    return [os.path.basename(m.rel) for m in assets]


def _files_label(names: List[str], limit: int) -> str:
    """Make a compact label for file names with '+N more' when too many."""
    if len(names) <= limit:
        return "|".join(names)
    head = "|".join(names[:limit])
    rest = len(names) - limit
    return f"{head} +{rest} more"


def _collapsible(summary: str, inner_html: str, open_default: bool = False) -> str:
    """Wrap inner_html into an HTML details/summary block."""
    open_attr = " open" if open_default else ""
    return (
        f"<details{open_attr} style='margin:6px 0;'>"
        f"<summary style='cursor:pointer; color:#0366d6;'>{escape(summary)}</summary>"
        f"{inner_html}</details>"
    )


def _pre_block(text: str) -> str:
    """Base pre block with consistent style."""
    return (
        '<pre style="padding:6px; background:#f6f8fa; border-radius:4px; '
        'overflow-x:auto; white-space:pre; word-wrap:normal;">'
        f"{escape(text)}</pre>"
    )


def _render_text_block(text: str, collapse_threshold: int = None) -> str:
    """Render text; collapse when exceeds threshold."""
    if text is None:
        return ""
    text = str(text)
    if collapse_threshold and len(text) > collapse_threshold:
        preview = _pre_block(text[:collapse_threshold] + "...")
        full = _pre_block(text)
        return preview + _collapsible(f"Show more 展开更多 ({len(text) - collapse_threshold} more chars)", full)
    return _pre_block(text)


def _render_images(paths: List[str]) -> str:
    """Render images with inline limit; collapse the rest."""
    if not paths:
        return ""
    if len(paths) <= IMAGES_INLINE_LIMIT:
        items = "".join(f'<img src="{escape(p)}" width="160" style="margin:4px;"/>' for p in paths)
        return f'<div class="image-grid">{items}</div>'
    # split
    head = paths[:IMAGES_INLINE_LIMIT]
    tail = paths[IMAGES_INLINE_LIMIT:]
    head_items = "".join(f'<img src="{escape(p)}" width="160" style="margin:4px;"/>' for p in head)
    tail_items = "".join(f'<img src="{escape(p)}" width="160" style="margin:4px;"/>' for p in tail)
    return f'<div class="image-grid">{head_items}</div>' + _collapsible(
        f"Show {len(tail)} more images 展开更多图片", f'<div class="image-grid">{tail_items}</div>'
    )


def _render_videos(paths: List[str]) -> str:
    """Render videos with inline limit; collapse the rest."""
    if not paths:
        return ""
    if len(paths) <= VIDEOS_INLINE_LIMIT:
        items = "".join(f'<video src="{escape(p)}" controls width="320" style="margin:4px;"></video>' for p in paths)
        return f'<div class="video-grid">{items}</div>'
    head = paths[:VIDEOS_INLINE_LIMIT]
    tail = paths[VIDEOS_INLINE_LIMIT:]
    head_items = "".join(f'<video src="{escape(p)}" controls width="320" style="margin:4px;"></video>' for p in head)
    tail_items = "".join(f'<video src="{escape(p)}" controls width="320" style="margin:4px;"></video>' for p in tail)
    return f'<div class="video-grid">{head_items}</div>' + _collapsible(
        f"Show {len(tail)} more videos 展开更多视频", f'<div class="video-grid">{tail_items}</div>'
    )


def _render_audios(paths: List[str]) -> str:
    """Render audios with inline limit; collapse the rest."""
    if not paths:
        return ""
    if len(paths) <= AUDIOS_INLINE_LIMIT:
        items = "".join(
            f'<audio src="{escape(p)}" controls style="display:block; margin:4px 0;"></audio>' for p in paths
        )
        return f'<div class="audio-list">{items}</div>'
    head = paths[:AUDIOS_INLINE_LIMIT]
    tail = paths[AUDIOS_INLINE_LIMIT:]
    head_items = "".join(
        f'<audio src="{escape(p)}" controls style="display:block; margin:4px 0;"></audio>' for p in head
    )
    tail_items = "".join(
        f'<audio src="{escape(p)}" controls style="display:block; margin:4px 0;"></audio>' for p in tail
    )
    return f'<div class="audio-list">{head_items}</div>' + _collapsible(
        f"Show {len(tail)} more audios 展开更多音频", f'<div class="audio-list">{tail_items}</div>'
    )


def _render_meta(meta: Dict[str, Any]) -> str:
    """Render metadata as an HTML table with group-level zebra backgrounds.
    All rows inside a group share the same background.
    Apply styles on TD with !important to survive Sphinx theme overrides."""

    if not meta:
        return ""

    zebra_light = "#f8f9fa"
    zebra_white = "#ffffff"
    border_color = "#eaecef"  # subtle border color

    def indent_style(level: int) -> str:
        return f"padding-left: {level * 16}px;" if level > 0 else ""

    def table_style() -> str:
        # Force a border around each table to keep visible separation under themes
        return f"border-collapse:collapse; width:100%; border:1px solid {border_color} !important;"

    def td_style(
        group_bg: str, level: int = 0, nowrap: bool = False, padding: str = "4px 8px", bold: bool = False
    ) -> str:
        # Put background and borders on TD to avoid being overridden by theme table/tr rules
        parts = [
            "text-align:left;",
            "vertical-align:top;",
            f"padding:{padding};",
            f"background-color:{group_bg} !important;" if group_bg else "",
            f"border-bottom:1px solid {border_color} !important;",
        ]
        if nowrap:
            parts.append("white-space:nowrap;")
        if bold:
            parts.append("font-weight:bold; color:#555;")
        if level > 0:
            parts.append(indent_style(level))
        return " ".join(p for p in parts if p)

    def should_expand_list(lst: list) -> bool:
        """Expand a child list as a titled block if:
        - length > 3 and (all items are dicts OR total string length > 20)."""
        if len(lst) <= 3:
            return False
        all_dicts = all(isinstance(x, dict) for x in lst)
        total_len = sum(len(str(x)) for x in lst)
        return all_dicts or total_len > 20

    def render_list_items(lst: list, level: int, group_bg: str) -> str:
        """Render list as one-item-per-row; dict items render as nested dict tables; all TDs carry same bg."""
        rows = []
        for item in lst:
            if isinstance(item, dict) and item:
                nested_rows = render_dict_as_rows(item, level + 1, group_bg)
                content = f"<table class='meta-table' style='{table_style()}'>{nested_rows}</table>"
            else:
                content = escape(str(item))
            rows.append(f"<tr>" f"<td colspan='2' style='{td_style(group_bg, level)}'>{content}</td>" f"</tr>")
        return "".join(rows)

    def render_dict_as_rows(d: Dict[str, Any], level: int, group_bg: str) -> str:
        """Render dict as key-value rows; all TDs carry same bg."""
        rows = []
        for ck, cv in d.items():
            # Child list
            if isinstance(cv, list) and cv:
                if should_expand_list(cv):
                    # Sub-title row
                    rows.append(
                        f"<tr>"
                        f"<td colspan='2' style='{td_style(group_bg, level, bold=True)}'>{escape(str(ck))}</td>"
                        f"</tr>"
                    )
                    rows.append(render_list_items(cv, level + 1, group_bg))
                else:
                    # Render list as plain string
                    rows.append(
                        f"<tr>"
                        f"<td style='{td_style(group_bg, level, nowrap=True)}'>{escape(str(ck))}</td>"
                        f"<td style='{td_style(group_bg)}'>{escape(str(cv))}</td>"
                        f"</tr>"
                    )
                continue

            # Child dict: nested table
            if isinstance(cv, dict) and cv:
                nested_rows = render_dict_as_rows(cv, level + 1, group_bg)
                rows.append(
                    f"<tr>"
                    f"<td style='{td_style(group_bg, level, nowrap=True)}'>{escape(str(ck))}</td>"
                    f"<td style='{td_style(group_bg)}'>"
                    f"<table class='meta-table' style='{table_style()}'>{nested_rows}</table>"
                    f"</td>"
                    f"</tr>"
                )
                continue

            # Scalars or empty containers
            rows.append(
                f"<tr>"
                f"<td style='{td_style(group_bg, level, nowrap=True)}'>{escape(str(ck))}</td>"
                f"<td style='{td_style(group_bg)}'>{escape(str(cv))}</td>"
                f"</tr>"
            )
        return "".join(rows)

    # Build top-level table
    out_rows = []
    group_idx = 0  # zebra only across top-level dict/list groups

    for k, v in meta.items():
        if isinstance(v, dict) and v:
            group_bg = zebra_light if group_idx % 2 == 0 else zebra_white
            group_idx += 1
            out_rows.append(
                f"<tr>"
                f"<td colspan='2' style='{td_style(group_bg, bold=True, padding='6px 8px')}'>{escape(str(k))}</td>"
                f"</tr>"
            )
            out_rows.append(render_dict_as_rows(v, level=1, group_bg=group_bg))

        elif isinstance(v, list):
            group_bg = zebra_light if group_idx % 2 == 0 else zebra_white
            group_idx += 1
            out_rows.append(
                f"<tr>"
                f"<td colspan='2' style='{td_style(group_bg, bold=True, padding='6px 8px')}'>{escape(str(k))}</td>"
                f"</tr>"
            )
            out_rows.append(render_list_items(v, level=1, group_bg=group_bg))

        else:
            # Non-group simple row (no zebra background)
            out_rows.append(
                f"<tr>"
                f"<td style='text-align:left; vertical-align:top; padding:4px 8px; white-space:nowrap; border-bottom:1px solid {border_color} !important;'>{escape(str(k))}</td>"
                f"<td style='text-align:left; vertical-align:top; padding:4px 8px; border-bottom:1px solid {border_color} !important;'>{escape(str(v))}</td>"
                f"</tr>"
            )

    return (
        "<div class='meta' style='margin-top:6px;'>"
        f"<table class='meta-table' style='{table_style()}'>{''.join(out_rows)}</table>"
        "</div>"
    )


def _render_sample_header(sample_idx: int, sample: Dict[str, Any]) -> str:
    """Render sample header with index and content summary."""
    content_types = []
    if sample.get("text"):
        content_types.append("text")
    if sample.get("answer"):
        content_types.append("answer")
    if sample.get("list"):
        content_types.append("list")
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
    return (
        f'<div class="sample-header" '
        f'style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; '
        f'font-size:0.9em; color:#666; border-left:3px solid #007acc;">'
        f"<strong>Sample {sample_idx + 1}:</strong> {summary}</div>"
    )


def _render_sample_card(sample: Dict[str, Any], sample_idx: int) -> str:
    """Render individual sample as HTML card with header and clear content organization."""
    parts: List[str] = []

    # Add sample header
    parts.append(_render_sample_header(sample_idx, sample))

    # Handle Q&A format
    if sample.get("text") and sample.get("answer"):
        q = str(sample.get("text", "") or "")
        a = str(sample.get("answer", "") or "")
        q_block = _render_text_block(q, collapse_threshold=TEXT_COLLAPSE_THRESHOLD)
        a_block = _render_text_block(a, collapse_threshold=TEXT_COLLAPSE_THRESHOLD)
        parts.append(
            '<div class="qa" style="margin-bottom:6px;">'
            f"<div><strong>Q:</strong> {q_block}</div>"
            f"<div><strong>A:</strong> {a_block}</div>"
            "</div>"
        )
    # Handle text-only content
    elif sample.get("text"):
        parts.append(_render_text_block(str(sample["text"]), collapse_threshold=TEXT_COLLAPSE_THRESHOLD))

    # Handle list-only content
    if sample.get("list"):
        parts.append(_render_text_block(str(sample["list"]), collapse_threshold=LIST_COLLAPSE_THRESHOLD))

    # Handle media content with counts and labels
    if sample.get("images"):
        names = _files_names(sample["images"])
        label = _files_label(names, IMAGES_INLINE_LIMIT)
        parts.append(
            f'<div class="media-section" style="margin-bottom:8px;">'
            f'<div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">{escape(label)}:</div>'
            f'{_render_images(_paths(sample["images"]))}</div>'
        )

    if sample.get("videos"):
        names = _files_names(sample["videos"])
        label = _files_label(names, VIDEOS_INLINE_LIMIT)
        parts.append(
            f'<div class="media-section" style="margin-bottom:8px;">'
            f'<div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">{escape(label)}:</div>'
            f'{_render_videos(_paths(sample["videos"]))}</div>'
        )

    if sample.get("audios"):
        names = _files_names(sample["audios"])
        label = _files_label(names, AUDIOS_INLINE_LIMIT)
        parts.append(
            f'<div class="media-section" style="margin-bottom:8px;">'
            f'<div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">{escape(label)}:</div>'
            f'{_render_audios(_paths(sample["audios"]))}</div>'
        )

    # Handle metadata
    if sample.get("meta"):
        parts.append(_render_meta(sample["meta"]))

    inner = "".join(parts) if parts else "<div style='color:#999; font-style:italic;'>(empty sample)</div>"
    return (
        f'<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; '
        f'background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);">{inner}</div>'
    )


def _render_payload(payload: Dict[str, Any]) -> str:
    """Render payload samples with enhanced sample-aware layout."""
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        return ""
    return "".join(_render_sample_card(s, i) for i, s in enumerate(samples))


def to_legacy_view(ex: ExampleIR) -> Dict:
    """Convert ExampleIR to legacy view format with rendered HTML content."""
    return {
        "input": _render_payload(ex.input or {}),
        "output": _render_payload(ex.output or {}),
    }
