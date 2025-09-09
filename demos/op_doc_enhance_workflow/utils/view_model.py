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
    """Render metadata as a structured HTML table with nested dict support."""
    if not meta:
        return ""
    
    def _render_value(value: Any, indent_level: int = 0) -> str:
        """Recursively render values, with special handling for nested dicts."""
        indent_style = f"padding-left: {indent_level * 20}px;" if indent_level > 0 else ""
        
        if isinstance(value, dict) and value:
            # 如果是字典，创建嵌套的表格结构
            nested_rows = []
            for k, v in value.items():
                nested_value = _render_value(v, indent_level + 1)
                nested_rows.append(
                    f"<tr>"
                    f"<td style='padding:2px 8px; color:#777; white-space:nowrap; {indent_style}'>{escape(str(k))}</td>"
                    f"<td style='padding:2px 8px; {indent_style}'>{nested_value}</td>"
                    f"</tr>"
                )
            return "".join(nested_rows)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            # 如果是字典列表，直接展示每个字典的内容，不显示索引
            list_items = []
            for item in value:
                if isinstance(item, dict):
                    item_content = _render_value(item, indent_level + 1)
                    list_items.append(item_content)
                else:
                    list_items.append(
                        f"<tr>"
                        f"<td style='padding:2px 8px; color:#777; white-space:nowrap; {indent_style}'>-</td>"
                        f"<td style='padding:2px 8px; {indent_style}'>{escape(str(item))}</td>"
                        f"</tr>"
                    )
            return "".join(list_items)
        else:
            # 普通值直接显示
            return escape(str(value))
    
    # 构建主表格
    rows = []
    for k, v in meta.items():
        if isinstance(v, dict) and v:
            # 字典类型：先显示键名，然后显示嵌套内容
            rows.append(
                f"<tr>"
                f"<td style='padding:4px 8px; color:#555; white-space:nowrap; font-weight:bold;' colspan='2'>{escape(str(k))}</td>"
                f"</tr>"
            )
            nested_content = _render_value(v, 1)
            rows.append(nested_content)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            # 字典列表类型
            rows.append(
                f"<tr>"
                f"<td style='padding:4px 8px; color:#555; white-space:nowrap; font-weight:bold;' colspan='2'>{escape(str(k))}</td>"
                f"</tr>"
            )
            list_content = _render_value(v, 1)
            rows.append(list_content)
        else:
            # 普通键值对
            rows.append(
                f"<tr>"
                f"<td style='padding:4px 8px; color:#555; white-space:nowrap;'>{escape(str(k))}</td>"
                f"<td style='padding:4px 8px;'>{escape(str(v))}</td>"
                f"</tr>"
            )
    
    return (
        "<div class='meta' style='margin-top:6px;'>"
        f"<table style='border-collapse:collapse; margin-top:6px;'>{''.join(rows)}</table>"
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