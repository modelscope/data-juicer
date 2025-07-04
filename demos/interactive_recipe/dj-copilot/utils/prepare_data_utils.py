# -*- coding: utf-8 -*-
"""
Prepare tutorial data of AgentScope
"""
import os
from typing import Optional


def prepare_docstring_txt(repo_path: Optional[str] = None, text_dir: Optional[str] = None) -> None:
    """
    If repo_path and text_dir are provided, and text_dir is empty,
    it prepares the docstring in html, and save it.
    Args:
        repo_path (`str`):
            The path of the repo
        text_dir (`str`):
            The path of the text dir
    Returns:
        None
    """
    print(f"DJ repo path: {repo_path}, text_dir: {text_dir}")
    if (
        repo_path
        and text_dir
        and os.path.exists(repo_path)
        and not os.path.exists(text_dir)
    ):
        os.system(
            f"sphinx-build -b text asset/knowledge {text_dir}"
        )