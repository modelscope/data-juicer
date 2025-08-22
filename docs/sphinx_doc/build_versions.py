#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
from pathlib import Path
from packaging import version as pv

# Repository structure and build configuration
REPO_ROOT = Path(__file__).resolve().parents[2]   
SITE_DIR = REPO_ROOT / "docs" / "sphinx_doc" / "build"            # Build output directory
WORKTREES_DIR = REPO_ROOT / ".worktrees"          # Temporary worktree directory for version builds
DOCS_REL = Path("docs/sphinx_doc")
LANGS = ["en", "zh_CN"]                           # Supported documentation languages
MIN_TAG = "v1.4.0"                               # Minimum version tag to build
REMOTE = "origin"                                 # Git remote name

# Build options
KEEP_WORKTREES = False     # Whether to keep worktrees after build (default: cleanup)
HAS_SUBMODULES = False     # Set True if repo uses submodules and needs initialization

def run(cmd, cwd=None, env=None, check=True):
    """Execute shell command with logging"""
    print(f"[RUN] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, cwd=cwd, env=env, check=check)

def is_valid_tag(tag: str) -> bool:
    """Check if tag matches version pattern and meets minimum version requirement"""
    if not re.match(r"^v\d+\.\d+\.\d+$", tag):
        return False
    try:
        return pv.parse(tag) >= pv.parse(MIN_TAG)
    except Exception:
        return False

def get_tags():
    """Fetch and filter valid version tags from remote repository"""
    run(["git", "fetch", "--tags", "--force", REMOTE])
    out = subprocess.check_output(["git", "tag"], text=True).strip()
    tags = [t for t in out.splitlines() if t]
    return [t for t in tags if is_valid_tag(t)]

def ensure_clean_worktree(path: Path):
    """Remove existing worktree if present to ensure clean state"""
    if path.exists():
        try:
            run(["git", "worktree", "remove", "--force", str(path)])
        except Exception:
            shutil.rmtree(path, ignore_errors=True)

def copy_docs_source_to(wt_root: Path):
    """Copy current docs source to worktree to unify templates and extensions"""
    src = REPO_ROOT / DOCS_REL
    dst = wt_root / DOCS_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[COPY] {src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", "build", ".pyc"))

def maybe_init_submodules(wt_root: Path):
    """Initialize submodules in worktree if repository uses them"""
    if HAS_SUBMODULES:
        try:
            run(["git", "submodule", "update", "--init", "--recursive"], cwd=wt_root)
        except Exception as e:
            print(f"[WARN] submodule init failed: {e}")

def build_one(ref: str, ref_label: str, available_versions: list[str]):
    """Build documentation for a single version/branch"""
    # Create and setup worktree for the specific git reference
    wt = WORKTREES_DIR / ref_label
    ensure_clean_worktree(wt)
    run(["git", "worktree", "add", "--force", str(wt), ref])
    maybe_init_submodules(wt)

    # Override docs/sphinx_doc with current repo version for unified templates
    copy_docs_source_to(wt)

    src = wt / DOCS_REL / "source"
    if not src.exists():
        print(f"[SKIP] {ref_label}: {src} not found")
        if not KEEP_WORKTREES:
            run(["git", "worktree", "remove", "--force", str(wt)])
        return

    # Build documentation for each supported language
    for lang in LANGS:
        out_dir = SITE_DIR / lang / ref_label
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup environment variables for Sphinx build
        env = os.environ.copy()
        env["DOCS_VERSION"] = ref_label              # Documentation version label (e.g., latest, v1.5.0)
        env["GIT_REF_FOR_LINKS"] = ref               # Git reference for GitHub links
        env["AVAILABLE_VERSIONS"] = ",".join(available_versions)  # All available versions for switcher
        env["REPO_ROOT"] = str(wt)                   # Version-specific repo root for copying markdown files
        env["CODE_ROOT"] = str(wt)                   # Version-specific code root for autodoc imports
        
        # Execute Sphinx build command
        cmd = [
            "sphinx-build",
            "-b", "html",                            # HTML builder
            "-D", f"language={lang}",                # Set language for this build
            "-j", "auto",
            str(src),                                # Source directory
            str(out_dir),                            # Output directory
        ]
        run(cmd, env=env)

    # Cleanup worktree after successful build
    if not KEEP_WORKTREES:
        run(["git", "worktree", "remove", "--force", str(wt)])
        try:
            run(["git", "worktree", "prune"])        # Clean up worktree references
        except Exception:
            pass

def main():
    """Main entry point: build documentation for all versions"""
    WORKTREES_DIR.mkdir(exist_ok=True)
    tags = get_tags()
    tags.sort(key=pv.parse, reverse=True)
    versions = ["main"] + tags                      # Build main branch + all valid tags

    # Build main branch first, then all tagged versions
    build_one("main", "main", versions)
    for t in tags:
        build_one(t, t, versions)

if __name__ == "__main__":
    main()