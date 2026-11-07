# context/context.py
"""Build context/context.md and a small context/context.json statistics file.

    python context/context.py
    python context/context.py --check

No third-party dependencies. Scans relative to this file's parent repository.
"""

import argparse
import codecs
import json
import math
import os
import re
from collections import Counter
from pathlib import Path


SKIP_DIRECTORIES = {
    "context", "results", ".git", ".venv", "venv", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", "build", "dist", "site-packages",
}
SKIP_FILES = {"context.yml", "context.yaml"}
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
    ".csv", ".tsv", ".parquet", ".arrow", ".tex", ".ipynb",
    ".npy", ".npz", ".pkl", ".pickle", ".pt", ".pth", ".ckpt",
    ".safetensors", ".bin", ".exe", ".dll", ".so", ".dylib",
    ".zip", ".gz", ".tar", ".7z", ".mp4", ".mov", ".mp3",
    ".wav", ".db", ".sqlite", ".sqlite3", ".log", ".lock",
    ".pem", ".key", ".p12", ".pfx", ".crt", ".cer",
}
LANGUAGES = {
    ".py": "python", ".pyi": "python", ".md": "markdown", ".mdx": "mdx",
    ".rst": "rst", ".txt": "text", ".toml": "toml", ".yml": "yaml",
    ".yaml": "yaml", ".json": "json", ".json5": "json5",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp",
    ".cxx": "cpp", ".hpp": "cpp", ".hxx": "cpp", ".cu": "cuda",
    ".cuh": "cuda", ".html": "html", ".htm": "html", ".css": "css",
    ".js": "javascript", ".jsx": "jsx", ".ts": "typescript",
    ".tsx": "tsx", ".m": "matlab", ".jl": "julia", ".r": "r",
    ".rs": "rust", ".go": "go", ".java": "java", ".kt": "kotlin",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".fish": "fish",
    ".sql": "sql", ".xml": "xml", ".svg": "xml", ".ini": "ini",
    ".cfg": "ini", ".properties": "properties", ".f": "fortran",
    ".for": "fortran", ".f90": "fortran", ".f95": "fortran",
    ".f03": "fortran", ".f08": "fortran", ".lua": "lua",
    ".rb": "ruby", ".php": "php", ".swift": "swift",
}
def readable_text(path: Path) -> bool:
    """Accept UTF-8 text, rejecting binary data even for unfamiliar extensions."""
    try:
        with path.open("rb") as stream:
            sample = stream.read(8192)
        if b"\x00" in sample:
            return False
        codecs.getincrementaldecoder("utf-8-sig")().decode(sample, final=False)
        # Reject binary-like control bytes, but allow tabs/newlines/carriage returns.
        return not any(b < 32 and b not in (9, 10, 13) for b in sample)
    except (OSError, UnicodeError):
        return False


def included_files(root: Path) -> list[Path]:
    paths = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            d for d in dirs
            if d not in SKIP_DIRECTORIES
            and not d.startswith("_")
            and (not d.startswith(".") or d == ".github")
            and not (Path(directory) / d).is_symlink()
        )
        for name in files:
            path = Path(directory) / name
            suffix = path.suffix.lower()
            if (name.lower() in SKIP_FILES or name.startswith(".")
                    or suffix in SKIP_EXTENSIONS or path.is_symlink()
                    or not path.is_file()):
                continue
            # Include all other UTF-8 text files, even if their extension is new.
            if readable_text(path):
                paths.append(path.relative_to(root))
    return sorted(paths, key=lambda path: path.as_posix())


def code_block(content: str, suffix: str) -> str:
    tag = LANGUAGES.get(suffix.lower(), "text")
    longest = max((len(match.group()) for match in re.finditer(r"`+", content)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{tag}\n{content.rstrip()}\n{fence}\n"


def generate(root: Path, paths: list[Path]) -> str:
    parts = [
        "# Repository Context\n\n",
        "Generated automatically from repository source files. Do not edit manually.\n\n",
        "## Files\n\n",
        *(f"- `{path.as_posix()}`\n" for path in paths),
    ]
    for path in paths:
        source = (root / path).read_text(encoding="utf-8-sig")
        parts.extend((
            f"\n## `{path.as_posix()}`\n\n",
            code_block(source, path.suffix),
        ))
    return "".join(parts).rstrip() + "\n"


def statistics(root: Path, paths: list[Path], context: str) -> str:
    """Minimal, deterministic summary; token counts are rough estimates."""
    sources = [(root / path).read_text(encoding="utf-8-sig") for path in paths]
    source_bytes = sum(len(source.encode("utf-8")) for source in sources)
    context_bytes = len(context.encode("utf-8"))
    data = {
        "file_count": len(paths),
        "source_lines": sum(len(source.splitlines()) for source in sources),
        "source_bytes": source_bytes,
        "source_tokens_estimate": math.ceil(source_bytes / 3.5),
        "context_md_lines": len(context.splitlines()),
        "context_md_bytes": context_bytes,
        "context_md_tokens_estimate": math.ceil(context_bytes / 3.5),
        "token_estimate_method": "UTF-8 bytes / 3.5; approximate, not a tokenizer",
        "files_by_extension": dict(sorted(Counter(
            path.suffix.lower() or "[none]" for path in paths
        ).items())),
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--check", action="store_true", help="Exit 1 if context is outdated.")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")
    output = root / "context" / "context.md"
    summary_output = root / "context" / "context.json"
    paths = included_files(root)
    content = generate(root, paths)
    summary = statistics(root, paths, content)

    if args.check:
        if (output.is_file() and summary_output.is_file()
                and output.read_text(encoding="utf-8") == content
                and summary_output.read_text(encoding="utf-8") == summary):
            print(f"Up to date: {output} and {summary_output} ({len(paths)} files)")
            return 0
        print("Out of date: context.md or context.json")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    summary_output.write_text(summary, encoding="utf-8")
    print(f"Updated {output} and {summary_output} ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
