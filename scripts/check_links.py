#!/usr/bin/env python3
"""Check that relative links in Markdown files point to existing files."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def main() -> int:
    bad = []
    for md in ROOT.rglob("*.md"):
        if any(p in {"build", ".git", "install"} for p in md.relative_to(ROOT).parts):
            continue
        for target in LINK.findall(md.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = target.split("#", 1)[0]
            if path and not (md.parent / path).exists():
                bad.append(f"{md.relative_to(ROOT)}: broken link {target}")
    for line in bad:
        print(line, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
