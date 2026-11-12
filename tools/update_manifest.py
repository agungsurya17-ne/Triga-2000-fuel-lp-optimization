"""Refresh the public release manifest after reviewed source changes."""

import hashlib
from pathlib import Path
import sys

# Manifest generation must not add bytecode artifacts to the release being audited.
sys.dont_write_bytecode = True
from audit_public_inputs import audit

ROOT = Path(__file__).resolve().parents[1]


def update(root=ROOT):
    """Audit ``root`` and write sorted SHA-256 entries for all release files; return count."""
    root = Path(root)
    audit(root)
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.relative_to(root).parts
        and p.name != "PACKAGE_MANIFEST.sha256"
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}\n")
    (root / "PACKAGE_MANIFEST.sha256").write_text("".join(lines))
    return len(lines)


if __name__ == "__main__":
    print(f"Updated manifest for {update()} public release files.")
