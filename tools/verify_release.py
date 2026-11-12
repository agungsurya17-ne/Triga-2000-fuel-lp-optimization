"""Verify the distributed release files against PACKAGE_MANIFEST.sha256."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify(root):
    """Check SHA-256 entries below package ``root``; return the count or raise on mismatch."""
    count = 0
    listed = set()
    for line in (root / "PACKAGE_MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid manifest path: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
        if relative in listed:
            raise ValueError(f"Duplicate manifest entry: {relative}")
        listed.add(relative)
        count += 1
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.relative_to(root).parts
        and p.name != "PACKAGE_MANIFEST.sha256"
    }
    if actual != listed:
        raise ValueError(f"Unlisted or missing release files: {sorted(actual ^ listed)}")
    return count


if __name__ == "__main__":
    print(f"Verified {verify(ROOT)} release files.")
