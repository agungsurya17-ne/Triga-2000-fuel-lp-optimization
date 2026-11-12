"""Configure local executables and nuclear data before starting a new campaign."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]


def configure(cross_sections, executable):
    """Set XML index ``cross_sections`` and OpenMC ``executable``; return edited file names."""
    if any((ROOT / "results").rglob("checkpoint.json")):
        raise RuntimeError(
            "Existing campaign checkpoints found. Configure a fresh release copy instead."
        )
    index = Path(cross_sections).expanduser().resolve(strict=True)
    if ET.parse(index).getroot().tag != "cross_sections":
        raise ValueError("Expected an OpenMC cross_sections.xml index.")
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"OpenMC executable not found: {executable}")
    version = subprocess.check_output([resolved, "--version"], text=True, timeout=20)
    if not re.search(r"OpenMC version 0\.15\.3\b", version):
        raise ValueError("This release requires the OpenMC 0.15.3 executable.")
    edits = {}
    materials = ROOT / "config/materials.xml"
    text, count = re.subn(
        r"(<cross_sections>).*?(</cross_sections>)",
        lambda match: match[1] + escape(str(index)) + match[2],
        materials.read_text(),
    )
    if count != 1:
        raise ValueError("Expected exactly one materials cross_sections entry.")
    edits[materials] = text
    for name in ["config.ini", "smoke_generation0.ini"]:
        path = ROOT / "config" / name
        text, count = re.subn(
            r"^openmc_executable = .*$",
            lambda match: "openmc_executable = " + str(Path(resolved).resolve()),
            path.read_text(),
            flags=re.M,
        )
        if count != 1:
            raise ValueError(f"Expected one executable setting in {name}.")
        edits[path] = text
    for path, text in edits.items():
        path.write_text(text)
    return [str(path.relative_to(ROOT)) for path in edits]


def main():
    """Read local path arguments, configure new-campaign inputs, and return no value."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-sections", required=True, type=Path)
    parser.add_argument("--openmc-executable", default="openmc")
    arguments = parser.parse_args()
    for name in configure(arguments.cross_sections, arguments.openmc_executable):
        print("Configured:", name)
    print(
        "Local configuration changes release checksums. Keep the original release as the reference."
    )


if __name__ == "__main__":
    main()
