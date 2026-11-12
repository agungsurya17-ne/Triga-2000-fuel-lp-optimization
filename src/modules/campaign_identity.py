"""Validate portable campaign IDs before creating result directories.

validate_campaign_id rejects empty names, path traversal, and path separators,
then returns an ID suitable for checkpoint and campaign artifact paths."""

from __future__ import annotations

import re
from typing import Any


CAMPAIGN_ID_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?")
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


def validate_campaign_id(value: Any) -> str:
    """Validate one portable campaign ID and return it unchanged.

    Parameters
    ----------
    value : object
        Candidate identifier used as a results-directory name and run-ID prefix.

    Returns
    -------
    str
        Identifier containing 1-128 portable filename characters."""
    if not isinstance(value, str) or not value:
        raise ValueError("campaign_id harus berupa string non-kosong.")
    if value != value.strip():
        raise ValueError("campaign_id tidak boleh memiliki whitespace di tepi.")
    placeholder_signature = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    if (
        placeholder_signature.startswith("GANTI_DENGAN")
        or "CAMPAIGN_ID_UNIK" in placeholder_signature
    ):
        raise ValueError("campaign_id masih berupa placeholder; isi ID campaign yang unik.")
    if CAMPAIGN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "campaign_id hanya boleh memakai huruf ASCII, angka, titik, underscore, atau tanda hubung; karakter pertama dan terakhir harus alfanumerik."
        )
    if value.split(".", maxsplit=1)[0].upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError("campaign_id memakai nama perangkat yang dicadangkan Windows.")
    return value
