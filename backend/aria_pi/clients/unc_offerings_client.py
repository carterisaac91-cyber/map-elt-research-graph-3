"""UNC offerings client — pure dict lookup, zero network calls.

Loads unc_offerings_map.json once at module import and exposes
resolve_unc_offerings(sector_key_or_name) which returns the list of UNC
pathway blocks that match the given sector.  The function normalises the
sector string via the embedded alias table so callers can pass either the
canonical sector code ("H"), the full label ("Health & Life Sciences"),
or any common abbreviation ("pharma", "pharmaceutical", "biotech", …).

Architecture contract:
  • deterministic — same input always yields same output
  • zero network — file is read once at boot, no I/O in the hot path
  • returns EMPTY_OFFERINGS instead of None so the frontend never needs
    a null-guard on this field
"""

from __future__ import annotations
import json
import pathlib
from typing import Any

# ---------------------------------------------------------------------------
# Load data once at module import
# ---------------------------------------------------------------------------
_DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "unc_offerings_map.json"

try:
    with open(_DATA_PATH, "r", encoding="utf-8") as _f:
        _DATA: dict[str, Any] = json.load(_f)
except FileNotFoundError:
    _DATA = {"sectors": {}, "_meta": {"sector_aliases": {}}}

_SECTORS: dict[str, Any] = _DATA.get("sectors", {})
_ALIASES: dict[str, str] = _DATA.get("_meta", {}).get("sector_aliases", {})

# What the frontend receives when no matching sector data exists
EMPTY_OFFERINGS: dict[str, Any] = {
    "found": False,
    "sector_code": None,
    "sector_label": None,
    "pathways": [],
    "total_offerings": 0,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _normalise(raw: str) -> str | None:
    """Return the two-letter sector code or None if unrecognised."""
    if not raw:
        return None
    clean = raw.strip().upper()
    # Already a sector code?
    if clean in _SECTORS:
        return clean
    # Try alias lookup (lowercase)
    lower = raw.strip().lower()
    # Replace spaces and hyphens with underscores for alias matching
    alias_key = lower.replace(" ", "_").replace("-", "_")
    if alias_key in _ALIASES:
        return _ALIASES[alias_key]
    # Try direct lower key (e.g. "health")
    if lower in _ALIASES:
        return _ALIASES[lower]
    # Partial match — find first alias key that appears in the input string
    for alias_k, code in _ALIASES.items():
        if alias_k in alias_key or alias_key in alias_k:
            return code
    return None


def resolve_unc_offerings(sector: str) -> dict[str, Any]:
    """Return UNC offerings for a given sector string.

    Args:
        sector: Any of the canonical code ("H"), full label
                ("Health & Life Sciences"), or a recognised alias
                ("pharmaceutical", "pharma", "biotech", …).

    Returns:
        A dict with keys:
          found          – bool, False if sector not recognised
          sector_code    – str, e.g. "H"
          sector_label   – str, e.g. "Health & Life Sciences"
          pathways       – list of pathway dicts (see unc_offerings_map.json)
          total_offerings – int, total count of individual offering strings
    """
    code = _normalise(sector)
    if code is None or code not in _SECTORS:
        return EMPTY_OFFERINGS

    sector_data = _SECTORS[code]
    pathways = sector_data.get("pathways", [])
    total = sum(len(p.get("offerings", [])) for p in pathways)

    return {
        "found": True,
        "sector_code": code,
        "sector_label": sector_data.get("label", code),
        "pathways": pathways,
        "total_offerings": total,
    }


def all_sector_codes() -> list[str]:
    """Return the list of all supported sector codes."""
    return list(_SECTORS.keys())
