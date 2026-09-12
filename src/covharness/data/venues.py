"""Official venue codes for the paper-style single-exchange (P3) adapter.

CRSP ``exchcd`` values follow the CRSP US Stock documentation
(1 = NYSE, 3 = NASDAQ). TAQ ``ex`` values follow the Daily TAQ Client
Specification Exchange field (N = New York Stock Exchange, T/Q = NASDAQ
Stock Exchange).

Barndorff-Nielsen et al. (2011), Section 5.1, retain a single listing-venue
quote stream before Q1-Q4. NYSE-listed names use NYSE. NASDAQ-listed names
in their sample (INTC, MSFT) use NASDAQ. This module maps listing venue to
TAQ codes. It does not re-implement Q1-Q4.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# CRSP stocknames.exchcd / HEXCH (CRSP US Stock user guide).
CRSP_EXCHCD_NYSE = 1
CRSP_EXCHCD_NASDAQ = 3

CRSP_EXCHCD_NAME = {
    CRSP_EXCHCD_NYSE: "NYSE",
    2: "NYSE American",
    CRSP_EXCHCD_NASDAQ: "NASDAQ",
}

# Daily TAQ Client Specification, quote Exchange field.
TAQ_EX_NYSE = ("N",)
TAQ_EX_NASDAQ = ("T", "Q")

CRSP_EXCHCD_TO_TAQ_EX = {
    CRSP_EXCHCD_NYSE: TAQ_EX_NYSE,
    CRSP_EXCHCD_NASDAQ: TAQ_EX_NASDAQ,
}

TAQ_EX_CODE_SOURCE = (
    "NYSE Daily TAQ Client Specification, Daily Quotes Exchange field. "
    "N is New York Stock Exchange. T/Q is NASDAQ Stock Exchange."
)
CRSP_EXCHCD_SOURCE = (
    "CRSP stocknames.exchcd. 1 is NYSE. 3 is NASDAQ "
    "(CRSP US Stock documentation)."
)
TAQ_SUFFIX_SOURCE = (
    "NYSE Daily TAQ Client Specification Appendix B. A blank or NULL "
    "suffix is the ordinary/common share. Preferred series use PR plus a letter."
)

# Ordinary/common share. This TAQ millisecond table stores that suffix as SQL NULL.
TAQ_ORDINARY_SHARE_SUFFIX = None


@dataclass(frozen=True)
class ListingVenue:
    """Listing venue, TAQ ``ex`` codes, and TAQ suffix retained by P3.

    ``taq_sym_suffix`` is ``None`` for the ordinary/common share. Preferred,
    warrant, and class shares carry an explicit suffix and are a different
    security.
    """

    asset: str
    crsp_exchcd: int
    crsp_exchange_name: str
    taq_ex_codes: tuple[str, ...]
    source: str
    taq_sym_suffix: str | None = TAQ_ORDINARY_SHARE_SUFFIX


def is_ordinary_taq_suffix(value: object) -> bool:
    """True when ``sym_suffix`` is SQL NULL, pandas NA, or a blank string."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def taq_suffix_matches(value: object, intended: str | None) -> bool:
    """Match a quote suffix to the requested TAQ security identity."""
    if intended is None or str(intended).strip() == "":
        return is_ordinary_taq_suffix(value)
    if is_ordinary_taq_suffix(value):
        return False
    return str(value).strip() == str(intended).strip()


def taq_suffix_sql_predicate(column: str, intended: str | None) -> str:
    """SQL predicate for one TAQ suffix identity.

    Ordinary shares on the 2009 millisecond CQM sample are stored as NULL.
    Blank strings are also accepted so the filter does not depend on one
    encoding. Nonblank suffixes are matched after trim.
    """
    if intended is None or str(intended).strip() == "":
        return f"({column} IS NULL OR btrim(CAST({column} AS VARCHAR)) = '')"
    escaped = str(intended).strip().replace("'", "''")
    return f"btrim(CAST({column} AS VARCHAR)) = '{escaped}'"


def listing_venue_from_crsp_exchcd(
    asset: str,
    crsp_exchcd: int,
    *,
    taq_sym_suffix: str | None = TAQ_ORDINARY_SHARE_SUFFIX,
) -> ListingVenue:
    """Map a CRSP listing code to TAQ quote-exchange codes and suffix.

    Raises
    ------
    ValueError
        If ``crsp_exchcd`` has no TAQ mapping in this adapter.
    """
    exchcd = int(crsp_exchcd)
    # Map CRSP listing code to the official TAQ Exchange-field codes.
    taq_codes = CRSP_EXCHCD_TO_TAQ_EX.get(exchcd)
    if taq_codes is None:
        raise ValueError(
            f"no TAQ exchange mapping for CRSP exchcd={exchcd} (asset={asset})"
        )
    suffix = None if taq_sym_suffix is None or str(taq_sym_suffix).strip() == "" else str(taq_sym_suffix).strip()
    return ListingVenue(
        asset=asset,
        crsp_exchcd=exchcd,
        crsp_exchange_name=CRSP_EXCHCD_NAME.get(exchcd, f"CRSP_{exchcd}"),
        taq_ex_codes=taq_codes,
        taq_sym_suffix=suffix,
        source=f"{CRSP_EXCHCD_SOURCE} {TAQ_EX_CODE_SOURCE} {TAQ_SUFFIX_SOURCE}",
    )


def venue_code_map(venues: dict[str, ListingVenue] | dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Normalize asset -> TAQ ``ex`` codes."""
    out: dict[str, tuple[str, ...]] = {}
    for asset, value in venues.items():
        if isinstance(value, ListingVenue):
            out[asset] = value.taq_ex_codes
        else:
            out[asset] = tuple(value)
    return out


def suffix_identity_map(
    venues: dict[str, ListingVenue] | dict[str, tuple[str, ...]],
) -> dict[str, str | None]:
    """Normalize asset -> TAQ ``sym_suffix``. Missing identities are ordinary shares."""
    out: dict[str, str | None] = {}
    for asset, value in venues.items():
        if isinstance(value, ListingVenue):
            out[str(asset)] = value.taq_sym_suffix
        else:
            out[str(asset)] = TAQ_ORDINARY_SHARE_SUFFIX
    return out
