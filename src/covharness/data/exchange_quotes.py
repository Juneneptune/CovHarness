"""Paper-style single-exchange quote cleaning. P3, then existing Q1-Q4.

Barndorff-Nielsen et al. (2011), Section 5.1, retain quotes from one listing
venue before applying the quote filters. This adapter performs that P3
selection on exchange-level TAQ quotes and then calls
``clean_nbbo_quotes`` for P1, P2, and Q1-Q4. It does not copy those rules.

Consolidated NBBO cleaning remains available as ``clean_nbbo_quotes`` on
``best_bid`` / ``best_ask`` and is a labeled alternative path.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from covharness.data.quotes import (
    CleanedNbboQuotes,
    QuoteCleaningDiagnostics,
    clean_nbbo_quotes,
)
from covharness.data.venues import (
    ListingVenue,
    suffix_identity_map,
    taq_suffix_matches,
    venue_code_map,
)


@dataclass(frozen=True)
class SingleExchangeCleaningResult:
    """P3-filtered quotes after the production Q1-Q4 cleaner.

    ``quotes`` has the same columns as ``CleanedNbboQuotes.quotes``.
    ``n_before_p3`` counts exchange-level input rows. ``n_after_p3`` is the
    Q1-Q4 input. The Q1-Q4 diagnostics do not include P3 removals.
    """

    quotes: pd.DataFrame
    diagnostics: QuoteCleaningDiagnostics
    n_before_p3: int
    n_after_p3: int
    venue_by_asset: dict[str, tuple[str, ...]]

    @property
    def n_removed_p3(self) -> int:
        return self.n_before_p3 - self.n_after_p3


def apply_p3(
    data: pd.DataFrame,
    venues: dict[str, ListingVenue] | dict[str, tuple[str, ...]],
    *,
    asset_col: str = "asset",
    exchange_col: str = "ex",
    suffix_col: str = "sym_suffix",
) -> pd.DataFrame:
    """Keep quotes from the selected listing venue and TAQ suffix.

    This is P3 plus security identity. Quote filters Q1-Q4 are not applied
    here. If ``suffix_col`` is absent, only the exchange filter is applied so
    existing venue tests remain valid.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if asset_col not in data.columns or exchange_col not in data.columns:
        raise ValueError("data must contain asset and exchange columns")

    codes = venue_code_map(venues)
    if not codes:
        raise ValueError("venues must map at least one asset to TAQ exchange codes")

    # Allowed (asset, ex) pairs from the listing-venue map.
    allowed: set[tuple[str, str]] = set()
    for asset, ex_codes in codes.items():
        for code in ex_codes:
            allowed.add((str(asset), str(code)))

    # Keep only quotes issued by each asset's selected venue.
    keys = list(zip(data[asset_col].astype(str), data[exchange_col].astype(str)))
    venue_ok = pd.Series([key in allowed for key in keys], index=data.index)
    selected = data.loc[venue_ok].copy()

    # Keep only the requested TAQ suffix when identity is present on the frame.
    if suffix_col in selected.columns:
        intended = suffix_identity_map(venues)
        suffix_ok = pd.Series(
            [
                taq_suffix_matches(suffix, intended.get(str(asset), None))
                for asset, suffix in zip(selected[asset_col], selected[suffix_col])
            ],
            index=selected.index,
        )
        selected = selected.loc[suffix_ok].copy()
    return selected


def clean_single_exchange_quotes(
    data: pd.DataFrame,
    venues: dict[str, ListingVenue] | dict[str, tuple[str, ...]],
    *,
    timestamp_col: str = "timestamp",
    asset_col: str = "asset",
    exchange_col: str = "ex",
    bid_col: str = "bid",
    ask_col: str = "ask",
) -> SingleExchangeCleaningResult:
    """P3 on exchange-level quotes, then production P1, P2, Q1-Q4.

    ``bid`` / ``ask`` are mapped onto the existing cleaner as ``best_bid`` /
    ``best_ask``. That mapping is an input adapter, not a new Q1-Q4 rule.
    """
    n_before_p3 = len(data)
    # P3 and suffix identity. Retain the requested listing-venue security only.
    selected = apply_p3(
        data,
        venues,
        asset_col=asset_col,
        exchange_col=exchange_col,
    )
    n_after_p3 = len(selected)

    # Hand the selected stream to the existing Q1-Q4 implementation.
    frame = selected.rename(columns={bid_col: "best_bid", ask_col: "best_ask"})
    cleaned: CleanedNbboQuotes = clean_nbbo_quotes(
        frame,
        timestamp_col=timestamp_col,
        asset_col=asset_col,
        bid_col="best_bid",
        ask_col="best_ask",
    )
    return SingleExchangeCleaningResult(
        quotes=cleaned.quotes,
        diagnostics=cleaned.diagnostics,
        n_before_p3=n_before_p3,
        n_after_p3=n_after_p3,
        venue_by_asset=venue_code_map(venues),
    )
