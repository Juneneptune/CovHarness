"""Unit tests for the paper-style P3 adapter. Q1-Q4 are not re-implemented here."""

from __future__ import annotations

import pandas as pd
import pytest

from covharness.data.exchange_quotes import apply_p3, clean_single_exchange_quotes
from covharness.data.quotes import clean_nbbo_quotes
from covharness.data.venues import listing_venue_from_crsp_exchcd, taq_suffix_sql_predicate


DAY = pd.Timestamp("2009-02-13")


def _ts(hour: int, minute: int, second: int = 0) -> pd.Timestamp:
    return pd.Timestamp(
        year=DAY.year,
        month=DAY.month,
        day=DAY.day,
        hour=hour,
        minute=minute,
        second=second,
    )


def _row(ts: pd.Timestamp, asset: str, ex: str, bid: float, ask: float) -> dict[str, object]:
    return {
        "timestamp": ts,
        "asset": asset,
        "ex": ex,
        "bid": bid,
        "ask": ask,
    }


def test_crsp_exchcd_maps_to_official_taq_codes() -> None:
    nyse = listing_venue_from_crsp_exchcd("IBM", 1)
    nasdaq = listing_venue_from_crsp_exchcd("MSFT", 3)
    assert nyse.taq_ex_codes == ("N",)
    assert nasdaq.taq_ex_codes == ("T", "Q")
    assert nyse.crsp_exchange_name == "NYSE"
    assert nasdaq.crsp_exchange_name == "NASDAQ"
    with pytest.raises(ValueError):
        listing_venue_from_crsp_exchcd("SPY", 4)


def test_p3_keeps_only_selected_exchange() -> None:
    data = pd.DataFrame(
        [
            _row(_ts(10, 0, 0), "JPM", "N", 25.00, 25.02),
            _row(_ts(10, 0, 1), "JPM", "T", 22.50, 22.52),
            _row(_ts(10, 0, 2), "JPM", "Z", 22.60, 22.62),
        ]
    )
    kept = apply_p3(data, {"JPM": ("N",)})
    assert len(kept) == 1
    assert kept.iloc[0]["ex"] == "N"
    assert kept.iloc[0]["bid"] == 25.00


def test_p3_uses_listing_venue_per_asset() -> None:
    data = pd.DataFrame(
        [
            _row(_ts(10, 0, 0), "IBM", "N", 90.00, 90.02),
            _row(_ts(10, 0, 0), "IBM", "T", 89.00, 89.02),
            _row(_ts(10, 0, 0), "MSFT", "T", 19.00, 19.02),
            _row(_ts(10, 0, 0), "MSFT", "N", 18.00, 18.02),
        ]
    )
    venues = {
        "IBM": listing_venue_from_crsp_exchcd("IBM", 1),
        "MSFT": listing_venue_from_crsp_exchcd("MSFT", 3),
    }
    kept = apply_p3(data, venues)
    assert set(zip(kept["asset"], kept["ex"])) == {("IBM", "N"), ("MSFT", "T")}


def test_single_exchange_cleaner_reuses_q1_median() -> None:
    stamp = _ts(10, 0, 0)
    data = pd.DataFrame(
        [
            _row(stamp, "JPM", "N", 25.00, 25.06),
            _row(stamp, "JPM", "N", 25.02, 25.08),
            _row(stamp, "JPM", "N", 25.04, 25.10),
            _row(stamp, "JPM", "T", 22.00, 22.02),
        ]
    )
    result = clean_single_exchange_quotes(data, {"JPM": ("N",)})
    assert result.n_before_p3 == 4
    assert result.n_after_p3 == 3
    assert result.n_removed_p3 == 1
    assert len(result.quotes) == 1
    assert result.quotes.iloc[0]["best_bid"] == pytest.approx(25.02)
    assert result.quotes.iloc[0]["best_ask"] == pytest.approx(25.08)


def test_single_exchange_path_keeps_locked_quotes_like_q2() -> None:
    data = pd.DataFrame(
        [
            _row(_ts(10, 0, 0), "JPM", "N", 25.00, 25.00),
            _row(_ts(10, 0, 1), "JPM", "N", 25.01, 25.03),
        ]
    )
    result = clean_single_exchange_quotes(data, {"JPM": ("N",)})
    assert result.diagnostics.n_removed_q2 == 0
    assert (result.quotes["spread"] == 0.0).sum() == 1


def test_nbbo_cleaner_remains_available_as_alternative_path() -> None:
    data = pd.DataFrame(
        {
            "timestamp": [_ts(10, 0, 0)],
            "asset": ["JPM"],
            "best_bid": [25.00],
            "best_ask": [25.02],
        }
    )
    nbbo = clean_nbbo_quotes(data)
    assert len(nbbo.quotes) == 1
    assert nbbo.quotes.iloc[0]["midquote"] == pytest.approx(25.01)


def test_p3_does_not_merge_root_sharing_suffixes() -> None:
    """Preferred quotes that share a root must not enter the ordinary stream.

    Root-only venue selection would keep the three NYSE rows. Identity-aware
    selection keeps only the ordinary/common suffix on the listing venue.
    """
    data = pd.DataFrame(
        [
            {**_row(_ts(10, 0, 0), "JPM", "N", 25.00, 25.02), "sym_suffix": None},
            {**_row(_ts(10, 0, 1), "JPM", "N", 17.73, 17.81), "sym_suffix": "PRK"},
            {**_row(_ts(10, 0, 2), "JPM", "N", 19.72, 19.90), "sym_suffix": "PRW"},
            {**_row(_ts(10, 0, 3), "JPM", "T", 24.90, 24.92), "sym_suffix": None},
        ]
    )
    venues = {"JPM": listing_venue_from_crsp_exchcd("JPM", 1)}
    kept = apply_p3(data, venues)
    assert len(kept) == 1
    assert kept.iloc[0]["ex"] == "N"
    assert kept.iloc[0]["bid"] == 25.00
    assert pd.isna(kept.iloc[0]["sym_suffix"])


def test_p3_treats_null_and_blank_suffix_as_ordinary_share() -> None:
    data = pd.DataFrame(
        [
            {**_row(_ts(10, 0, 0), "IBM", "N", 90.00, 90.02), "sym_suffix": None},
            {**_row(_ts(10, 0, 1), "IBM", "N", 90.04, 90.06), "sym_suffix": ""},
            {**_row(_ts(10, 0, 2), "IBM", "N", 90.08, 90.10), "sym_suffix": "  "},
            {**_row(_ts(10, 0, 3), "IBM", "N", 80.00, 80.02), "sym_suffix": "PRK"},
        ]
    )
    kept = apply_p3(data, {"IBM": listing_venue_from_crsp_exchcd("IBM", 1)})
    assert len(kept) == 3
    assert set(kept["bid"]) == {90.00, 90.04, 90.08}


def test_p3_keeps_only_requested_nonblank_suffix() -> None:
    data = pd.DataFrame(
        [
            {**_row(_ts(10, 0, 0), "JPM", "N", 25.00, 25.02), "sym_suffix": None},
            {**_row(_ts(10, 0, 1), "JPM", "N", 17.73, 17.81), "sym_suffix": "PRK"},
            {**_row(_ts(10, 0, 2), "JPM", "N", 19.72, 19.90), "sym_suffix": "PRW"},
        ]
    )
    venues = {
        "JPM": listing_venue_from_crsp_exchcd("JPM", 1, taq_sym_suffix="PRK"),
    }
    kept = apply_p3(data, venues)
    assert len(kept) == 1
    assert kept.iloc[0]["sym_suffix"] == "PRK"
    assert kept.iloc[0]["bid"] == 17.73


def test_production_cleaner_does_not_merge_suffixes_into_one_midquote() -> None:
    stamp = _ts(10, 0, 0)
    data = pd.DataFrame(
        [
            {**_row(stamp, "JPM", "N", 25.00, 25.02), "sym_suffix": None},
            {**_row(stamp, "JPM", "N", 17.73, 17.81), "sym_suffix": "PRK"},
        ]
    )
    result = clean_single_exchange_quotes(
        data, {"JPM": listing_venue_from_crsp_exchcd("JPM", 1)}
    )
    assert result.n_after_p3 == 1
    assert len(result.quotes) == 1
    assert result.quotes.iloc[0]["best_bid"] == pytest.approx(25.00)
    assert result.quotes.iloc[0]["midquote"] == pytest.approx(25.01)


def test_ordinary_share_sql_predicate_matches_null_or_blank() -> None:
    ordinary = taq_suffix_sql_predicate("sym_suffix", None)
    assert "IS NULL" in ordinary
    assert "''" in ordinary
    preferred = taq_suffix_sql_predicate("sym_suffix", "PRK")
    assert "PRK" in preferred
    assert "IS NULL" not in preferred
