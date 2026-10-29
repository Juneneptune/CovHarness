"""Build shifted-day training panels through SCREEN.

Task D3a. The spec commit must precede this run. Evaluation targets stay
the UTC-day realized covariance. The script does not train a model, does
not rewrite the production panel, and does not build a CONFIRM label.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from covharness.data.binance_calendar import HALT_DATE, SEGMENT_BOUNDS, dates_through_screen
from covharness.realized.binance_panel import (
    PHASE_HOURS,
    build_preconfirm_phase_panels,
    phase_ends_by_label_close,
    phase_touches_halt,
    phase_usable_at_origin,
    production_rcov_through_screen,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "d45a5bce027b50be38bc5aaf722ddbd183392d10"
PANEL_SHA256 = "2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a"
OVERLAP_HOURS = 21


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise RuntimeError(f"hash mismatch for {path}: {digest} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "post_screen" / "D3a",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = REPO_ROOT / "data" / "processed" / "binance_five_asset_panel.npz"
    cache_root = (
        REPO_ROOT
        / "data"
        / "cache"
        / "open_data"
        / "binance"
        / "spot"
        / "monthly"
        / "klines"
        / "1m"
    )
    require_hash(panel_path, PANEL_SHA256)
    labels = dates_through_screen()
    confirm_start = SEGMENT_BOUNDS["CONFIRM"][0]
    print("D3a start", flush=True)
    panels = build_preconfirm_phase_panels(cache_root, hours=PHASE_HOURS)
    if tuple(panel.hour for panel in panels) != PHASE_HOURS:
        raise RuntimeError("phase hours do not match the registered set")

    # Phase 00 is the production day. Compare it before writing anything.
    phase_zero = panels[0]
    if phase_zero.dates != labels or phase_zero.dropped_dates != ():
        raise RuntimeError("phase 00 did not keep the production labels")
    stored = production_rcov_through_screen(panel_path, phase_zero.dates)
    if not np.array_equal(phase_zero.rcov, stored):
        raise RuntimeError("phase 00 realized covariance is not the production panel")

    summary_phases: list[dict[str, object]] = []
    for panel in panels:
        if any(day >= confirm_start or day == HALT_DATE for day in panel.dates):
            raise RuntimeError(f"phase {panel.hour} contains the halt or a CONFIRM date")
        if any(not phase_ends_by_label_close(day, panel.hour) for day in panel.dates):
            raise RuntimeError(f"phase {panel.hour} has a row that ends after its label")
        for day in panel.dates:
            previous = day - dt.timedelta(days=1)
            if phase_usable_at_origin(day, panel.hour, previous):
                raise RuntimeError(
                    f"{day.isoformat()} phase {panel.hour} is usable before its label date"
                )
            if not phase_usable_at_origin(day, panel.hour, day):
                raise RuntimeError(f"{day.isoformat()} phase {panel.hour} is not usable at its label")
        expected_dropped = tuple(day for day in labels if phase_touches_halt(day, panel.hour))
        if panel.dropped_dates != expected_dropped:
            raise RuntimeError(f"phase {panel.hour} dropped dates do not match the halt rule")
        if panel.hour == 0 and expected_dropped != ():
            raise RuntimeError("phase 00 dropped a production day")
        if panel.hour != 0 and dt.date(2023, 3, 25) not in panel.dropped_dates:
            raise RuntimeError(f"phase {panel.hour} kept the halt-touching label")
        if panel.dates[-1] != labels[-1]:
            raise RuntimeError(f"phase {panel.hour} does not end on the last SCREEN date")
        path = output_dir / f"phase_{panel.hour:02d}.npz"
        np.savez(
            path,
            hour=np.int64(panel.hour),
            dates=np.array([day.isoformat() for day in panel.dates]),
            dropped_dates=np.array([day.isoformat() for day in panel.dropped_dates]),
            assets=np.array(panel.assets),
            rcov=panel.rcov,
            rq=panel.rq,
            daily_returns=panel.daily_returns,
            contiguous_with_previous=panel.contiguous_with_previous,
        )
        summary_phases.append(
            {
                "hour": panel.hour,
                "n_rows": len(panel.dates),
                "n_dropped": len(panel.dropped_dates),
                "dropped_dates": [day.isoformat() for day in panel.dropped_dates],
                "date_start": panel.dates[0].isoformat(),
                "date_end": panel.dates[-1].isoformat(),
                "n_lag_breaks": int(np.sum(~panel.contiguous_with_previous)),
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }
        )
        print(f"D3a wrote phase {panel.hour:02d} rows={len(panel.dates)}", flush=True)

    require_hash(panel_path, PANEL_SHA256)
    summary = {
        "task": "D3a",
        "spec_commit": SPEC_COMMIT,
        "phase00_matches_production": True,
        "panel_sha256": PANEL_SHA256,
        "confirm_rows": 0,
        "overlap_hours_at_three_hour_offset": OVERLAP_HOURS,
        "overlap_note": (
            "Phase offsets three hours apart share 21 hours. Same-phase successive "
            "days share only the boundary. The samples are correlated and do not "
            "multiply the effective sample size."
        ),
        "phases": summary_phases,
    }
    summary_path = output_dir / "d3a_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("D3a done", flush=True)


if __name__ == "__main__":
    main()
