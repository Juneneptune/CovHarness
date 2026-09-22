"""Run Binance SCREEN forecasts, SPA/MCS, and median-rank finalist selection.

CONFIRM remains locked. VALIDATION grids are not rerun.
"""

from __future__ import annotations

from pathlib import Path

from covharness.protocol.binance_screen import (
    freeze_screen_configuration,
    run_binance_screen,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    screen_config = root / "configs/binance_open_data_screen.yaml"
    if not screen_config.is_file():
        freeze_screen_configuration(
            core_path=root / "configs/binance_open_data_core.yaml",
            selection_path=root / "results/binance_validation_selection.json",
            output_path=screen_config,
            summary_path=root / "results/binance_validation_candidate_summary.csv",
        )
    result = run_binance_screen(
        panel_path=root / "data/processed/binance_five_asset_panel.npz",
        core_path=root / "configs/binance_open_data_core.yaml",
        selection_path=root / "results/binance_validation_selection.json",
        screen_config_path=screen_config,
        output_dir=root / "results",
        summary_path=root / "results/binance_validation_candidate_summary.csv",
    )
    finalists = result["finalists"]
    print(
        "SCREEN finalists (selection data only) "
        f"DL={finalists['dl_finalist']} ECON={finalists['econ_finalist']}",
        flush=True,
    )
    print("artifact hashes", result["artifact_hashes"], flush=True)


if __name__ == "__main__":
    main()
