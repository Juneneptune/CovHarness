"""Run first-stage Binance VALIDATION fitting and within-family selection.

SCREEN and CONFIRM are not requested. CONFIRM remains locked.
"""

from __future__ import annotations

from pathlib import Path

from covharness.protocol.binance_validation import run_binance_validation


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = run_binance_validation(
        panel_path=root / "data/processed/binance_five_asset_panel.npz",
        config_path=root / "configs/binance_open_data_core.yaml",
        output_dir=root / "results",
    )
    selection = result["selection"]["families"]
    print("VALIDATION selection (development only)", flush=True)
    for family, info in selection.items():
        print(
            f"{family} {info['status']} {info.get('selected_id')} "
            f"qlike={info.get('mean_qlike')} frobenius={info.get('mean_frobenius')}",
            flush=True,
        )
    print("artifact hashes", result["artifact_hashes"], flush=True)


if __name__ == "__main__":
    main()
