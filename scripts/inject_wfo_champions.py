from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.live_hyperparameters import (
    LiveHyperparameterValidationError,
    build_live_hyperparameters_payload,
    default_live_hyperparameters_path,
    extract_params_payload,
    load_live_hyperparameters,
    resolve_champion_params_path,
    validate_strategy_param_overrides,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge champion_params.json artifacts into live_hyperparameters.json",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="WFO output directories containing champion_params.json, or direct champion_params.json paths",
    )
    parser.add_argument(
        "--output",
        default=str(default_live_hyperparameters_path()),
        help="Destination live_hyperparameters.json path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        output_path = Path(args.output).expanduser().resolve()
        existing_params = load_live_hyperparameters(output_path) if output_path.exists() else {}

        champion_sources: list[tuple[Path, dict[str, object], dict[str, object]]] = []
        for source in args.sources:
            champion_path = resolve_champion_params_path(source)
            raw_payload = json.loads(champion_path.read_text(encoding="utf-8"))
            params = validate_strategy_param_overrides(extract_params_payload(raw_payload))
            meta = raw_payload.get("meta", {}) if isinstance(raw_payload, dict) else {}
            champion_sources.append((champion_path, params, meta if isinstance(meta, dict) else {}))

        payload = build_live_hyperparameters_payload(
            existing_params=existing_params,
            champion_sources=champion_sources,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", delete=False, dir=output_path.parent, encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(output_path)

        print(f"Wrote merged live hyperparameters to {output_path}")
        return 0
    except LiveHyperparameterValidationError as exc:
        print(f"CRITICAL: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"CRITICAL: unexpected injector failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())