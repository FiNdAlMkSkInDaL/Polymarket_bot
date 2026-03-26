from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
import json
import logging
from pathlib import Path
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.archive_market_analyzer import load_market_map_entries, parse_iso_datetime  # noqa: E402
from src.data.universe_builder import (  # noqa: E402
    MarketCandidate,
    UniverseBuilder,
    UniverseBuilderConfig,
)
from src.signals.microstructure_utils import CausalLagConfig  # noqa: E402
from src.tools.contagion_validator import (  # noqa: E402
    ContagionValidationReport,
    ContagionValidator,
    ContagionValidatorConfig,
)


logging.getLogger().setLevel(logging.WARNING)


DEFAULT_ARCHIVE_PATH = PROJECT_ROOT / "data" / "vps_march2026"
DEFAULT_MARKET_MAP = PROJECT_ROOT / "data" / "market_map.json"
DEFAULT_MAX_LAGGER_AGE_MS = 600_000
DEFAULT_MAX_LEADER_AGE_MS = 5_000
DEFAULT_MAX_CAUSAL_LAG_MS = 600_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate market clusters against the archive-backed UniverseBuilder "
            "and ContagionValidator using the accepted 600000 ms lagger freshness baseline."
        ),
    )
    parser.add_argument(
        "--clusters-json",
        required=True,
        help=(
            "Path to a JSON file containing a list of cluster candidates or an object with a 'clusters' array. "
            "Each cluster must include eval_window_start, eval_window_end, and a candidates array."
        ),
    )
    parser.add_argument(
        "--archive-path",
        default=str(DEFAULT_ARCHIVE_PATH),
        help=f"Archive root used by both tools (default: {DEFAULT_ARCHIVE_PATH}).",
    )
    parser.add_argument(
        "--market-map",
        default=str(DEFAULT_MARKET_MAP),
        help=f"Market map used to resolve market_id to asset ids (default: {DEFAULT_MARKET_MAP}).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the full evaluation report as JSON.",
    )
    parser.add_argument(
        "--min-correlation",
        type=str,
        default="0.10",
        help="UniverseBuilder minimum empirical correlation threshold.",
    )
    parser.add_argument(
        "--min-events-per-day",
        type=int,
        default=3,
        help="UniverseBuilder minimum leader events per day.",
    )
    parser.add_argument(
        "--min-archive-days",
        type=int,
        default=1,
        help="UniverseBuilder minimum archive days required.",
    )
    parser.add_argument(
        "--require-causal-ordering",
        action="store_true",
        help="Reject pairs where the lagger historically leads the leader.",
    )
    parser.add_argument(
        "--max-lagger-age-ms",
        type=int,
        default=DEFAULT_MAX_LAGGER_AGE_MS,
        help=f"Lagger freshness threshold for builder and validator (default: {DEFAULT_MAX_LAGGER_AGE_MS}).",
    )
    parser.add_argument(
        "--max-leader-age-ms",
        type=int,
        default=DEFAULT_MAX_LEADER_AGE_MS,
        help=f"Leader freshness threshold for validator (default: {DEFAULT_MAX_LEADER_AGE_MS}).",
    )
    parser.add_argument(
        "--max-causal-lag-ms",
        type=int,
        default=DEFAULT_MAX_CAUSAL_LAG_MS,
        help=f"Maximum causal lag for validator (default: {DEFAULT_MAX_CAUSAL_LAG_MS}).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional replay event cap for faster iteration.",
    )
    parser.add_argument(
        "--emit-per-event-telemetry",
        action="store_true",
        help="Include per-pair telemetry in the validator output files.",
    )
    return parser.parse_args(argv)


def _read_clusters(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        clusters = payload.get("clusters")
        if isinstance(clusters, list):
            return [item for item in clusters if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("clusters JSON must be a list or an object with a 'clusters' array")


def _normalize_tags(raw: Any) -> frozenset[str]:
    if isinstance(raw, str):
        values = [tag.strip() for tag in raw.split(",") if tag.strip()]
        return frozenset(values)
    if isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset(str(tag).strip() for tag in raw if str(tag).strip())
    return frozenset()


def _build_market_candidate(raw: dict[str, Any]) -> MarketCandidate:
    market_id = str(raw.get("market_id") or "").strip()
    question = str(raw.get("question") or market_id).strip()
    expected_role = str(raw.get("expected_role") or "EITHER").strip().upper()
    if expected_role not in {"LEADER", "LAGGER", "EITHER"}:
        raise ValueError(f"invalid expected_role for market {market_id!r}: {expected_role!r}")
    if not market_id:
        raise ValueError("candidate market is missing market_id")
    return MarketCandidate(
        market_id=market_id,
        question=question,
        thematic_tags=_normalize_tags(raw.get("thematic_tags") or raw.get("tags") or []),
        expected_role=expected_role,
    )


def _resolve_cluster_market_rows(
    cluster_market_ids: list[str],
    market_map_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = {market_id for market_id in cluster_market_ids}
    return [entry for entry in market_map_entries if str(entry.get("market_id")) in selected]


def _serialize_validation_report(report: ContagionValidationReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    payload = asdict(report)
    payload["causal_lag_config"] = asdict(report.causal_lag_config)
    return payload


def _print_cluster_report(result: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print(f"Cluster: {result['cluster_name']}")
    print(f"Status: {result['status']}")
    print(f"Replay date: {result['replay_date']}")
    print(f"Evaluation window: {result['eval_window_start']} -> {result['eval_window_end']}")
    build_report = result["builder_report"]
    print(
        "Builder funnel: "
        f"candidates={build_report['candidates_evaluated']}, "
        f"corr={build_report['pairs_passing_correlation']}, "
        f"fresh={build_report['pairs_passing_freshness']}, "
        f"causal={build_report['pairs_passing_causal_ordering']}"
    )
    leader = build_report.get("leader_market_id") or "none"
    recommended = build_report.get("recommended_cluster") or []
    print(f"Leader: {leader}")
    print(f"Recommended cluster: {', '.join(recommended) if recommended else 'none'}")

    validation = result.get("validation_report")
    if validation:
        print(
            "Validator: "
            f"pairs={validation['cross_market_pairs_evaluated']}, "
            f"pass_rate={validation['causal_gate_pass_rate'] * 100.0:.2f}%, "
            f"signals={validation['signals_fired']}, "
            f"dominant_suppressor={validation['dominant_suppressor']}"
        )
        print(
            "Validator lagger age: "
            f"median={validation['median_lagger_age_ms']:.2f} ms, "
            f"p95={validation['p95_lagger_age_ms']:.2f} ms"
        )
    else:
        print("Validator: skipped")

    rejection_reasons = build_report.get("rejection_reasons") or {}
    if rejection_reasons:
        print("Rejections:")
        for market_id, reason in sorted(rejection_reasons.items()):
            print(f"  - {market_id}: {reason}")
    print()


def _evaluate_cluster(
    cluster: dict[str, Any],
    *,
    builder: UniverseBuilder,
    validator_config: ContagionValidatorConfig,
    validator_causal_config: CausalLagConfig,
    market_map_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    cluster_name = str(cluster.get("name") or cluster.get("cluster_name") or "unnamed_cluster")
    eval_window_start = str(cluster.get("eval_window_start") or "").strip()
    eval_window_end = str(cluster.get("eval_window_end") or "").strip()
    if not eval_window_start or not eval_window_end:
        raise ValueError(f"cluster {cluster_name!r} is missing eval_window_start or eval_window_end")

    replay_date = str(cluster.get("replay_date") or parse_iso_datetime(eval_window_start).date().isoformat())
    candidates_raw = cluster.get("candidates") or cluster.get("candidate_markets")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError(f"cluster {cluster_name!r} must define a non-empty candidates array")
    candidates = [_build_market_candidate(item) for item in candidates_raw if isinstance(item, dict)]
    if not candidates:
        raise ValueError(f"cluster {cluster_name!r} contains no valid candidates")

    build_report = builder.build_cluster(candidates, eval_window_start, eval_window_end)
    validation_report: ContagionValidationReport | None = None
    status = "REJECTED"
    notes: list[str] = []

    if build_report.recommended_cluster and len(build_report.recommended_cluster) >= 2:
        resolved_market_rows = _resolve_cluster_market_rows(build_report.recommended_cluster, market_map_entries)
        if len(resolved_market_rows) == len(build_report.recommended_cluster):
            with tempfile.TemporaryDirectory(prefix="universe_builder_") as temp_dir:
                temp_universe_path = Path(temp_dir) / f"{cluster_name}_universe.json"
                temp_output_path = Path(temp_dir) / f"{cluster_name}_validation.json"
                temp_universe_path.write_text(json.dumps(resolved_market_rows, indent=2), encoding="utf-8")
                cluster_validator = ContagionValidator(
                    ContagionValidatorConfig(
                        archive_path=validator_config.archive_path,
                        universe_path=str(temp_universe_path),
                        max_events=validator_config.max_events,
                        emit_per_event_telemetry=validator_config.emit_per_event_telemetry,
                    )
                )
                validation_report = cluster_validator.run(
                    replay_date=replay_date,
                    causal_lag_config=validator_causal_config,
                    output_path=str(temp_output_path),
                )
            status = "ACCEPTED"
        else:
            missing = sorted(set(build_report.recommended_cluster) - {row["market_id"] for row in resolved_market_rows})
            notes.append(f"validator skipped: unresolved market ids in market map: {', '.join(missing)}")
    else:
        notes.append("validator skipped: no viable 2+ market cluster produced by UniverseBuilder")

    if validation_report is not None and validation_report.cross_market_pairs_evaluated <= 0:
        status = "REJECTED"
        notes.append("validator found zero cross-market pairs to evaluate")

    return {
        "cluster_name": cluster_name,
        "status": status,
        "replay_date": replay_date,
        "eval_window_start": eval_window_start,
        "eval_window_end": eval_window_end,
        "builder_report": asdict(build_report),
        "validation_report": _serialize_validation_report(validation_report),
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clusters_path = Path(args.clusters_json)
    archive_path = Path(args.archive_path)
    market_map_path = Path(args.market_map)

    if not clusters_path.exists():
        print(f"ERROR: candidate cluster file not found: {clusters_path}", file=sys.stderr)
        return 1
    if not archive_path.exists():
        print(f"ERROR: archive path not found: {archive_path}", file=sys.stderr)
        return 1
    if not market_map_path.exists():
        print(f"ERROR: market map not found: {market_map_path}", file=sys.stderr)
        return 1

    clusters = _read_clusters(clusters_path)
    if not clusters:
        print("ERROR: no cluster definitions found", file=sys.stderr)
        return 1

    market_map_entries = load_market_map_entries(market_map_path)
    builder = UniverseBuilder(
        UniverseBuilderConfig(
            min_correlation=Decimal(str(args.min_correlation)),
            min_events_per_day=int(args.min_events_per_day),
            min_archive_days=int(args.min_archive_days),
            max_lagger_age_ms=int(args.max_lagger_age_ms),
            require_causal_ordering=bool(args.require_causal_ordering),
        )
    )
    builder._archive_path = archive_path
    builder._market_map_entries = market_map_entries

    validator_config = ContagionValidatorConfig(
        archive_path=str(archive_path),
        universe_path=str(market_map_path),
        max_events=args.max_events,
        emit_per_event_telemetry=bool(args.emit_per_event_telemetry),
    )
    causal_config = CausalLagConfig(
        max_leader_age_ms=float(args.max_leader_age_ms),
        max_lagger_age_ms=float(args.max_lagger_age_ms),
        max_causal_lag_ms=float(args.max_causal_lag_ms),
        allow_negative_lag=False,
    )

    results: list[dict[str, Any]] = []
    for cluster in clusters:
        results.append(
            _evaluate_cluster(
                cluster,
                builder=builder,
                validator_config=validator_config,
                validator_causal_config=causal_config,
                market_map_entries=market_map_entries,
            )
        )

    print("UNIVERSE BUILDER / CONTAGION VALIDATOR REPORT")
    print(f"Archive: {archive_path}")
    print(f"Market map: {market_map_path}")
    print(f"Threshold: max_lagger_age_ms={args.max_lagger_age_ms}")
    print()
    for result in results:
        _print_cluster_report(result)
        if result["notes"]:
            for note in result["notes"]:
                print(f"Note: {note}")
            print()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"clusters": results}, indent=2), encoding="utf-8")
        print(f"Full JSON report written to: {output_path}")

    accepted = sum(1 for result in results if result["status"] == "ACCEPTED")
    print(f"Accepted clusters: {accepted}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())