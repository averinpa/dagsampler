import argparse
import json
from pathlib import Path

from .causal_sim import CausalDataGenerator


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate causal simulation dataset.")
    parser.add_argument("--config", required=True, help="Path to JSON simulation config.")
    parser.add_argument("--output", required=True, help="Path to output CSV dataset.")
    parser.add_argument(
        "--params-out",
        default=None,
        help="Optional path to output JSON with final parametrization.",
    )
    parser.add_argument(
        "--edges-out",
        default=None,
        help="Optional path to output JSON list of DAG edges.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    result = CausalDataGenerator(config).simulate()
    result["data"].to_csv(output_path, index=False)

    if args.params_out:
        params_path = Path(args.params_out)
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(result["parametrization"]), f, indent=2)

    if args.edges_out:
        edges_path = Path(args.edges_out)
        edges = [list(edge) for edge in result["dag"].edges()]
        with edges_path.open("w", encoding="utf-8") as f:
            json.dump(edges, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
