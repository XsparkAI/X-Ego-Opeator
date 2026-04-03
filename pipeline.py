#!/usr/bin/env python3
"""
Ego-X Operator Pipeline — orchestrate video processing operators.

Usage:
  python pipeline.py --config pipeline_config.yaml
  python pipeline.py --episode path/to/episode
  python pipeline.py --config pipeline_config.yaml --report report.json
  python pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from operators.operator_base import Operator, OperatorResult  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Operator registry ──────────────────────────────────────────────

# Maps operator name → (module_path, class_name, config_class_name)
OPERATOR_REGISTRY: dict[str, tuple[str, str, str]] = {
    "video_quality":      ("operators.video_quality.op_impl",   "VideoQualityOperator",   "VideoQualityConfig"),
    "hand_detection":     ("operators.hand.op_impl",            "HandDetectionOperator",  "HandDetectionConfig"),
    "privacy_blur":       ("operators.privacy_blur.op_impl",    "PrivacyBlurOperator",    "PrivacyBlurConfig"),
    "transcode":          ("operators.transcode.op_impl",       "TranscodeOperator",      "TranscodeConfig"),
    "video_segmentation": ("operators.caption.op_impl",         "SegmentationOperator",   "SegmentationConfig"),
}

DEFAULT_ORDER = [
    "video_quality",
    "hand_detection",
    "privacy_blur",
    "transcode",
    "video_segmentation",
]

# Dependency graph: operator → set of operators it must wait for.
# Operators not listed here (or with empty sets) have no dependencies.
DEPENDENCIES: dict[str, set[str]] = {
    "transcode": {"privacy_blur"},  # transcode prefers rgb_blurred.mp4
}


def _import_operator(name: str, config_kwargs: dict[str, Any]) -> Operator:
    """Lazily import and instantiate an operator by registry name."""
    if name not in OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator: {name!r}. Available: {list(OPERATOR_REGISTRY)}")

    mod_path, cls_name, cfg_cls_name = OPERATOR_REGISTRY[name]
    mod = importlib.import_module(mod_path)
    operator_cls = getattr(mod, cls_name)
    config_cls = getattr(mod, cfg_cls_name)

    config = config_cls(**config_kwargs)
    return operator_cls(config=config)


# ── Pipeline ───────────────────────────────────────────────────────


class Pipeline:
    """Parallel-aware operator pipeline with dependency resolution."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.on_error: str = config.get("on_error", "continue")
        self.parallel: bool = config.get("parallel", True)
        self.operators: list[Operator] = self._build_operators()

    def _build_operators(self) -> list[Operator]:
        """Instantiate enabled operators in default order."""
        op_configs = self.config.get("operators", {})
        operators: list[Operator] = []

        for name in DEFAULT_ORDER:
            oc = op_configs.get(name, {})
            if not oc.get("enabled", True):
                log.info(f"  [{name}] disabled — skipping")
                continue
            # Separate 'enabled' from operator-specific kwargs
            config_kwargs = {k: v for k, v in oc.items() if k != "enabled"}
            try:
                op = _import_operator(name, config_kwargs)
                operators.append(op)
                log.info(f"  [{name}] loaded")
            except Exception as e:
                log.error(f"  [{name}] failed to load: {e}")
                if self.on_error == "fail_fast":
                    raise

        return operators

    def _build_stages(self) -> list[list[Operator]]:
        """Group operators into parallel stages based on dependencies.

        Returns a list of stages; operators within a stage can run in
        parallel, stages execute sequentially.
        """
        enabled_names = {op.name for op in self.operators}
        op_by_name = {op.name: op for op in self.operators}
        placed: set[str] = set()
        stages: list[list[Operator]] = []

        remaining = list(self.operators)
        while remaining:
            # Operators whose dependencies are all satisfied
            ready = [
                op for op in remaining
                if DEPENDENCIES.get(op.name, set()).intersection(enabled_names)
                .issubset(placed)
            ]
            if not ready:
                # Safety: break circular deps by forcing next operator
                ready = [remaining[0]]
            stages.append(ready)
            for op in ready:
                placed.add(op.name)
                remaining.remove(op)

        return stages

    def _run_operator(
        self, op: Operator, episode_dir: Path,
    ) -> OperatorResult:
        """Run a single operator with timing."""
        log.info(f"▶ {op.name} on {episode_dir.name}")
        t0 = time.time()
        result = op.run(episode_dir)
        elapsed = time.time() - t0
        result.metrics["_elapsed_sec"] = round(elapsed, 2)

        if result.status == "ok":
            log.info(f"  ✓ {op.name} completed ({elapsed:.1f}s)")
        else:
            log.error(f"  ✗ {op.name} failed: {result.errors}")
        return result

    def run(self, episode_dir: Path) -> list[OperatorResult]:
        """Run all enabled operators on one episode directory.

        Independent operators execute in parallel; operators with
        dependencies wait for their prerequisites to finish first.
        """
        results: list[OperatorResult] = []
        stages = self._build_stages()
        failed = False

        for stage in stages:
            if failed and self.on_error == "fail_fast":
                break

            if self.parallel and len(stage) > 1:
                # Parallel execution within stage
                log.info(
                    f"  ║ parallel stage: "
                    f"{', '.join(op.name for op in stage)}"
                )
                with ThreadPoolExecutor(max_workers=len(stage)) as pool:
                    futures = {
                        pool.submit(self._run_operator, op, episode_dir): op
                        for op in stage
                    }
                    for fut in as_completed(futures):
                        result = fut.result()
                        results.append(result)
                        if result.status == "error":
                            failed = True
            else:
                # Sequential execution (single op or parallel disabled)
                for op in stage:
                    result = self._run_operator(op, episode_dir)
                    results.append(result)
                    if result.status == "error":
                        failed = True
                        if self.on_error == "fail_fast":
                            log.error("  Pipeline stopped (on_error=fail_fast)")
                            break

        return results

    def run_all(self, episode_dirs: list[Path]) -> dict[str, Any]:
        """Run pipeline on multiple episodes, return summary report."""
        all_results: dict[str, list[dict]] = {}
        total_t0 = time.time()

        for ep_dir in episode_dirs:
            log.info(f"{'─' * 60}")
            log.info(f"Episode: {ep_dir.name}")
            results = self.run(ep_dir)
            all_results[ep_dir.name] = [asdict(r) for r in results]

        total_elapsed = time.time() - total_t0

        # Summary statistics
        ok = sum(
            1 for ep in all_results.values()
            for r in ep if r["status"] == "ok"
        )
        err = sum(
            1 for ep in all_results.values()
            for r in ep if r["status"] == "error"
        )

        summary = {
            "total_episodes": len(episode_dirs),
            "operators_ok": ok,
            "operators_error": err,
            "total_elapsed_sec": round(total_elapsed, 2),
            "episodes": all_results,
        }
        return summary


# ── Episode discovery ──────────────────────────────────────────────


def _resolve_episodes(config: dict[str, Any]) -> list[Path]:
    """Resolve episode directories from config."""
    episodes: list[Path] = []

    # Explicit list (accept string or list)
    raw = config.get("episodes", [])
    if isinstance(raw, str):
        raw = [raw]
    for ep in raw:
        p = Path(ep)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_dir():
            episodes.append(p)
        else:
            log.warning(f"Episode directory not found: {p}")

    # Auto-discover under episode_root
    root = config.get("episode_root")
    if root:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = PROJECT_ROOT / root_path
        if root_path.is_dir():
            for d in sorted(root_path.iterdir()):
                if d.is_dir() and (d / "rgb.mp4").exists():
                    episodes.append(d)
        else:
            log.warning(f"episode_root not found: {root_path}")

    return episodes


# ── CLI ────────────────────────────────────────────────────────────


def _print_summary(summary: dict[str, Any]) -> None:
    """Print human-readable summary to stdout."""
    print(f"\n{'═' * 60}")
    print(f"Pipeline Summary")
    print(f"{'═' * 60}")
    print(f"  Episodes processed : {summary['total_episodes']}")
    print(f"  Operators OK       : {summary['operators_ok']}")
    print(f"  Operators Error    : {summary['operators_error']}")
    print(f"  Total time         : {summary['total_elapsed_sec']:.1f}s")
    print(f"{'═' * 60}")

    for ep_name, results in summary["episodes"].items():
        print(f"\n  {ep_name}:")
        for r in results:
            icon = "✓" if r["status"] == "ok" else "✗"
            elapsed = r["metrics"].get("_elapsed_sec", "?")
            print(f"    {icon} {r['operator']:20s}  {r['status']:6s}  ({elapsed}s)")
            if r["errors"]:
                for err in r["errors"]:
                    print(f"      └─ {err}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Ego-X Operator Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --config pipeline_config.yaml
  python pipeline.py --episode caption/FusionX-Multimodal-Sample-Data-V2/waterpour1
  python pipeline.py --config pipeline_config.yaml --report results.json
  python pipeline.py --dry-run
""",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("pipeline_config.yaml"),
        help="Pipeline YAML config file (default: pipeline_config.yaml)",
    )
    parser.add_argument(
        "--episode", type=Path, default=None,
        help="Process a single episode directory (overrides config episodes)",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Save JSON report to this path",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show which operators would run, without executing",
    )
    args = parser.parse_args()

    # Load config
    if args.config.exists():
        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    else:
        if not args.episode:
            parser.error(f"Config file not found: {args.config} (use --episode for single episode)")
        config = {}

    # Build pipeline
    log.info("Building pipeline...")
    pipeline = Pipeline(config)

    if not pipeline.operators:
        log.warning("No operators enabled — nothing to do.")
        return

    # Resolve episodes
    if args.episode:
        ep = args.episode if args.episode.is_absolute() else PROJECT_ROOT / args.episode
        episode_dirs = [ep]
    else:
        episode_dirs = _resolve_episodes(config)

    if not episode_dirs:
        log.warning("No episode directories found — check config.")
        return

    # Dry run
    if args.dry_run:
        stages = pipeline._build_stages()
        print("\n[Dry Run] Pipeline plan:")
        print(f"  Episodes: {[str(e) for e in episode_dirs]}")
        print(f"  Parallel: {pipeline.parallel}")
        print(f"  Error mode: {pipeline.on_error}")
        print(f"  Stages ({len(stages)}):")
        for i, stage in enumerate(stages):
            names = [op.name for op in stage]
            mode = "parallel" if len(names) > 1 and pipeline.parallel else "sequential"
            print(f"    Stage {i + 1} ({mode}): {', '.join(names)}")
        print()
        return

    # Execute
    summary = pipeline.run_all(episode_dirs)
    _print_summary(summary)

    # Save report
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(f"Report saved to {args.report}")

    # Exit code
    if summary["operators_error"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
