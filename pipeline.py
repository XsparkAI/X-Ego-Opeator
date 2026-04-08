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
import os
import queue
import sys
import threading
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
    "video_segmentation": ("operators.caption.op_impl",         "SegmentationOperator",   "SegmentationConfig"),
    "segment_cut":        ("operators.segment_cut.op_impl",     "SegmentCutOperator",     "SegmentCutConfig"),
    "video_quality":      ("operators.video_quality.op_impl",   "VideoQualityOperator",   "VideoQualityConfig"),
    "hand_analysis":      ("operators.hand.op_impl",            "HandAnalysisOperator",   "HandAnalysisConfig"),
    "privacy_blur":       ("operators.privacy_blur.op_impl",    "PrivacyBlurOperator",    "PrivacyBlurConfig"),
    "transcode":          ("operators.transcode.op_impl",       "TranscodeOperator",      "TranscodeConfig"),
}

DEFAULT_ORDER = [
    "video_segmentation",   # 1. VLM 分段标注（输出 caption_v2t.json）
    "segment_cut",          # 2. 按标注裁剪短视频（fan-out 到 segments/ 子目录）
    "video_quality",        # 3. 视频质量检测（在短片段上运行）
    "hand_analysis",        # 4. 手部分析（method: yolo | vlm）
    "privacy_blur",         # 5. 隐私模糊
    "transcode",            # 6. 转码
]

# Dependency graph: operator → set of operators it must wait for.
# Operators not listed here (or with empty sets) have no dependencies.
DEPENDENCIES: dict[str, set[str]] = {
    "segment_cut":    {"video_segmentation"},  # needs caption_v2t.json
    "video_quality":  {"segment_cut"},         # runs on segment clips when available
    "hand_analysis":  {"segment_cut"},         # runs on segment clips when available
    "privacy_blur":   {"segment_cut"},         # runs on segment clips when available
    "transcode":      {"privacy_blur"},        # prefers rgb_blurred.mp4
}
# NOTE: When segment_cut is disabled, _build_stages() ignores these
# dependencies (intersects with enabled_names), so quality/hand/blur
# run immediately — backward compatible.


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

        # Propagate API key from config to env (env var takes precedence)
        api_key = config.get("dashscope_api_key", "")
        if api_key and not os.environ.get("DASHSCOPE_API_KEY"):
            os.environ["DASHSCOPE_API_KEY"] = api_key

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

    def _run_stage(
        self, stage: list[Operator], work_dir: Path,
    ) -> list[OperatorResult]:
        """Run all operators in a stage on a single directory."""
        results: list[OperatorResult] = []

        if self.parallel and len(stage) > 1:
            log.info(
                f"  ║ parallel stage: "
                f"{', '.join(op.name for op in stage)}"
            )
            with ThreadPoolExecutor(max_workers=len(stage)) as pool:
                futures = {
                    pool.submit(self._run_operator, op, work_dir): op
                    for op in stage
                }
                for fut in as_completed(futures):
                    results.append(fut.result())
        else:
            for op in stage:
                results.append(self._run_operator(op, work_dir))

        return results

    def run(self, episode_dir: Path) -> list[OperatorResult]:
        """Run all enabled operators on one episode directory.

        Independent operators execute in parallel; operators with
        dependencies wait for their prerequisites to finish first.

        Supports fan-out: if an operator returns
        ``metrics["segment_dirs"]``, subsequent stages run on each
        segment directory instead of the original episode directory.
        """
        all_results: list[OperatorResult] = []
        stages = self._build_stages()
        failed = False

        # Active working directories — starts as just the episode.
        # After segment_cut, this expands to segment sub-directories.
        active_dirs: list[Path] = [episode_dir]

        for stage in stages:
            if failed and self.on_error == "fail_fast":
                break

            stage_results: list[OperatorResult] = []
            for work_dir in active_dirs:
                results = self._run_stage(stage, work_dir)
                stage_results.extend(results)
                if any(r.status == "error" for r in results):
                    failed = True
                    if self.on_error == "fail_fast":
                        log.error("  Pipeline stopped (on_error=fail_fast)")
                        break

            all_results.extend(stage_results)

            # Check for fan-out signal from segment_cut
            for r in stage_results:
                seg_dirs = r.metrics.get("segment_dirs")
                if seg_dirs:
                    active_dirs = [Path(d) for d in seg_dirs]
                    log.info(
                        f"  ⤷ fan-out: {len(active_dirs)} segment directories"
                    )
                    break

        return all_results

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
        return _build_summary(episode_dirs, all_results, total_elapsed)

    def _build_pipeline_stages(self) -> list[list[Operator]]:
        """Build stages for pipeline/streaming mode.

        Unlike ``_build_stages()`` which groups independent operators into
        a single parallel stage, this method creates one stage per operator
        (in dependency-safe order) so that different episodes can overlap
        at the operator level:

            Episode 1: [quality] → [hand] → [blur]
            Episode 2:    [quality] → [hand] → [blur]
            Episode 3:       [quality] → [hand] → [blur]
        """
        enabled_names = {op.name for op in self.operators}
        placed: set[str] = set()
        stages: list[list[Operator]] = []

        remaining = list(self.operators)
        while remaining:
            # Pick operators whose dependencies are satisfied
            ready = [
                op for op in remaining
                if DEPENDENCIES.get(op.name, set()).intersection(enabled_names)
                .issubset(placed)
            ]
            if not ready:
                ready = [remaining[0]]
            # In pipeline mode, emit each ready operator as its own stage
            for op in ready:
                stages.append([op])
                placed.add(op.name)
                remaining.remove(op)

        return stages

    # ── Pipeline (streaming) mode ─────────────────────────────────

    def run_all_pipeline(
        self,
        episode_dirs: list[Path],
        stage_workers: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run pipeline on multiple episodes in streaming/pipeline mode.

        Instead of completing all operators for episode N before starting
        episode N+1, each operator gets its own worker pool and episodes
        flow through operators like an assembly line:

            Episode 1 → [quality] → [hand] → [blur] → done
            Episode 2 →    [quality] → [hand] → [blur] → done
            Episode 3 →       [quality] → [hand] → [blur] → done

        This overlaps I/O-bound and GPU-bound operators across episodes.

        Args:
            episode_dirs: Directories to process.
            stage_workers: Number of worker threads per stage.
                If None, defaults to 1 worker per stage.
        """
        stages = self._build_pipeline_stages()
        n_stages = len(stages)

        if stage_workers is None:
            stage_workers = [1] * n_stages
        elif len(stage_workers) < n_stages:
            # Pad with 1s if user provided fewer values than stages
            stage_workers = list(stage_workers) + [1] * (n_stages - len(stage_workers))

        log.info(
            f"Pipeline streaming mode: {n_stages} stages, "
            f"workers={stage_workers[:n_stages]}"
        )
        for i, stage in enumerate(stages):
            names = [op.name for op in stage]
            log.info(f"  Stage {i}: {', '.join(names)} (×{stage_workers[i]} workers)")

        # Per-episode results accumulator
        _SENTINEL = object()
        lock = threading.Lock()
        all_results: dict[str, list[dict]] = {}  # ep_name → list of result dicts
        completed_count = 0

        def _run_stage_ops(
            stage_ops: list[Operator], work_dir: Path,
        ) -> list[OperatorResult]:
            """Run all operators in a single stage for one directory."""
            results: list[OperatorResult] = []
            if self.parallel and len(stage_ops) > 1:
                with ThreadPoolExecutor(max_workers=len(stage_ops)) as pool:
                    futures = {
                        pool.submit(self._run_operator, op, work_dir): op
                        for op in stage_ops
                    }
                    for fut in as_completed(futures):
                        results.append(fut.result())
            else:
                for op in stage_ops:
                    results.append(self._run_operator(op, work_dir))

            with lock:
                for r in results:
                    all_results.setdefault(work_dir.name, []).append(asdict(r))
            return results

        def _stage_worker_loop(
            stage_idx: int,
            stage_ops: list[Operator],
            in_q: queue.Queue,
            out_q: queue.Queue | None,
        ) -> None:
            """Worker loop: pull episodes from in_q, process, push to out_q."""
            nonlocal completed_count
            while True:
                item = in_q.get()
                if item is _SENTINEL:
                    if out_q is not None:
                        out_q.put(_SENTINEL)
                    in_q.task_done()
                    break

                work_dir: Path = item
                results = _run_stage_ops(stage_ops, work_dir)

                # Check for fan-out (segment_cut produces segment_dirs)
                fan_out_dirs: list[str] | None = None
                for r in results:
                    seg_dirs = r.metrics.get("segment_dirs")
                    if seg_dirs:
                        fan_out_dirs = seg_dirs
                        break

                if out_q is not None:
                    if fan_out_dirs:
                        # Fan-out: push each segment dir to next stage
                        for d in fan_out_dirs:
                            out_q.put(Path(d))
                    else:
                        out_q.put(work_dir)
                else:
                    # Last stage — work unit fully done
                    with lock:
                        completed_count += 1
                    log.info(
                        f"  ✓ [{completed_count}] "
                        f"{work_dir.name} complete"
                    )

                in_q.task_done()

        # Build inter-stage queues
        queues = [queue.Queue() for _ in range(n_stages)]

        # Launch worker threads per stage
        total_t0 = time.time()
        threads_by_stage: list[list[threading.Thread]] = []

        for stage_idx, stage_ops in enumerate(stages):
            in_q = queues[stage_idx]
            out_q = queues[stage_idx + 1] if stage_idx + 1 < n_stages else None
            n_workers = stage_workers[stage_idx]
            stage_threads: list[threading.Thread] = []

            for _ in range(n_workers):
                t = threading.Thread(
                    target=_stage_worker_loop,
                    args=(stage_idx, stage_ops, in_q, out_q),
                    daemon=True,
                )
                t.start()
                stage_threads.append(t)
            threads_by_stage.append(stage_threads)

        # Feed all episodes into stage 0
        for ep_dir in episode_dirs:
            queues[0].put(ep_dir)

        # Send sentinels to stage 0 (one per worker)
        for _ in range(stage_workers[0]):
            queues[0].put(_SENTINEL)

        # Drain stages sequentially: wait for stage K workers to finish,
        # then inject extra sentinels if stage K+1 has more workers.
        for stage_idx in range(n_stages):
            for t in threads_by_stage[stage_idx]:
                t.join()

            # Stage K sent prev_workers sentinels to stage K+1.
            # If stage K+1 has more workers, inject the delta.
            if stage_idx + 1 < n_stages:
                sent = stage_workers[stage_idx]
                needed = stage_workers[stage_idx + 1]
                for _ in range(max(0, needed - sent)):
                    queues[stage_idx + 1].put(_SENTINEL)

        total_elapsed = time.time() - total_t0
        return _build_summary(episode_dirs, all_results, total_elapsed)


def _build_summary(
    episode_dirs: list[Path],
    all_results: dict[str, list[dict]],
    total_elapsed: float,
) -> dict[str, Any]:
    """Build the standard pipeline summary report."""
    ok = sum(
        1 for ep in all_results.values()
        for r in ep if r["status"] == "ok"
    )
    err = sum(
        1 for ep in all_results.values()
        for r in ep if r["status"] == "error"
    )
    return {
        "total_episodes": len(episode_dirs),
        "operators_ok": ok,
        "operators_error": err,
        "total_elapsed_sec": round(total_elapsed, 2),
        "episodes": all_results,
    }


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
        "--mode", choices=["sequential", "pipeline"], default=None,
        help="Execution mode: sequential (default, one episode at a time) "
             "or pipeline (streaming, episodes overlap across stages)",
    )
    parser.add_argument(
        "--stage-workers", type=str, default=None,
        help="Pipeline mode: comma-separated workers per stage, e.g. '2,2,4'. "
             "Defaults to 1 per stage.",
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

    # Resolve execution mode: CLI flag > config > default
    exec_mode = args.mode or config.get("execution_mode", "sequential")

    # Parse stage workers
    stage_workers: list[int] | None = None
    sw_raw = args.stage_workers or config.get("stage_workers")
    if sw_raw:
        if isinstance(sw_raw, str):
            stage_workers = [int(x) for x in sw_raw.split(",")]
        elif isinstance(sw_raw, list):
            stage_workers = [int(x) for x in sw_raw]

    # Dry run
    if args.dry_run:
        if exec_mode == "pipeline":
            stages = pipeline._build_pipeline_stages()
        else:
            stages = pipeline._build_stages()
        print("\n[Dry Run] Pipeline plan:")
        print(f"  Episodes: {[str(e) for e in episode_dirs]}")
        print(f"  Parallel: {pipeline.parallel}")
        print(f"  Execution mode: {exec_mode}")
        print(f"  Error mode: {pipeline.on_error}")
        print(f"  Stages ({len(stages)}):")
        for i, stage in enumerate(stages):
            names = [op.name for op in stage]
            mode = "parallel" if len(names) > 1 and pipeline.parallel else "sequential"
            workers = (stage_workers[i] if stage_workers and i < len(stage_workers)
                       else 1) if exec_mode == "pipeline" else "-"
            print(f"    Stage {i + 1} ({mode}): {', '.join(names)}"
                  f"  [workers={workers}]")
        print()
        return

    # Execute
    if exec_mode == "pipeline" and len(episode_dirs) > 1:
        summary = pipeline.run_all_pipeline(episode_dirs, stage_workers)
    else:
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
