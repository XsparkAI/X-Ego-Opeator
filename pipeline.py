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
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

# Make in-repo operator packages importable when running ``python pipeline.py`` directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from operators.operator_base import Operator, OperatorResult  # noqa: E402
from operators.video_path import episode_has_input_video  # noqa: E402
from operators.vlm_limit import get_cpu_global_limit, set_cpu_global_limit, set_vlm_global_limit  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Maps operator name -> (module_path, class_name, config_class_name).
OPERATOR_REGISTRY: dict[str, tuple[str, str, str]] = {
    "frame_cache":        ("operators.frame_cache.op_impl",      "FrameCacheOperator",    "FrameCacheConfig"),
    "video_segmentation": ("operators.caption.op_impl",         "SegmentationOperator",   "SegmentationConfig"),
    "segment_cut":        ("operators.segment_cut.op_impl",     "SegmentCutOperator",     "SegmentCutConfig"),
    "video_quality":      ("operators.video_quality.op_impl",   "VideoQualityOperator",   "VideoQualityConfig"),
    "hand_analysis":      ("operators.hand.op_impl",            "HandAnalysisOperator",   "HandAnalysisConfig"),
    "privacy_blur":       ("operators.privacy_blur.op_impl",    "PrivacyBlurOperator",    "PrivacyBlurConfig"),
    "transcode":          ("operators.transcode.op_impl",       "TranscodeOperator",      "TranscodeConfig"),
}

DEFAULT_ORDER = [
    "frame_cache",          # 0. 预抽取稀疏帧缓存（供 caption/VLM hand 复用）
    "video_segmentation",   # 1. VLM 分段标注（输出 caption_v2t.json）
    "segment_cut",          # 2. 按标注裁剪短视频（fan-out 到 segments/ 子目录）
    "video_quality",        # 3. 视频质量检测（在短片段上运行）
    "hand_analysis",        # 4. 手部分析（method: yolo | vlm）
    "privacy_blur",         # 5. 隐私模糊
    "transcode",            # 6. 转码
]

# Dependency graph: operator → set of operators it must wait for.
# Dependencies are evaluated only among enabled operators.
DEPENDENCIES: dict[str, set[str]] = {
    "video_segmentation": {"frame_cache"},
    "segment_cut": {"video_segmentation"},   # reads caption_v2t.json
    "video_quality": {"frame_cache"},        # can reuse cached quality frames
    "hand_analysis": {"frame_cache"},        # both backends can start from the original video
    "privacy_blur": set(),                    # runs on the current work directory directly
    "transcode": {"privacy_blur"},           # prefers rgb_blurred.mp4 when available
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

        # Allow YAML config to supply the API key without overriding an existing env var.
        api_key = config.get("dashscope_api_key", "")
        if api_key and not os.environ.get("DASHSCOPE_API_KEY"):
            os.environ["DASHSCOPE_API_KEY"] = api_key
        vlm_api_key = config.get("vlm_api_key", "")
        if vlm_api_key and not os.environ.get("VLM_API_KEY"):
            os.environ["VLM_API_KEY"] = vlm_api_key
        ark_api_key = config.get("ark_api_key", "")
        if ark_api_key and not os.environ.get("ARK_API_KEY"):
            os.environ["ARK_API_KEY"] = ark_api_key
        vlm_provider = str(config.get("vlm_api_provider", "")).strip()
        if vlm_provider and not os.environ.get("VLM_API_PROVIDER"):
            os.environ["VLM_API_PROVIDER"] = vlm_provider
        vlm_base_url = str(config.get("vlm_base_url", "")).strip()
        if vlm_base_url and not os.environ.get("VLM_BASE_URL"):
            os.environ["VLM_BASE_URL"] = vlm_base_url
        ark_base_url = str(config.get("ark_base_url", "")).strip()
        if ark_base_url and not os.environ.get("ARK_BASE_URL"):
            os.environ["ARK_BASE_URL"] = ark_base_url
        vlm_default_model = str(config.get("vlm_default_model", "")).strip()
        if vlm_default_model and not os.environ.get("VLM_DEFAULT_MODEL"):
            os.environ["VLM_DEFAULT_MODEL"] = vlm_default_model
        caption_model = str(config.get("vlm_caption_model", "")).strip()
        if caption_model and not os.environ.get("VLM_CAPTION_MODEL"):
            os.environ["VLM_CAPTION_MODEL"] = caption_model
        hand_model = str(config.get("vlm_hand_model", "")).strip()
        if hand_model and not os.environ.get("VLM_HAND_MODEL"):
            os.environ["VLM_HAND_MODEL"] = hand_model
        scene_model = str(config.get("vlm_scene_model", "")).strip()
        if scene_model and not os.environ.get("VLM_SCENE_MODEL"):
            os.environ["VLM_SCENE_MODEL"] = scene_model
        input_video_path = str(config.get("input_video_path", "rgb.mp4")).strip() or "rgb.mp4"
        os.environ["EGOX_INPUT_VIDEO_PATH"] = input_video_path
        set_cpu_global_limit(config.get("cpu_global_max_concurrency", 2))
        set_vlm_global_limit(config.get("vlm_global_max_concurrency", 1))

        self.operators: list[Operator] = self._build_operators()

    def _build_operators(self) -> list[Operator]:
        """Instantiate enabled operators in default order."""
        op_configs = self.config.get("operators", {})
        operators: list[Operator] = []
        segmentation_enabled = op_configs.get("video_segmentation", {}).get("enabled", True)

        for name in DEFAULT_ORDER:
            oc = op_configs.get(name, {})
            if not oc.get("enabled", True):
                log.info(f"  [{name}] disabled — skipping")
                continue
            if name == "segment_cut" and not segmentation_enabled:
                log.info("  [segment_cut] skipped because video_segmentation is disabled")
                continue
            # ``enabled`` controls pipeline selection only; the dataclass config should not receive it.
            config_kwargs = {k: v for k, v in oc.items() if k != "enabled"}
            if name == "frame_cache":
                config_kwargs = self._build_frame_cache_config(op_configs, config_kwargs)
            elif name == "hand_analysis":
                config_kwargs = self._build_hand_analysis_config(config_kwargs)
            try:
                op = _import_operator(name, config_kwargs)
                operators.append(op)
                log.info(f"  [{name}] loaded")
            except Exception as e:
                log.error(f"  [{name}] failed to load: {e}")
                if self.on_error == "fail_fast":
                    raise

        return operators

    def _build_frame_cache_config(
        self,
        op_configs: dict[str, Any],
        config_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        seg_cfg = op_configs.get("video_segmentation", {})
        hand_cfg = op_configs.get("hand_analysis", {})

        config_kwargs.setdefault("include_caption", seg_cfg.get("enabled", True))
        config_kwargs.setdefault("caption_window_sec", seg_cfg.get("window_sec", 10.0))
        config_kwargs.setdefault("caption_step_sec", seg_cfg.get("step_sec", 5.0))
        config_kwargs.setdefault("caption_frames_per_window", seg_cfg.get("frames_per_window", 12))
        config_kwargs.setdefault(
            "include_hand_vlm",
            hand_cfg.get("enabled", True)
            and hand_cfg.get("method", "yolo").lower() == "vlm",
        )
        config_kwargs.setdefault(
            "hand_frame_step",
            self.config.get("hand_vlm_sample_frame_step", hand_cfg.get("frame_step", 120)),
        )
        quality_cfg = op_configs.get("video_quality", {})
        config_kwargs.setdefault(
            "include_video_quality",
            quality_cfg.get("enabled", True)
            and quality_cfg.get("sample_fps") is not None,
        )
        config_kwargs.setdefault("quality_sample_fps", quality_cfg.get("sample_fps"))
        return config_kwargs

    def _build_hand_analysis_config(self, config_kwargs: dict[str, Any]) -> dict[str, Any]:
        config_kwargs = dict(config_kwargs)
        legacy_frame_step = config_kwargs.pop("frame_step", None)
        method = str(config_kwargs.get("method", "yolo")).lower()

        if method == "vlm":
            config_kwargs["vlm_sample_frame_step"] = self.config.get(
                "hand_vlm_sample_frame_step",
                legacy_frame_step if legacy_frame_step is not None else 120,
            )
        else:
            config_kwargs.setdefault(
                "yolo_frame_step",
                legacy_frame_step if legacy_frame_step is not None else 1,
            )
        return config_kwargs

    @staticmethod
    def _cleanup_episode_cache(episode_dir: Path) -> None:
        cache_dir = episode_dir / ".frame_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            log.info(f"  cleaned frame cache: {cache_dir}")

    def _build_stages(self) -> list[list[Operator]]:
        """Group enabled operators into dependency-safe parallel stages."""
        enabled_names = {op.name for op in self.operators}
        op_by_name = {op.name: op for op in self.operators}
        placed: set[str] = set()
        stages: list[list[Operator]] = []

        remaining = list(self.operators)
        while remaining:
            # Operators in the same stage can run together once all enabled prerequisites are placed.
            ready = [
                op for op in remaining
                if DEPENDENCIES.get(op.name, set()).intersection(enabled_names)
                .issubset(placed)
            ]
            if not ready:
                # Defensive fallback for malformed dependency graphs.
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
        elif result.status == "pending":
            log.info(f"  … {op.name} submitted batch work ({elapsed:.1f}s)")
        else:
            log.error(f"  ✗ {op.name} failed: {result.errors}")
        return result

    def _collect_pending(
        self,
        op: Operator,
        work_dir: Path,
    ) -> OperatorResult:
        if not hasattr(op, "collect"):
            return OperatorResult(
                status="error",
                operator=op.name,
                errors=[f"{op.name} returned pending but does not implement collect()"],
            )
        log.info(f"⏳ waiting for pending {op.name} on {work_dir.name}")
        t0 = time.time()
        result = op.collect(work_dir)
        elapsed = time.time() - t0
        result.metrics["_elapsed_sec"] = round(elapsed, 2)
        if result.status == "ok":
            log.info(f"  ✓ {op.name} collected ({elapsed:.1f}s)")
        else:
            log.error(f"  ✗ {op.name} collect failed: {result.errors}")
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
        op_by_name = {op.name: op for op in self.operators}
        pending_ops: dict[tuple[Path, str], OperatorResult] = {}

        # Starts at the episode root; segment_cut can fan this out into per-segment work dirs.
        active_dirs: list[Path] = [episode_dir]

        try:
            for stage in stages:
                if failed and self.on_error == "fail_fast":
                    break

                stage_results: list[OperatorResult] = []
                for work_dir in active_dirs:
                    required_pending = {
                        dep_name
                        for op in stage
                        for dep_name in DEPENDENCIES.get(op.name, set())
                        if (work_dir, dep_name) in pending_ops
                    }
                    for dep_name in sorted(required_pending):
                        collect_result = self._collect_pending(op_by_name[dep_name], work_dir)
                        pending_ops.pop((work_dir, dep_name), None)
                        stage_results.append(collect_result)
                        all_results.append(collect_result)
                        if collect_result.status == "error":
                            failed = True
                            if self.on_error == "fail_fast":
                                log.error("  Pipeline stopped (on_error=fail_fast)")
                                break
                    if failed and self.on_error == "fail_fast":
                        break

                    results = self._run_stage(stage, work_dir)
                    for result in results:
                        if result.status == "pending":
                            pending_ops[(work_dir, result.operator)] = result
                        else:
                            stage_results.append(result)
                            all_results.append(result)
                    if any(r.status == "error" for r in results):
                        failed = True
                        if self.on_error == "fail_fast":
                            log.error("  Pipeline stopped (on_error=fail_fast)")
                            break

                # segment_cut can switch downstream work from the episode root to segment dirs.
                for r in stage_results:
                    seg_dirs = r.metrics.get("segment_dirs")
                    if seg_dirs:
                        active_dirs = [Path(d) for d in seg_dirs]
                        log.info(
                            f"  ⤷ fan-out: {len(active_dirs)} segment directories"
                        )
                        break
            for work_dir, op_name in list(pending_ops):
                collect_result = self._collect_pending(op_by_name[op_name], work_dir)
                pending_ops.pop((work_dir, op_name), None)
                all_results.append(collect_result)
            return all_results
        finally:
            self._cleanup_episode_cache(episode_dir)

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
        for ep_dir in episode_dirs:
            self._cleanup_episode_cache(ep_dir)
        return _build_summary(episode_dirs, all_results, total_elapsed)

    def _build_pipeline_stages(self) -> list[list[Operator]]:
        """Build dependency-safe stages for multi-episode streaming execution.

        Each stage may contain multiple independent operators for one work item,
        while different episodes can overlap across stages.
        """
        enabled_names = {op.name for op in self.operators}
        placed: set[str] = set()
        stages: list[list[Operator]] = []

        remaining = list(self.operators)
        while remaining:
            # Keep operators with the same ready set in the same stage.
            ready = [
                op for op in remaining
                if DEPENDENCIES.get(op.name, set()).intersection(enabled_names)
                .issubset(placed)
            ]
            if not ready:
                ready = [remaining[0]]
            stages.append(ready)
            for op in ready:
                placed.add(op.name)
                remaining.remove(op)

        return stages

    # ── Pipeline (streaming) mode ─────────────────────────────────

    def run_all_pipeline(
        self,
        episode_dirs: list[Path],
        stage_workers: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run multiple episodes in stage-streaming mode.

        A later episode can enter an earlier stage before the previous episode
        has finished the whole pipeline, which improves throughput for batch runs.

        Args:
            episode_dirs: Directories to process.
            stage_workers: Number of worker threads per stage.
                If None, defaults to cpu_global_max_concurrency per stage.
        """
        stages = self._build_pipeline_stages()
        n_stages = len(stages)

        if stage_workers is None:
            stage_workers = [get_cpu_global_limit()] * n_stages
        elif len(stage_workers) < n_stages:
            # Preserve explicit overrides and fall back to one worker for missing entries.
            stage_workers = list(stage_workers) + [1] * (n_stages - len(stage_workers))

        log.info(
            f"Pipeline streaming mode: {n_stages} stages, "
            f"workers={stage_workers[:n_stages]}"
        )
        for i, stage in enumerate(stages):
            names = [op.name for op in stage]
            log.info(f"  Stage {i}: {', '.join(names)} (×{stage_workers[i]} workers)")

        # Episode results are collected under the current work-dir name.
        _SENTINEL = object()
        lock = threading.Lock()
        all_results: dict[str, list[dict]] = {}  # ep_name → list of result dicts
        completed_count = 0
        op_by_name = {op.name: op for op in self.operators}
        pending_ops: dict[tuple[str, str], OperatorResult] = {}

        def _pending_key(work_dir: Path, op_name: str) -> tuple[str, str]:
            return (str(work_dir.resolve()), op_name)

        def _collect_required_pending(stage_ops: list[Operator], work_dir: Path) -> list[OperatorResult]:
            required = {
                dep_name
                for op in stage_ops
                for dep_name in DEPENDENCIES.get(op.name, set())
                if _pending_key(work_dir, dep_name) in pending_ops
            }
            collected: list[OperatorResult] = []
            for dep_name in sorted(required):
                result = self._collect_pending(op_by_name[dep_name], work_dir)
                pending_ops.pop(_pending_key(work_dir, dep_name), None)
                with lock:
                    all_results.setdefault(work_dir.name, []).append(asdict(result))
                collected.append(result)
            return collected

        def _collect_all_pending(work_dir: Path) -> list[OperatorResult]:
            pending_names = [
                op_name
                for key, _result in list(pending_ops.items())
                if key[0] == str(work_dir.resolve())
                for op_name in [key[1]]
            ]
            collected: list[OperatorResult] = []
            for op_name in sorted(set(pending_names)):
                result = self._collect_pending(op_by_name[op_name], work_dir)
                pending_ops.pop(_pending_key(work_dir, op_name), None)
                with lock:
                    all_results.setdefault(work_dir.name, []).append(asdict(result))
                collected.append(result)
            return collected

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
                    if r.status == "pending":
                        pending_ops[_pending_key(work_dir, r.operator)] = r
                    else:
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
                collected = _collect_required_pending(stage_ops, work_dir)
                if any(r.status == "error" for r in collected):
                    if out_q is None:
                        with lock:
                            completed_count += 1
                    in_q.task_done()
                    continue
                results = _run_stage_ops(stage_ops, work_dir)

                # segment_cut forwards per-segment work items instead of the episode root.
                fan_out_dirs: list[str] | None = None
                for r in results:
                    seg_dirs = r.metrics.get("segment_dirs")
                    if seg_dirs:
                        fan_out_dirs = seg_dirs
                        break

                if out_q is not None:
                    if fan_out_dirs:
                        # Fan-out: each segment becomes an independent downstream work item.
                        for d in fan_out_dirs:
                            out_q.put(Path(d))
                    else:
                        out_q.put(work_dir)
                else:
                    _collect_all_pending(work_dir)
                    # Last stage owns final pending collection and completion accounting.
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

        # Drain stages in order and forward enough sentinels for downstream worker counts.
        for stage_idx in range(n_stages):
            for t in threads_by_stage[stage_idx]:
                t.join()

            # Upstream workers already forwarded one sentinel each; add only the missing delta.
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

    # Accept either a single path or a list of episode paths.
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

    # Optionally auto-discover episodes under a root directory.
    root = config.get("episode_root")
    if root:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = PROJECT_ROOT / root_path
        if root_path.is_dir():
            for d in sorted(root_path.iterdir()):
                if d.is_dir() and episode_has_input_video(d):
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
        help="Advanced override: comma-separated workers per stage, e.g. '2,2,4'. "
             "If omitted, each stage defaults to cpu_global_max_concurrency.",
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

    # Instantiate enabled operators and dependency state.
    log.info("Building pipeline...")
    pipeline = Pipeline(config)

    if not pipeline.operators:
        log.warning("No operators enabled — nothing to do.")
        return

    # Resolve input episodes from CLI or config.
    if args.episode:
        ep = args.episode if args.episode.is_absolute() else PROJECT_ROOT / args.episode
        episode_dirs = [ep]
    else:
        episode_dirs = _resolve_episodes(config)

    if not episode_dirs:
        log.warning("No episode directories found — check config.")
        return

    # CLI flag wins over config for execution mode.
    exec_mode = args.mode or config.get("execution_mode", "sequential")

    # Parse optional per-stage worker overrides.
    stage_workers: list[int] | None = None
    sw_raw = args.stage_workers
    if sw_raw:
        if isinstance(sw_raw, str):
            stage_workers = [int(x) for x in sw_raw.split(",")]
        elif isinstance(sw_raw, list):
            stage_workers = [int(x) for x in sw_raw]

    # Dry-run only prints the stage plan.
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
            workers = (
                stage_workers[i] if stage_workers and i < len(stage_workers)
                else get_cpu_global_limit()
            ) if exec_mode == "pipeline" else "-"
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

    # Save report when requested.
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(f"Report saved to {args.report}")

    # Surface operator failures as a non-zero process exit.
    if summary["operators_error"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
