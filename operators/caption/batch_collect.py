#!/usr/bin/env python3
"""
Batch collect: poll DashScope Batch API, download results, assemble video segments.

Usage:
  python batch_collect.py                    # Poll until done, then process
  python batch_collect.py --status           # Check status only
  python batch_collect.py --preview          # Also generate preview videos
  python batch_collect.py --no-refine        # Skip VLM refine step (v2t_desc only)
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import segment_v2t as _sv
from segment_v2t import (
    API_KEY, SCRIPT_DIR, SOP_DIR, TASK_SOP_MAP,
    build_segments_via_cuts, generate_preview,
    get_task_type, Window,
)
from scene_classifier import classify_video_scene
try:
    from ..video_path import resolve_episode_video_path
except ImportError:
    from video_path import resolve_episode_video_path

PROJECT_ROOT = SCRIPT_DIR.parent.parent
BATCH_TMP_DIR = PROJECT_ROOT / "tmp" / "batch"
BATCH_STATE_FILE = BATCH_TMP_DIR / "batch_state.json"
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "test_video" / "FusionX-Multimodal-Sample-Data-V2"


def _load_state():
    if not BATCH_STATE_FILE.exists():
        log.error(f"No state file at {BATCH_STATE_FILE}. Run batch_submit.py first.")
        return None
    return json.loads(BATCH_STATE_FILE.read_text())


def _poll_batch(client, batch_id, interval=30):
    """Poll until batch reaches a terminal state."""
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = ""
        if hasattr(batch, "request_counts") and batch.request_counts:
            rc = batch.request_counts
            counts = f" (done={rc.completed}, fail={rc.failed}, total={rc.total})"
        log.info(f"Status: {batch.status}{counts}")
        if batch.status in terminal:
            return batch
        time.sleep(interval)


def _download_results(client, batch):
    """Download output JSONL and parse into {custom_id: result} map."""
    results = {}

    if batch.output_file_id:
        out_path = BATCH_TMP_DIR / "batch_output.jsonl"
        content = client.files.content(batch.output_file_id)
        content.write_to_file(str(out_path))
        log.info(f"Output: {out_path}")

        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj["custom_id"]
            resp = obj.get("response", {})
            body = resp.get("body", {})

            text = None
            if body.get("choices"):
                text = body["choices"][0].get("message", {}).get("content", "")

            results[cid] = {
                "text": text,
                "usage": body.get("usage"),
                "status_code": resp.get("status_code", 200),
            }

    if batch.error_file_id:
        err_path = BATCH_TMP_DIR / "batch_errors.jsonl"
        content = client.files.content(batch.error_file_id)
        content.write_to_file(str(err_path))
        log.info(f"Errors: {err_path}")

    return results


def _parse_vlm_json(text):
    """Extract JSON object from VLM text output."""
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _process_episode(episode_name, ep_state, result_map, sop):
    """Run clustering + IMU snap for one episode. Returns (segments, fps)."""
    fps = ep_state["fps"]
    nframes = ep_state["nframes"]
    valleys = np.array(ep_state["imu_valleys"]) if ep_state["imu_valleys"] else None

    windows = [
        Window(w["window_id"], w["start_frame"], w["end_frame"], w["frame_ids"])
        for w in ep_state["windows"]
    ]

    window_results = []
    for w in windows:
        key = f"{episode_name}__win{w.window_id}"
        r = result_map.get(key)
        parsed = _parse_vlm_json(r["text"]) if r and r.get("text") else None
        window_results.append(parsed)

    n_ok = sum(1 for r in window_results if r is not None)
    log.info(f"[{episode_name}] {n_ok}/{len(windows)} windows returned valid JSON")

    segments = build_segments_via_cuts(windows, window_results, fps, nframes, sop, valleys)
    return segments, fps


def main():
    p = argparse.ArgumentParser(description="Collect batch results and assemble segments")
    p.add_argument("--status", action="store_true",
                   help="Check status only, don't download/process")
    p.add_argument("--preview", action="store_true",
                   help="Generate preview videos")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Poll interval in seconds (default: 30)")
    p.add_argument("--data-root", type=str, default=None,
                   help=f"Data directory (default: {DEFAULT_DATA_ROOT})")
    p.add_argument("--no-refine", action="store_true",
                   help="Skip VLM refine step (v2t_desc variant only)")
    args = p.parse_args()

    state = _load_state()
    if not state:
        return

    batch_id = state.get("batch_id")
    if not batch_id:
        log.error("No batch_id in state (was it a dry-run?)")
        return

    from openai import OpenAI
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # ── Status-only mode ─────────────────────────────────────────────────────
    if args.status:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id}: {batch.status}")
        if hasattr(batch, "request_counts") and batch.request_counts:
            rc = batch.request_counts
            print(f"  Completed: {rc.completed}  Failed: {rc.failed}  Total: {rc.total}")
        return

    # ── Poll until terminal ──────────────────────────────────────────────────
    log.info(f"Polling batch {batch_id} (interval={args.poll_interval}s)...")
    batch = _poll_batch(client, batch_id, interval=args.poll_interval)

    if batch.status != "completed":
        log.error(f"Batch ended with status: {batch.status}")
        return

    # ── Download results ─────────────────────────────────────────────────────
    result_map = _download_results(client, batch)
    log.info(f"Downloaded {len(result_map)} results")

    # ── Process each episode ─────────────────────────────────────────────────
    variant = state.get("variant", "v2t")
    output_suffix = "v2t" if variant == "v2t" else "v2t_desc"

    # Set module globals
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    _sv.DATA_ROOT = data_root
    _sv.OUTPUT_SUFFIX = output_suffix

    total_segments = 0
    in_tok_total, out_tok_total = 0, 0

    # Accumulate token usage from batch results
    for r in result_map.values():
        u = r.get("usage")
        if u:
            in_tok_total += u.get("prompt_tokens", 0) or u.get("input_tokens", 0)
            out_tok_total += u.get("completion_tokens", 0) or u.get("output_tokens", 0)

    for episode_name, ep_state in state["episodes"].items():
        task_type = ep_state.get("task_type") or get_task_type(episode_name)
        sop_file = SOP_DIR / TASK_SOP_MAP.get(task_type, "")
        if not sop_file.exists():
            log.warning(f"[{episode_name}] SOP not found: {sop_file}, skipping")
            continue
        sop = json.loads(sop_file.read_text())

        segments, fps = _process_episode(episode_name, ep_state, result_map, sop)

        # VLM refine for v2t_desc variant (synchronous, 1 call per episode)
        if variant == "v2t_desc" and not args.no_refine:
            try:
                from segment_v2t_desc_only import _vlm_refine_segments
                segments, ref_in, ref_out = _vlm_refine_segments(segments, sop)
                in_tok_total += ref_in
                out_tok_total += ref_out
            except Exception as e:
                log.warning(f"[{episode_name}] Refine failed: {e}, using raw segments")

        # Build caption output
        ep_dir = data_root / episode_name
        scene = classify_video_scene(resolve_episode_video_path(ep_dir), fps=fps, nframes=ep_state["nframes"])
        caption = {
            "scene": scene,
            "tasks": [
                {
                    "instruction": sop["task_name"],
                    "frame_interval": [0, ep_state["nframes"]],
                    "atomic_actions": [
                        {
                            "frame_interval": seg["frame_interval"],
                            "caption": seg["instruction"],
                        }
                        for seg in segments
                    ],
                }
            ],
        }

        out_path = ep_dir / f"caption_{output_suffix}.json"
        out_path.write_text(json.dumps(caption, ensure_ascii=False, indent=2))
        log.info(f"[{episode_name}] Saved: {out_path}")

        # Print segment summary
        print(f"\n{'─' * 60}")
        task = caption["tasks"][0]
        print(f"  {episode_name}: {task['instruction']}")
        print(f"  Scene: {caption.get('scene', 'unknown')}")
        print(f"  Segments: {len(task['atomic_actions'])}")
        for i, a in enumerate(task["atomic_actions"], start=1):
            s, e = a["frame_interval"]
            dur = (e - s) / fps
            print(f"    Step {i}: [{s:>4d}, {e:>4d}] ({dur:5.1f}s) — {a['caption']}")
        print(f"{'─' * 60}")
        total_segments += len(segments)

        if args.preview:
            preview_path = generate_preview(ep_dir, caption, fps)
            log.info(f"[{episode_name}] Preview: {preview_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Episodes: {len(state['episodes'])}")
    print(f"  Segments: {total_segments}")
    print(f"  Tokens: {in_tok_total:,} in / {out_tok_total:,} out")
    print(f"  Batch API — 50% cost vs real-time")
    print(f"{'═' * 60}")

    # ── Cleanup batch temp files ────────────────────────────────────────────
    import shutil
    if BATCH_TMP_DIR.exists():
        shutil.rmtree(BATCH_TMP_DIR)
        log.info(f"Cleaned up batch temp dir: {BATCH_TMP_DIR}")


if __name__ == "__main__":
    main()
