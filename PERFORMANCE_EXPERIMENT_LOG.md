# Performance Experiment Log

## Overview

This document records the code changes, benchmark setup, intermediate failed experiments, and final retained optimizations from this round.

## Environment

- Conda env: `operator`
- Python: `/home/pc/miniconda3/envs/operator/bin/python`
- NumPy: `2.2.6`
- OpenCV: `4.13.0`
- Torch: `2.9.1+cu128`
- CUDA available: `False`

Implication:

- This round is CPU-path only.
- `privacy_blur` GPU-path performance was not re-benchmarked in this round.
- No external VLM API benchmark was run; only local frame extraction and cache paths were measured.

## Code Changes

### 1. Video rotation metadata caching

File:

- [operators/video_utils.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/video_utils.py)

Change:

- Added `lru_cache` around `get_manual_rotation()` using resolved path + file stat.

Reason:

- Avoid repeated `VideoCapture` and `ffprobe`-like metadata work across operators.

### 2. Shared frame extraction helpers

File:

- [operators/video_utils.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/video_utils.py)

Change:

- Added:
  - `extract_frames_sequential()`
  - `extract_frames_random_seek()`
  - `extract_frames_auto()`

Reason:

- Centralize extraction strategy instead of each operator duplicating frame access logic.

### 3. Incremental frame cache reuse

File:

- [operators/frame_cache/cache_utils.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/frame_cache/cache_utils.py)

Change:

- `build_cache()` / `build_quality_cache()` now reuse already-materialized frame files.
- Missing frames are filled incrementally instead of rebuilding the whole profile every run.

Reason:

- Before this change, repeated runs on the same episode still paid nearly full cache rebuild cost.

### 4. Caption / hand extraction switched to shared auto strategy

Files:

- [operators/caption/segment_v2t.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/caption/segment_v2t.py)
- [operators/hand/vlm_hand_audit.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/hand/vlm_hand_audit.py)

Change:

- Replaced duplicated `CAP_PROP_POS_FRAMES` loops with `extract_frames_auto()`.

Reason:

- Keep sparse sampling competitive while still allowing sequential scan for denser workloads.

## Validation

Syntax regression check:

```bash
conda run -n operator python -m py_compile \
  operators/video_utils.py \
  operators/frame_cache/cache_utils.py \
  operators/caption/segment_v2t.py \
  operators/hand/vlm_hand_audit.py
```

Result:

- Passed.

## Benchmarks

## Benchmark 1: `frame_cache` before optimization

Input:

- `tmp/test_episode_30s/rgb.mp4`

Command shape:

```bash
conda run -n operator python -c "... FrameCacheOperator(...).run(...) ..."
```

Results on old code:

- Run 1: `70.081s`
- Run 2: `70.242s`

Observation:

- Repeated runs were effectively just as expensive as first runs.
- Existing cache files were not being reused in a meaningful way.

## Benchmark 2: `frame_cache` after optimization

Input:

- Clean episode copy with only `rgb.mp4`:
  - `tmp/perf_frame_cache_clean/rgb.mp4`

Results on new code:

- First run: `58.571s`
- Second run: `0.083s`

Interpretation:

- First-run improvement vs old code: about `16.4%`
- Repeat-run improvement vs old code: effectively `>99%`

Conclusion:

- Incremental cache reuse is the clearest win from this round.

## Benchmark 3: `video_quality` without frame cache

Input:

- `tmp/test_episode_30s/rgb.mp4`

Command:

```bash
conda run -n operator python -c "... process_video(... sample_fps=2.0 ...) ..."
```

Result:

- Total: `41.106s`
- Read frames: `37.67s`
- Frames processed: `60`

Observation:

- The major bottleneck was frame reading, not the quality algorithms themselves.

## Benchmark 4: `video_quality` with warmed frame cache

Input:

- `tmp/perf_frame_cache_clean/rgb.mp4`
- Frame cache already built by Benchmark 2

Result:

- Total: `4.663s`
- Read frames: `1.29s`
- Frames processed: `60`

Improvement vs no-cache baseline:

- End-to-end: about `88.7%`

Conclusion:

- `video_quality` benefits massively once cache reuse becomes practical.
- Most of the end-to-end gain comes from avoiding repeated decode.

## Benchmark 5: sparse VLM-style frame extraction strategy A/B

Input:

- `tmp/perf_seq_extract_clean/rgb.mp4`
- 9 sampled frames using `frame_step=120`

Compared strategies:

- Old behavior: repeated random seek with `CAP_PROP_POS_FRAMES`
- Candidate behavior: pure sequential scan
- Final behavior: auto strategy

Measured results:

- Old random seek: `12.521s`
- Pure sequential scan: `26.130s`
- New auto strategy: `12.632s`

Interpretation:

- For very sparse frame requests, pure sequential scanning is worse.
- The initial “always sequential” idea was rejected.
- The final auto strategy preserves sparse-sampling performance while still allowing sequential scan for denser workloads like cache building.

## Rejected Experiment

Rejected change:

- Force all caption / hand extraction to use pure sequential scan

Reason:

- Regressed sparse extraction from `12.521s` to `26.130s`

Resolution:

- Replaced with `extract_frames_auto()`

## Files Changed

- [operators/video_utils.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/video_utils.py)
- [operators/frame_cache/cache_utils.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/frame_cache/cache_utils.py)
- [operators/caption/segment_v2t.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/caption/segment_v2t.py)
- [operators/hand/vlm_hand_audit.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/hand/vlm_hand_audit.py)

## Current Takeaways

1. The most valuable improvement in this round is cache reuse, not raw algorithm changes.
2. `video_quality` becomes much cheaper once decode is amortized through cache.
3. Frame extraction strategy must be workload-aware; “one strategy fits all” was incorrect.
4. The next best target is still `privacy_blur`, but it needs a separate benchmark round because this machine currently has no CUDA available in the `operator` env.

## Recommended Next Round

1. Add persistent cache policy to pipeline-level execution so repeated full pipeline runs keep the warm cache.
2. Benchmark `caption` window extraction with dense vs sparse windows to tune the `extract_frames_auto()` heuristic.
3. Run a dedicated `privacy_blur` optimization round, ideally in a CUDA-enabled environment.

## Temporary Instrumentation Round

Goal:

- Validate whether `hand_analysis(method=vlm)` slowdown comes from local extraction or from VLM/API time.

Method:

- Added temporary timing logs only to [operators/hand/vlm_hand_audit.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/hand/vlm_hand_audit.py)
- Logged:
  - cache hit / miss
  - frame extraction elapsed time
  - API stage elapsed time

Important:

- This instrumentation was temporary and removed after testing.

### Test A: run hand VLM directly without prebuilt cache

Observed logs:

- `hand_vlm extract: cache miss`
- `hand_vlm extract: produced 9 frame paths in 15.57s`
- API stage failed immediately because `DASHSCOPE_API_KEY` was not injected in that direct call

Takeaway:

- Without prebuilt cache, local extraction alone is already a double-digit-second cost.

### Test B: run `frame_cache` first, then run hand VLM with API key injected

Observed logs:

- `hand_vlm extract: cache hit for 9/9 frames`
- `hand_vlm extract: produced 9 frame paths in 0.00s`
- `timing_breakdown.extract_sec = 0.001`
- `timing_breakdown.api_sec = 2.987`

This run still failed to reach DashScope because the sandbox blocks the configured proxy/network route, but it was enough to separate local time from API time.

Key conclusion:

- When `frame_cache` is available, local hand VLM preprocessing is essentially negligible.
- Under cache-hit conditions, hand-stage wall time is dominated by the API side, not local frame extraction.
- Therefore, the large hand-stage slowdown seen in the user's full pipeline comparison is unlikely to be caused by the local extraction code path alone.

## Quality Cache Format Benchmark

Goal:

- Validate the report hypothesis that `video_quality`'s grayscale PNG cache may be a poor tradeoff.

Method:

- Added external benchmark script [tmp/benchmark_quality_cache_formats.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/tmp/benchmark_quality_cache_formats.py)
- No core pipeline code was modified
- Input:
  - `tmp/test_episode_30s/rgb.mp4`
  - `sample_fps=2.0`
  - 60 grayscale sampled frames

Measured results:

- Decode source video to grayscale frames: `62.811s`

Format comparison:

- `png`
  - write: `6.042s`
  - read: `1.668s`
  - size: `75.603 MB`

- `jpg_q90`
  - write: `0.430s`
  - read: `0.479s`
  - size: `24.968 MB`

- `npy`
  - write: `0.109s`
  - read: `0.081s`
  - size: `158.203 MB`

- `npz_compressed`
  - write: `9.842s`
  - read: `0.950s`
  - size: `102.197 MB`

Takeaways:

1. The dominant cost is still the initial decode (`62.8s`), not the cache format itself.
2. Among file formats, PNG is not the best tradeoff:
   - much slower than JPEG to write/read
   - much larger than JPEG
   - much slower than NPY/NPZ on I/O
3. If preserving exact grayscale values is not mandatory, `jpg_q90` looks like the best practical replacement candidate.
4. If exact values matter and storage is acceptable, `npy` is the fastest format by far.

## VLM Frame Extraction Benchmark

Goal:

- Validate the report hypothesis that the current VLM preprocessing path is spending too much time on `random seek + temporary JPG`, especially in `caption` and `hand_analysis(vlm)`.

Method:

- Added external benchmark script [tmp/benchmark_vlm_frame_extraction.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/tmp/benchmark_vlm_frame_extraction.py)
- No core pipeline code was modified
- Environment:
  - conda env: `operator`
  - input video: `tmp/test_episode_30s/rgb.mp4`
- Benchmarked only the local preprocessing stage:
  - current implementation: `cap.set(CAP_PROP_POS_FRAMES, fid)` random seek
  - sequential alternative: single forward scan + on-hit resize/JPEG encode
  - compared both `tmp jpg` and in-memory `base64 jpg`

### Caption window benchmark

Input:

- first caption window
- 12 sampled frames
- frame ids: `[0, 27, 54, 81, 108, 135, 162, 189, 216, 243, 270, 298]`

Results:

- current `tmp jpg`: `25.831s`
- current `base64`: `25.529s`
- sequential `tmp jpg`: `12.674s`
- sequential `base64`: `11.862s`

Takeaways:

1. The main cost is not writing temporary JPGs. `tmp jpg` and `base64` are almost the same under the current random-seek path.
2. Replacing random seek with a single sequential scan cuts local preprocessing roughly in half for this caption window.
3. `base64` avoids a little more overhead than `tmp jpg`, but the big win comes from changing decode strategy, not from file-vs-memory alone.

### Hand VLM benchmark

Input:

- `frame_step=15`
- 61 sampled frames across the 30s video

Results:

- current `tmp jpg`: `137.837s`
- sequential `tmp jpg`: `34.773s`
- sequential `base64`: `31.554s`

Takeaways:

1. On sparse hand sampling, the current random-seek extraction path is dramatically more expensive than sequential scan.
2. For this sample, sequential scan reduces local extraction time by about `75%`:
   - `137.837s -> 34.773s` for `tmp jpg`
3. Moving from `tmp jpg` to in-memory `base64` helps a bit more, but again the dominant gain comes from removing random seek.

Key conclusion:

- This experiment strongly supports the report's P0 direction:
  - the real bottleneck in VLM preprocessing is `CAP_PROP_POS_FRAMES` style random seek
  - temporary JPGs are a secondary cost
- If this path is optimized later in core code, the highest-value change is:
  - first replace random seek with sequential extraction / unified frame provider
  - then decide whether `file://jpg` should be replaced with direct in-memory image passing

## Unified Frame Provider Prototype

Goal:

- Validate the larger P0 idea from the report: instead of letting `caption` and `hand_analysis(vlm)` extract frames independently, simulate a single shared provider that scans the video once and serves both consumers.

Method:

- Added external benchmark script [tmp/benchmark_unified_frame_provider.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/tmp/benchmark_unified_frame_provider.py)
- No core pipeline code was modified
- Input:
  - `tmp/test_episode_30s/rgb.mp4`
  - caption default sliding windows
  - hand sampling `frame_step=15`
- Compared:
  1. current behavior:
     - caption extracts each window independently using current code path
     - hand extracts its own sampled frames independently using current code path
  2. provider prototype:
     - collect union of caption + hand requested frame ids
     - perform one sequential scan
     - write each requested frame once as `640x480 jpg`

Results:

- caption:
  - windows: `6`
  - requested frames: `72`
  - unique frames: `68`
  - current local extraction: `113.197s`

- hand:
  - requested frames: `61`
  - unique frames: `61`
  - current local extraction: `103.939s`

- combined:
  - combined unique frames: `122`
  - current separate extraction total: `217.136s`
  - shared provider one-pass extraction: `25.576s`
  - saved: `191.560s`
  - speedup: `8.49x`

Takeaways:

1. This is the strongest result so far in the whole performance exploration.
2. For the tested episode, most of the local VLM preprocessing time is avoidable duplicated decode/seeking work.
3. A shared `FrameProvider` style design is much more promising than isolated micro-optimizations such as:
   - toggling temp JPG vs base64 alone
   - adjusting small config values
4. Even before touching API latency, a unified provider has the potential to cut well over 3 minutes of local preprocessing from the tested `caption + hand` combination.

Key conclusion:

- The report's top recommendation is now experimentally well supported:
  - first-class shared frame provisioning is likely the highest-value structural optimization available outside `privacy_blur`
  - if/when core code changes are considered again, this should be prioritized ahead of further tuning-only work
