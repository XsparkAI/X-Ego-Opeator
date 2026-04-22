# Caption Optimization Directions

## Goal

Summarize the most promising speed/efficiency optimization directions for the current `caption` / `video_segmentation` path, with emphasis on:

- what the current mechanism is doing
- where the main cost is coming from
- which changes are likely to deliver the highest practical gain
- what can be implemented incrementally without changing output schema


## Current State

### Current pipeline status

- The pipeline operator is `video_segmentation`.
- In the current config, `video_segmentation.enabled` is `false`.
- If enabled, the configured method is `task_action_v2t`.
- The current config also sets `batch_enabled: false`, so requests are sent through the direct request path instead of async batch submission.

Relevant files:

- [pipeline.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline.py)
- [pipeline_config.yaml](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline_config.yaml)
- [operators/caption/op_impl.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/caption/op_impl.py)


## Current Mechanism

The current `task_action_v2t` path is a two-level sliding-window pipeline:

1. Run one scene classification request for the whole video.
2. Run task-level sliding windows over the full video.
3. Merge window results into task segments.
4. Run one text-only VLM refinement step to merge adjacent task segments with the same objective.
5. For each refined task segment, run action-level sliding windows again.
6. Merge action-level window results into final `atomic_actions`.

Current default window settings:

- Task level: `12s` window, `6s` step, `12` frames per window
- Action level: `6s` window, `3s` step, `8` frames per window

Relevant files:

- [operators/caption/task_action_v2t.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/caption/task_action_v2t.py)
- [operators/caption/scene_classifier.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/caption/scene_classifier.py)
- [operators/caption/vlm_api.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/caption/vlm_api.py)


## Main Bottlenecks

### 1. Duplicate decode / random seek dominates local preprocessing

Even though frame caching exists, the VLM path still falls back to on-demand frame extraction. On cache miss, frame extraction uses `cv2.VideoCapture` plus repeated `cap.set(CAP_PROP_POS_FRAMES, fid)`, which is expensive on compressed video.

This matters because:

- overlapping windows cause many repeated frame requests
- task-level and action-level windows both need sampled frames
- other operators such as VLM hand analysis may request overlapping frames too

Internal experiment summary:

- switching a caption window from random seek to sequential scan roughly halved local preprocessing time
- a shared one-pass provider for `caption + hand` reduced tested local extraction time from about `217s` to about `25.6s`
- the biggest gain came from eliminating random seek, not from file-vs-base64 alone

Relevant files:

- [PERFORMANCE_OPTIMIZATION_REPORT.md](/home/pc/Desktop/zijian/ego/Ego-X_Operator/PERFORMANCE_OPTIMIZATION_REPORT.md)
- [PERFORMANCE_EXPERIMENT_LOG.md](/home/pc/Desktop/zijian/ego/Ego-X_Operator/PERFORMANCE_EXPERIMENT_LOG.md)
- [operators/frame_cache/cache_utils.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/frame_cache/cache_utils.py)


### 2. `frame_cache` is not aligned with the real `task_action_v2t` demand pattern

Current `frame_cache` warmup logic uses a single `CaptionSamplingSpec(window_sec, step_sec, frames_per_window)`.

However, the active caption method is not single-level sliding windows anymore. `task_action_v2t` has:

- a task-level window schedule
- an action-level window schedule

As a result, the current cache warmup strategy cannot fully represent what caption actually needs when `task_action_v2t` runs.

Implication:

- cache warmup may be incomplete
- later stages may still rebuild or reread missing frames
- some expected reuse is currently lost


### 3. Action-stage requests are fragmented by task

After task refinement, action analysis is executed separately for each task segment.

That means:

- action windows are built task-by-task
- request batches are submitted task-by-task
- frame loading / request construction / thread pools are repeated for each task

This is convenient logically, but inefficient physically. The episode is already known globally; action jobs could be flattened into a single episode-level job list and mapped back to tasks afterward.


### 4. Cached JPGs are still re-read and re-encoded into base64 repeatedly

Even when the frame cache is hit, `ensure_cached_frame_b64()` still reads JPG files and base64-encodes them on each call.

For heavily overlapping windows, this means:

- repeated disk reads
- repeated byte-to-base64 conversion
- avoidable CPU cost

This is smaller than the random-seek problem, but still worth addressing after the P0 structural changes.


### 5. Fixed dense overlap is expensive on long stable video segments

The current design assumes the entire video needs dense overlapping analysis:

- task level scans the whole video with 50% overlap
- action level scans every final task segment with 50% overlap again

This is robust, but it overpays on long stable regions where little changes.


## Recommended Directions

## P0. Build a real shared `FrameProvider` for VLM consumers

This remains the highest-value direction.

### Proposed change

Introduce a thin shared layer, for example:

- `operators/frame_cache/frame_provider.py`

Core behavior:

- accept a set of requested frame IDs for one profile
- scan the video sequentially once
- materialize each requested frame once
- serve both path-based and in-memory access

Initial scope should stay narrow:

- first support only the current VLM profile, such as `vlm_640x480_q85`
- integrate `caption` first
- then integrate `hand_analysis(method=vlm)`

### Why this is P0

- strongest experimental support
- no need to change caption schema
- directly attacks the biggest measured local bottleneck

### Expected impact

- large reduction in repeated decode work
- large reduction in seek overhead
- much better reuse across caption and other VLM operators


## P0. Align cache warmup with `task_action_v2t`

The cache planning logic should match the actual caption method.

### Proposed change

Instead of one generic caption sampling spec, cache planning should understand:

- task window config
- action window config
- potentially scene-classification samples too

Two implementation options:

1. Minimal fix:
   Extend frame-cache planning so it can warm task windows and action windows separately.
2. Better fix:
   Move planning into a reusable provider/planner that receives actual job requests from caption and hand paths, then warms the union.

### Expected impact

- higher real cache hit rate
- fewer fallback extractions
- simpler mental model: warmup matches real consumers


## P0. Flatten action jobs into one episode-level submission

Current action requests are generated and submitted one task at a time.

### Proposed change

Refactor action-stage execution into:

1. build refined task segments
2. generate all action windows for the full episode
3. submit one flattened action request list
4. parse all responses
5. regroup them back into per-task `atomic_actions`

### Why it helps

- fewer small request batches
- less repeated request-building overhead
- better cache locality
- easier to union frame IDs before extraction

### Notes

This change pairs very naturally with the shared `FrameProvider` direction.


## P1. Add an episode-level in-memory frame byte/base64 cache

After structural extraction is fixed, the next obvious waste is repeated file reads for overlapping windows.

### Proposed change

Add an optional episode-scoped cache for:

- JPEG bytes by `frame_id`
- or base64 strings by `frame_id`

Possible shape:

- `dict[int, bytes]`
- or a bounded LRU if memory is a concern

### Expected impact

- reduces repeated disk reads on overlapping windows
- lowers CPU overhead from repeated base64 conversion
- especially useful if direct API mode remains the default


## P1. Move from fixed dense sliding windows to coarse-to-fine analysis

This is likely the most promising algorithmic optimization after extraction is fixed.

### Proposed change

Use a two-pass strategy:

1. Global skim:
   Sample the full video sparsely to find candidate change regions.
2. Local refinement:
   Only run dense sliding windows near candidate boundaries.

Possible low-cost candidate generators:

- simple visual change detector
- scene/shot boundary detector
- motion or embedding-change heuristic
- lightweight temporal segmentation model

### Why this matters

Long videos often contain extended stable spans. Dense 50% overlap everywhere is robust, but expensive.

### Risk

Boundary recall could drop if the skim stage is too aggressive.

### Mitigation

- keep generous margin around candidate regions
- benchmark against current dense baseline
- retain fallback to full scan mode


## P1. Add smarter scheduling and request budgeting

The current code uses concurrency caps, but `caption` still creates nested pools and request groups.

### Proposed change

Introduce a lighter-weight resource model for:

- VLM request slots
- decode slots
- heavy I/O slots

This does not need to be a full scheduler rewrite. Even a shared local budget layer would reduce accidental over-parallelism.

### Why it matters

- prevents operator-level thread pools from fighting each other
- reduces noisy throughput collapse on multi-episode runs
- makes future optimizations easier to reason about


## P2. Split the problem: temporal segmentation model for boundaries, VLM for naming

Today the VLM does both:

- discover boundaries
- produce semantic task/action labels

A more efficient architecture is:

1. use a lightweight temporal segmentation model or heuristic to produce candidate segments
2. call the VLM only once per final segment (or a few times), mainly to name/describe it

### Why this is attractive

- VLM calls drop sharply
- caption becomes less sensitive to overlapping-window explosion
- model roles become cleaner

### Tradeoff

- requires either a learned temporal model or a new heuristic subsystem
- more engineering and evaluation work than the P0 changes


## P2. Evaluate long-video models that can replace many small window calls

There is active work on long-video multimodal models and efficient long-context video understanding.

Relevant examples:

- `LongVA`
- `LongVT`
- `VideoChat-Flash`

Potential value:

- process far longer clips in one shot
- reduce the need for hundreds of small overlapping window calls
- open the door to global reasoning over long sequences

Tradeoffs:

- likely requires self-hosting or a very different serving stack
- integration cost is much higher
- evaluation complexity is higher
- may not be the fastest route to practical wins in the current codebase


## Suggested Implementation Order

### 1-day scope

- document the exact frame demand shape of current `task_action_v2t`
- fix cache planning mismatch
- instrument timing for:
  - frame extraction
  - JPG/base64 preparation
  - API latency
  - response parsing

### 3-day scope

- add a thin shared `FrameProvider`
- route caption task-level windows through it
- flatten action job submission for one episode

### 1-week scope

- route action stage through the same provider
- route VLM hand analysis through the same provider
- add optional in-memory per-episode frame byte/base64 cache
- produce before/after throughput benchmarks on representative 10s / 30s / 1h samples


## Verification Plan

Any change should be evaluated on both quality and performance.

### Performance metrics

- total caption wall-clock time
- local frame preparation time
- number of unique requested frames
- number of duplicate frame requests
- VLM API time
- cache hit rate
- peak concurrency during caption

### Quality checks

- number of tasks
- number of atomic actions
- boundary drift vs current baseline
- instruction quality / readability
- failure rate on short videos and long videos


## Bottom Line

If only a few changes are made, the recommended order is:

1. shared sequential `FrameProvider`
2. align frame-cache planning with `task_action_v2t`
3. flatten action-stage submission to one episode-level request set

These three changes are the highest-confidence path to substantial speed gains without forcing a redesign of the output schema or the overall product behavior.
