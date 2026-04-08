# Ego-X Operator 性能优化报告

## 1. 结论摘要

结合当前配置和代码结构，当前最值得优先优化的不是单一算法，而是这 4 类问题：

1. 重复解码和重复抽帧仍然很多，`frame_cache` 只覆盖了一部分路径。
2. VLM 路径大量依赖“随机 seek + 临时 JPG 落盘”，I/O 和解码开销偏高。
3. 调度层存在“stage 并发 + operator 内并发 + 多 episode pipeline 并发”叠加，容易把 CPU / 磁盘 / GPU 同时打满。
4. `privacy_blur` 是当前最重的单算子，样例里只有 `7.84 fps`，已经足以成为整条流水线的主瓶颈。

从当前配置看，[pipeline_config.yaml](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline_config.yaml#L25) 开了较高的 VLM 并发，且启用了 `video_quality`、`hand_analysis(method=vlm)` 和 `privacy_blur`，而 `video_segmentation` 关闭，所以当前实际运行中最主要的耗时大概率集中在：

- `privacy_blur`
- `hand_analysis(vlm)`
- `frame_cache + video_quality` 的解码/缓存链路

## 2. 当前执行特征

### 2.1 调度层

- Pipeline 会把无依赖算子放进同一 stage 并行执行，[pipeline.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline.py#L179)。
- 多 episode 的 `pipeline` 模式又会给每个 stage 单独起 worker 线程池，[pipeline.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline.py#L346)。
- stage 内还会再次起 `ThreadPoolExecutor` 跑多个 operator，[pipeline.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline.py#L391)。

这意味着系统实际并发度不是单一数字，而是：

`stage_workers × stage 内 operator 数 × operator 自身 max_workers`

如果不做预算，很容易出现“看起来配置只写了 8，实际有几十个活跃线程/请求/解码器”的情况。

### 2.2 当前配置

- `vlm_global_max_concurrency: 8`，[pipeline_config.yaml](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline_config.yaml#L25)
- `cpu_global_max_concurrency: 4`，[pipeline_config.yaml](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline_config.yaml#L29)
- `execution_mode: pipeline`，[pipeline_config.yaml](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline_config.yaml#L46)
- `hand_analysis.method: vlm`，[pipeline_config.yaml](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline_config.yaml#L108)
- `video_quality.sample_fps: 2.0`，[pipeline_config.yaml](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline_config.yaml#L94)

这套配置更偏“吞吐优先”，但资源调度还不够精细。

### 2.3 已有样例性能信号

样例 `blur_report.json` 显示 326 帧视频处理耗时 41.57 秒，吞吐仅 7.84 fps，[blur_report.json](/home/pc/Desktop/zijian/ego/Ego-X_Operator/tmp/test_episode_10s/blur_report.json#L4)。这说明在当前工作负载下，`privacy_blur` 已经是非常明确的主瓶颈。

## 3. 优先级最高的优化项

## P0. 把帧缓存从“文件缓存”升级成“统一帧供应层”

### 现状

- `frame_cache` 会预先导出 VLM 用 JPG 和 quality 用 PNG，[frame_cache/op_impl.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/frame_cache/op_impl.py#L42)。
- 但 caption 和 hand VLM 在 cache miss 时，仍然会自己 `VideoCapture + CAP_PROP_POS_FRAMES + cv2.imwrite`，[segment_v2t.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/caption/segment_v2t.py#L123), [vlm_hand_audit.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/hand/vlm_hand_audit.py#L83)。
- quality cache 和 VLM cache 还分成两套独立的构建流程，[cache_utils.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/frame_cache/cache_utils.py#L161)。

### 问题

- 同一视频可能被多次完整扫描。
- 同一帧可能被重复 resize、重复编码、重复落盘。
- 文件级缓存降低了重复计算，但仍引入大量磁盘 I/O。

### 建议

做一个统一的 `FrameProvider` / `FrameSampler` 抽象：

- 一次顺序扫描视频，按 frame id 集合同时产出多种 profile。
- 支持内存直传和文件落盘两种模式。
- 让 `caption / hand / quality` 都只通过这一层拿帧。

### 预期收益

- 明显减少重复解码。
- 降低随机 seek 带来的性能抖动。
- 后续更容易加 mmap、共享内存、LMDB、WebDataset 等中间层。

### 落地顺序

1. 先抽象统一接口，不改业务逻辑。
2. 再把 `caption` 和 `hand_vlm` 接到统一 provider。
3. 最后考虑把 quality 也改成共享内存路径，减少 PNG 落盘。

## P0. 去掉 VLM 路径里的“随机 seek + 临时 JPG”

### 现状

- caption 每个 window 都会在临时目录生成 JPG 文件，再用 `file://` 传给 VLM，[segment_v2t.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/caption/segment_v2t.py#L187)。
- hand VLM 也是先导出临时 JPG，再并发请求，[vlm_hand_audit.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/hand/vlm_hand_audit.py#L291)。
- 抽帧时使用 `cap.set(CAP_PROP_POS_FRAMES, fid)`，这是典型的随机 seek 模式，[segment_v2t.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/caption/segment_v2t.py#L136), [vlm_hand_audit.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/hand/vlm_hand_audit.py#L106)。

### 问题

- OpenCV 对压缩视频做随机 seek 通常不便宜。
- 每次 `TemporaryDirectory + cv2.imwrite` 都会触发额外 I/O。
- 多线程下会明显放大 SSD/文件系统压力。

### 建议

优先考虑以下两种方案之一：

1. 如果 DashScope SDK 支持字节流/base64 图像输入，就直接走内存，不落盘。
2. 如果仍必须 `file://`，则让 `frame_cache` 产出的文件直接复用，不要在每次调用里再建临时 JPG。

### 预期收益

- 对 caption/hand VLM 路径是最直接的降耗项之一。
- 在多 episode 下，能明显减少磁盘抖动。

## P0. 为调度层增加“资源预算器”，不要只靠 semaphore

### 现状

- Pipeline 的 stage 并发来自调度层，[pipeline.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline.py#L368)。
- hand/caption 内部还会再起自己的线程池，[segment_v2t.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/caption/segment_v2t.py#L431), [vlm_hand_audit.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/hand/vlm_hand_audit.py#L261)。
- 现在只有“全局 CPU semaphore”和“全局 VLM semaphore”，粒度偏粗，[vlm_limit.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/vlm_limit.py#L7)。

### 问题

- CPU、磁盘、GPU、外部 API 的争抢混在一起。
- `cpu_global_max_concurrency=4` 并不能阻止某个 operator 内部再开更多 CPU 线程。
- `pipeline` 模式和 operator 内并发叠加后，可能反而降低吞吐。

### 建议

增加一个轻量资源预算模型：

- `decode_slots`
- `ffmpeg_slots`
- `gpu_slots`
- `vlm_slots`
- `io_heavy_slots`

然后给每个 operator 标注资源画像，例如：

- `frame_cache`: decode + io_heavy
- `video_quality`: decode/cpu
- `privacy_blur`: gpu + encode + decode
- `hand_vlm`: vlm + io/decode

### 预期收益

- 提升整体吞吐稳定性。
- 降低“局部并发提高但总耗时变长”的情况。

## 4. 高收益中期优化

## P1. 重写 `privacy_blur` 为解码/检测/编码流水线

### 现状

- 当前 `privacy_blur` 是单线程循环：读帧 -> resize -> detect -> blur -> pipe 给 ffmpeg，[blur_privacy.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/privacy_blur/blur_privacy.py#L367)。
- 检测器已做了模型复用，这很好，[privacy_blur/op_impl.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/privacy_blur/op_impl.py#L98)。
- 但整个处理仍基本是“串行帧处理”。

### 问题

- 解码、GPU 检测、CPU 模糊、编码没有重叠。
- 4K 输入下，CPU blur 和编码都可能成为尾部瓶颈。

### 建议

拆成 3 段流水：

1. 解码线程/进程：顺序读取帧。
2. GPU worker：批量做 detector 推理。
3. 编码线程：只负责写出结果。

可进一步加：

- 对“无检测帧”直接零拷贝透传。
- 按 batch 推理多个 frame，提升 GPU 利用率。
- 如果 EgoBlur 模型允许，直接在缩放图上生成 mask，再映射回原图做 blur。

### 预期收益

- 这是最可能把当前 `7.84 fps` 拉上去的项。
- 如果视频多为高分辨率，收益会非常明显。

## P1. `video_quality` 不要默认把全部灰度帧一次性装进内存

### 现状

- `read_frames()` 会先把采样后的所有灰度帧读进列表，[assess.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/video_quality/assess.py#L172)。
- 三个子算子再共享这一整批帧进行计算，[assess.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/video_quality/assess.py#L221)。

### 问题

- 对长视频或高分辨率视频，内存峰值会比较高。
- 当前只要某一步想流式化，就会被整个接口形态卡住。

### 建议

把质量检测拆成两类：

- 单帧统计：quality / exposure，可边读边算。
- 邻帧统计：stability，只保留前一帧和必要状态。

也就是改成真正的 streaming API，而不是“先读完，再批量算”。

### 预期收益

- 降低内存占用。
- 提升长视频的稳定性。
- 为后续多 episode 并行提供更大空间。

## P1. quality cache 不要使用 PNG

### 现状

- quality cache 当前写灰度 PNG，[cache_utils.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/frame_cache/cache_utils.py#L248)。

### 问题

- PNG 压缩比高，但 CPU 编解码成本不低。
- quality 只做数值分析，不需要无损文件格式的可移植性。

### 建议

按优先级选择：

1. `npy` / `npz`
2. JPEG 灰度
3. WebP lossless / near-lossless
4. LMDB/Parquet/Arrow 按 chunk 存

如果目标是纯性能，首推 chunked `npy` 或轻压缩 `npz`。

### 预期收益

- 减少 cache 构建时间。
- 降低 repeated load 的 CPU 开销。

## P1. `segment_cut` 改成单次 ffmpeg filtergraph 批量裁剪

### 现状

- 现在每个 segment 都单独调用一次 ffmpeg，[segment_cut/op_impl.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/segment_cut/op_impl.py#L98)。

### 问题

- segment 多时，ffmpeg 进程启动成本和重复解码成本明显。
- 如果后续重新启用 segmentation，这里会成为新的长尾。

### 建议

改成一次 ffmpeg 执行多个 trim / atrim / concat / map 输出，或者至少按批次切片。

### 预期收益

- segment 数越多，收益越高。

## 5. 低成本快收益优化

## P1. 缓存视频元信息和旋转信息

### 现状

- `get_manual_rotation()` 每次都可能重新 `VideoCapture`，必要时还会跑两次 `ffprobe`，[video_utils.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/operators/video_utils.py#L50)。
- 多个 operator 都会重复调用这部分逻辑。

### 建议

- 给每个 `episode/rgb.mp4` 做进程内 memoization。
- 把 `fps / total_frames / rotation / width / height / codec` 统一缓存到 manifest。

### 预期收益

- 单次收益不大，但全链路调用频繁，累计很划算。

## P1. pipeline 清理 cache 的策略改成“可配置”

### 现状

- episode 结束后会强制删 `.frame_cache`，[pipeline.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline.py#L172), [pipeline.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline.py#L293)。

### 问题

- 对重复调试、重复跑同一批 episode 很不友好。
- 失去了 warm cache 的价值。

### 建议

增加配置项：

- `frame_cache_policy: always_delete | keep | ttl`

默认建议 `keep`，调试阶段收益很高。

## P1. 增加真实 profiling 指标，不要只记总耗时

### 现状

- 目前 operator 级别只记录 `_elapsed_sec`，[pipeline.py](/home/pc/Desktop/zijian/ego/Ego-X_Operator/pipeline.py#L208)。

### 建议

每个重 operator 统一记录：

- 解码耗时
- 抽帧耗时
- resize/预处理耗时
- 模型推理耗时
- 编码/落盘耗时
- 命中 cache 比例
- 平均每帧耗时

这样后续调优不会靠猜。

## 6. 配置层优化建议

针对当前配置，建议先这样调：

### 建议配置 A：单机稳定吞吐

- `execution_mode: pipeline` 保持不变。
- `vlm_global_max_concurrency` 从 `8` 降到 `4~6`。
- `hand_analysis.max_workers` 从 `8` 降到 `4`。
- `cpu_global_max_concurrency` 暂时保持 `4`。

原因：

- 当前 `hand_analysis(vlm)` 和其他 stage 并发叠加后，请求数偏激进。
- 若机器磁盘不是很强，VLM 路径的临时 JPG 会放大抖动。

### 建议配置 B：优先降低尾延迟

- 把 `privacy_blur` 单独限制成独占 GPU stage。
- 其他 stage 允许并发，但不要和多路 `privacy_blur` 同时跑。

原因：

- `privacy_blur` 已经是强瓶颈，把它和别的重任务同时挤资源，通常不划算。

## 7. 建议实施顺序

### 第一阶段：1-2 天可完成

1. 增加 profiling 指标。
2. 缓存 video probe / rotation 元信息。
3. 让 `.frame_cache` 支持保留，不要每次删。
4. 调低 `hand_analysis` 与 VLM 全局并发，先换稳定性。

### 第二阶段：3-5 天

1. 去掉 VLM 临时 JPG 路径，改为直接复用 cache 或内存传图。
2. 把 `frame_cache` 重构成统一帧供应层。
3. 让 `video_quality` 改成流式计算。

### 第三阶段：5-10 天

1. 重构 `privacy_blur` 为解码/检测/编码流水线。
2. 视业务是否重新启用 segmentation，再优化 `segment_cut` 批量裁剪。

## 8. 我认为最值得先做的三件事

如果只做 3 件，我建议按这个顺序：

1. 重构 VLM 抽帧路径，去掉临时 JPG 和随机 seek。
2. 重构 `privacy_blur` 的流水线并做 batch 推理。
3. 建立统一 `FrameProvider`，把 `frame_cache / caption / hand / quality` 串起来。

这三项对总吞吐、资源占用和后续扩展性帮助最大。

## 9. `FrameProvider` 落地方案

基于本轮外部原型实验，这一项已经不只是“想法”，而是当前最值得正式设计的结构优化：

- 在 `tmp/test_episode_30s` 上，`caption + hand` 当前各自抽帧的本地耗时约 `217.136s`
- 外部原型中，一次顺序扫描统一供帧只需要 `25.576s`
- 本地抽帧阶段理论加速约 `8.49x`

### 目标

在不改变现有业务输出格式的前提下，把：

- `frame_cache`
- `caption`
- `hand_analysis(method=vlm)`

统一到同一条“按 frame id 集合一次顺序扫描”的供帧路径上。

第一阶段先不把 `video_quality` 强行并进来，避免把灰度流式处理、缓存格式和 VLM 供帧耦合在一轮改造里。

### 第一阶段范围

只解决 VLM 相关路径：

1. `frame_cache` 负责构建 `640x480 jpg` 的共享缓存。
2. `caption` 不再自己 `cap.set(..., fid)` 抽帧。
3. `hand_analysis(vlm)` 不再自己 `cap.set(..., fid)` 抽帧。
4. 两者都优先从统一 provider / cache 拿图。

不在第一阶段处理：

- `video_quality` 灰度帧格式
- `privacy_blur`
- `segment_cut`
- 跨 episode 持久缓存策略

### 建议接口

建议新增一个轻量层，例如：

- `operators/frame_cache/frame_provider.py`

核心接口可以很小：

```python
@dataclass
class FrameRequest:
    profile: str
    frame_ids: list[int]


class FrameProvider:
    def ensure_profile(
        self,
        episode_dir: Path,
        profile: str,
        frame_ids: list[int],
    ) -> dict: ...

    def get_paths(
        self,
        episode_dir: Path,
        profile: str,
        frame_ids: list[int],
    ) -> list[str] | None: ...
```

第一阶段只需要支持一个 profile：

- `vlm_640x480_q85`

这样可以避免一上来就把质量灰度图、内存对象、mmap 都塞进去。

### 代码改造顺序

#### Step 1. 把 VLM 缓存构建逻辑从 `cache_utils.py` 包成 provider

当前已有能力已经很接近：

- `build_cache(...)`
- `get_cached_frame_paths(...)`
- `manifest`

所以第一步不需要大改，只需要：

- 把“给定 frame id 集合，确保生成 profile”抽成稳定接口
- 明确 provider 的输入输出契约
- 让上层不再直接关心 `cv2.VideoCapture` / `cv2.imwrite`

#### Step 2. 让 `caption` 完全走 provider

当前 `caption` 路径是：

- `build_windows()`
- 每个 window 调 `save_frames_as_tmp_jpg()`
- cache miss 时自己随机 seek 抽帧

建议改成：

1. 先汇总整个 episode 所有 window 的 frame ids
2. 调一次 `provider.ensure_profile(..., all_ids)`
3. 每个 window 仅按 frame id 子集取路径
4. VLM 请求阶段继续使用现有 `file://...jpg`

这一步就能直接消灭 caption 里的随机 seek。

#### Step 3. 让 `hand_analysis(vlm)` 完全走 provider

当前 hand 路径也是：

- 采样 frame ids
- cache miss 时自己随机 seek 抽帧

建议改成：

1. 先算好 hand 的全部 `frame_ids`
2. 调 `provider.ensure_profile(..., hand_ids)`
3. 直接拿路径列表发给现有并发 VLM 调用逻辑

这样 hand 的改动面会比 caption 更小。

#### Step 4. 让 `frame_cache` 退化成“预热 provider”的薄封装

改造后，`frame_cache` 本质上只做一件事：

- 在 pipeline 早期把 caption/hand 可能需要的 VLM profile 预先构建好

它不再是独立的抽帧实现，而只是 provider 的一个 warmup 入口。

### 最小改动实现建议

为了降低回归风险，建议先不要改 VLM API 层，只改“拿图”这一段：

- 保留 `file://jpg` 传图方式
- 保留 `caption` 和 `hand` 现有 prompt / retry / 并发逻辑
- 只替换本地抽帧路径

这样收益已经足够大，而且验证成本最低。

### 回归验证清单

正式改代码时，至少要做这几类验证：

1. 输出一致性
- `caption_v2t.json` 结构不变
- `vlm_hand_audit.json` 结构不变

2. 帧对应关系
- provider 返回路径顺序必须与请求 `frame_ids` 对齐
- 重复 frame id 的窗口仍然要保持原调用顺序

3. cache 行为
- cache hit 时不应重复抽帧
- cache miss 时只构建缺失 profile

4. 清理逻辑
- pipeline 结束后 `.frame_cache` 仍按现有策略清理
- 单独 operator 调用时不应误删用户已有文件

5. 性能验收
- 先用 `test_episode_30s` 对比
- 验收指标优先看：
  - `caption` 本地抽帧耗时
  - `hand` 本地抽帧耗时
  - pipeline 总耗时

### 为什么 `video_quality` 不建议一起改

`video_quality` 虽然也和解码有关，但它和 VLM 路径有两个关键差异：

- 它需要灰度帧，不是 `640x480 jpg`
- 它当前瓶颈里还有“全部读入内存”和 cache format 设计问题

所以更稳妥的路线是：

1. 先把 VLM provider 做成
2. 确认 `caption + hand` 的真实收益
3. 再单独决定 `video_quality` 是接 provider、流式读取，还是两者结合

### 建议实施顺序更新

如果按当前实验结果重新排序，我会把真正的优先级改成：

1. `caption + hand` 统一 `FrameProvider`
2. `privacy_blur` 结构重构
3. `video_quality` 流式化 / 灰度缓存重构

原因很简单：

- 第 1 项已经有最强的实验支撑
- 第 2 项仍然是单算子最大热点
- 第 3 项值得做，但目前证据没有第 1 项这么强
