#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from runner import run_caption
except ModuleNotFoundError:
    from .runner import run_caption
from platform_input import resolve_local_video_inputs

JSON_SUFFIXES = {".json"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DIRECT_VIDEO_NAMES = ("rgb.mp4", "cam_head.mp4", "video.mp4")
VIDEO_KEYS = (
    "video",
    "videoPath",
    "video_path",
    "videoFile",
    "video_file",
    "localVideoPath",
    "local_video_path",
    "inputVideoPath",
    "input_video_path",
    "inputVideo",
    "input_video",
    "sourceVideoPath",
    "source_video_path",
    "inputPath",
    "input_path",
    "localPath",
    "local_path",
    "relativePath",
    "relative_path",
    "filePath",
    "file_path",
    "path",
)
DIR_KEYS = (
    "episodeDir",
    "episode_dir",
    "workDir",
    "work_dir",
    "directory",
    "dir",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json_file(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_bos_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("bos://"):
        raise ValueError(f"Not a BOS URI: {uri}")
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid BOS URI: {uri}")
    return bucket, key


def _pick_runtime_value(hyperparams: dict, env_key: str, *aliases: str, default: str = "") -> str:
    lowered_keys = {env_key.lower(), *(alias.lower() for alias in aliases)}
    for key in (env_key, *aliases):
        value = os.getenv(key, "").strip()
        if value:
            return value
    for existing_key, value in hyperparams.items():
        if str(existing_key).lower() in lowered_keys and value not in (None, ""):
            return str(value).strip()
    return default


def _pick_hyperparam(hyperparams: dict, key: str, *, env_key: str | None = None, default: str = "") -> str:
    """Read one canonical platform hyperparam, with one matching env fallback."""
    value = hyperparams.get(key)
    if value not in (None, ""):
        return str(value).strip()

    lowered_key = key.lower()
    for existing_key, existing_value in hyperparams.items():
        if str(existing_key).lower() == lowered_key and existing_value not in (None, ""):
            return str(existing_value).strip()

    if env_key:
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return default


def _build_bos_context(hyperparams: dict, payload_endpoint: str | None = None) -> dict:
    endpoint = _pick_runtime_value(
        hyperparams,
        "Endpoint",
        "ENDPOINT",
        "endpoint",
        "BOS_ENDPOINT",
        default=payload_endpoint or "https://bj.bcebos.com",
    )
    endpoint = endpoint or (payload_endpoint or "https://bj.bcebos.com")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"

    return {
        "endpoint": endpoint,
        "ak": _pick_runtime_value(
            hyperparams,
            "AccessKey",
            "ACCESSKEY",
            "accessKeyId",
            "access_key_id",
            "BOS_ACCESS_KEY_ID",
        ),
        "sk": _pick_runtime_value(
            hyperparams,
            "SecretKey",
            "SECRETKEY",
            "secretAccessKey",
            "secret_access_key",
            "BOS_SECRET_ACCESS_KEY",
        ),
    }


def _bcecmd_config_from_endpoint(endpoint: str) -> dict[str, str]:
    parsed = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}")
    domain = parsed.netloc or parsed.path or "bj.bcebos.com"
    region = domain.split(".", 1)[0] if "." in domain else domain or "bj"
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    return {
        "domain": domain,
        "region": region or "bj",
        "https": "yes" if scheme == "https" else "no",
    }


def _write_bcecmd_config(config_dir: Path, context: dict) -> None:
    ak = context.get("ak", "")
    sk = context.get("sk", "")
    if not ak or not sk:
        raise RuntimeError(
            "BOS video download requires AccessKey/SecretKey, "
            "BOS_ACCESS_KEY_ID/BOS_SECRET_ACCESS_KEY, or accessKeyId/secretAccessKey "
            "(env or hyperparams)"
        )

    config_dir.mkdir(parents=True, exist_ok=True)
    endpoint_config = _bcecmd_config_from_endpoint(str(context.get("endpoint", "")))
    (config_dir / "config").write_text(
        (
            "[Defaults]\n"
            f"Domain = {endpoint_config['domain']}\n"
            f"Region = {endpoint_config['region']}\n"
            "AutoSwitchDomain = yes\n"
            "BreakpointFileExpiration = 7\n"
            f"Https = {endpoint_config['https']}\n"
            "MultiUploadThreadNum = 10\n"
            "MultiDownloadThreadNum = 10\n"
            "SyncProcessingNum = 10\n"
            "MultiUploadPartSize = 12\n"
            "ProxyHost = \n"
        ),
        encoding="utf-8",
    )
    (config_dir / "credentials").write_text(
        f"[Defaults]\nAk = {ak}\nSk = {sk}\nSts = \n",
        encoding="utf-8",
    )


def _mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return value[:2] + "***"
    return f"{value[:4]}***{value[-4:]}"


def _debug_log_bos_credentials(context: dict, args: list[str]) -> None:
    print(
        "BOS credential debug "
        f"command={' '.join(args)} endpoint={context.get('endpoint', '')} "
        f"AccessKey={_mask_secret(str(context.get('ak', '')))} "
        f"SecretKey={_mask_secret(str(context.get('sk', '')))}",
        file=sys.stderr,
        flush=True,
    )


def _run_bcecmd(context: dict, args: list[str]) -> str:
    bcecmd_path = Path("/app/bcecmd")
    if not bcecmd_path.exists():
        raise RuntimeError("bcecmd binary is not available in the image. Rebuild the image with /app/bcecmd.")

    home_dir = Path("/tmp/egox-home")
    config_dir = home_dir / ".bcecmd"
    _write_bcecmd_config(config_dir, context)

    env = os.environ.copy()
    env["HOME"] = str(home_dir)

    _debug_log_bos_credentials(context, args)
    print(f"[egox] about to run bcecmd: {' '.join(args)}", file=sys.stderr, flush=True)

    try:
        completed = subprocess.run(
            [str(bcecmd_path), "--conf-path", str(config_dir), *args],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or str(exc)
        details = (
            f"{details}; endpoint={context.get('endpoint', '')}; "
            f"AccessKey={_mask_secret(str(context.get('ak', '')))}; "
            f"SecretKey={_mask_secret(str(context.get('sk', '')))}"
        )
        raise RuntimeError(f"bcecmd {' '.join(args)} failed: {details}") from exc

    print(f"[egox] bcecmd completed: {' '.join(args)}", file=sys.stderr, flush=True)
    return completed.stdout


def _candidate_rank(key: str) -> tuple[int, str]:
    lowered = key.lower()
    basename = Path(lowered).name
    if basename == "rgb.mp4":
        return (0, lowered)
    if basename == "cam_head.mp4":
        return (1, lowered)
    if basename == "video.mp4":
        return (2, lowered)
    return (10, lowered)


def _list_bos_video_candidates(context: dict, bucket: str, prefix: str) -> list[str]:
    bos_path = f"bos://{bucket}/{prefix}".rstrip("/")
    stdout = _run_bcecmd(context, ["bos", "ls", bos_path, "--recursive"])
    candidates: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        key = parts[-1]
        if Path(key).suffix.lower() in VIDEO_SUFFIXES:
            candidates.append(key)
    candidates.sort(key=_candidate_rank)
    return candidates


def _download_bos_object(context: dict, bucket: str, key: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{bucket}/{key}".encode("utf-8")).hexdigest()[:12]
    basename = Path(key).name or "input.bin"
    local_path = target_dir / f"{digest}-{basename}"
    if not local_path.exists():
        print(
            f"[egox] downloading BOS object bos://{bucket}/{key} -> {local_path}",
            file=sys.stderr,
            flush=True,
        )
        _run_bcecmd(
            context,
            ["bos", "cp", f"bos://{bucket}/{key}", str(local_path), "--yes", "--disable-bar"],
        )
    else:
        print(f"[egox] using cached BOS object: {local_path}", file=sys.stderr, flush=True)
    return local_path


def _prepare_caption_runtime_video(shared_video_path: Path, cache_root: Path) -> Path:
    digest = hashlib.sha1(str(shared_video_path).encode("utf-8")).hexdigest()[:12]
    runtime_dir = cache_root / "caption_runtime" / digest
    runtime_dir.mkdir(parents=True, exist_ok=True)

    target_name = shared_video_path.name.split("-", 1)[-1] or shared_video_path.name
    runtime_video_path = runtime_dir / target_name
    if runtime_video_path.exists():
        return runtime_video_path

    try:
        runtime_video_path.symlink_to(shared_video_path)
    except OSError:
        shutil.copy2(shared_video_path, runtime_video_path)
    return runtime_video_path


def _resolve_bos_runtime_input(uri: str, bos_context: dict) -> tuple[str, Path] | None:
    bucket, key = _parse_bos_uri(uri)
    temp_root = Path(os.getenv("EGOX_BOS_CACHE_DIR", "/tmp/egox-bos-cache"))

    if Path(key).suffix.lower() in VIDEO_SUFFIXES:
        local_video = _download_bos_object(bos_context, bucket, key, temp_root)
        return ("video", _prepare_caption_runtime_video(local_video, temp_root))

    prefix = key.rstrip("/")
    if prefix:
        prefix = prefix + "/"

    candidates = _list_bos_video_candidates(bos_context, bucket, prefix)
    if not candidates:
        return None

    local_video = _download_bos_object(bos_context, bucket, candidates[0], temp_root)
    return ("video", _prepare_caption_runtime_video(local_video, temp_root))


def _normalize_artifacts(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("artifacts", "items", "inputs"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise RuntimeError("INPUT_ARTIFACTS_PATH does not contain a supported artifact list structure")


def _load_input_artifacts() -> list[dict]:
    return _normalize_artifacts(_read_json_file(Path(os.environ["INPUT_ARTIFACTS_PATH"])))


def _load_node_data() -> dict:
    raw = os.getenv("NODE_DATA_JSON", "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("NODE_DATA_JSON must decode to a JSON object")
    return data


def _as_bool(value: object, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: object, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _as_float(value: object, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _resolve_candidate_path(candidate: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(candidate)
    if path.is_absolute() or base_dir is None:
        return path
    return (base_dir / path).resolve()


def _find_video_in_dir(directory: Path) -> tuple[str, Path] | None:
    configured_name = os.getenv("EGOX_INPUT_VIDEO_PATH", "rgb.mp4")
    for name in (configured_name, *DIRECT_VIDEO_NAMES):
        candidate = directory / name
        if candidate.exists():
            return ("episode", directory)

    direct_videos = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]
    if len(direct_videos) == 1:
        return ("video", direct_videos[0])
    return None


def _path_kind(path: Path) -> tuple[str, Path] | None:
    if not path.exists():
        return None
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return ("video", path)
    if path.is_dir():
        return _find_video_in_dir(path)
    return None


def _search_directory_contents(
    directory: Path,
    bos_context: dict,
    visited_json_files: set[Path],
    visited_dirs: set[Path],
) -> tuple[str, Path] | None:
    if directory in visited_dirs or not directory.exists() or not directory.is_dir():
        return None
    visited_dirs.add(directory)

    direct = _find_video_in_dir(directory)
    if direct is not None:
        return direct

    json_candidates = sorted(path for path in directory.rglob("*.json") if path.is_file())
    for json_path in json_candidates[:200]:
        resolved = _search_path_or_json(json_path, directory, bos_context, visited_json_files, visited_dirs)
        if resolved is not None:
            return resolved

    for child in sorted(path for path in directory.rglob("*") if path.is_dir())[:200]:
        direct = _find_video_in_dir(child)
        if direct is not None:
            return direct
    return None


def _search_path_or_json(
    candidate: str | Path,
    base_dir: Path | None,
    bos_context: dict,
    visited_json_files: set[Path],
    visited_dirs: set[Path],
) -> tuple[str, Path] | None:
    path = _resolve_candidate_path(candidate, base_dir)
    resolved = _path_kind(path)
    if resolved is not None:
        return resolved

    if path.is_file() and path.suffix.lower() in JSON_SUFFIXES:
        real_path = path.resolve()
        if real_path in visited_json_files:
            return None
        visited_json_files.add(real_path)
        return _search_for_runtime_input(
            _read_json_file(real_path),
            bos_context,
            real_path.parent,
            visited_json_files,
            visited_dirs,
        )

    if path.is_dir():
        return _search_directory_contents(path, bos_context, visited_json_files, visited_dirs)
    return None


def _search_for_runtime_input(
    payload: object,
    bos_context: dict,
    base_dir: Path | None = None,
    visited_json_files: set[Path] | None = None,
    visited_dirs: set[Path] | None = None,
) -> tuple[str, Path] | None:
    if visited_json_files is None:
        visited_json_files = set()
    if visited_dirs is None:
        visited_dirs = set()

    if isinstance(payload, str):
        if payload.startswith("bos://"):
            return _resolve_bos_runtime_input(payload, bos_context)
        return _search_path_or_json(payload, base_dir, bos_context, visited_json_files, visited_dirs)

    if isinstance(payload, list):
        for item in payload:
            resolved = _search_for_runtime_input(item, bos_context, base_dir, visited_json_files, visited_dirs)
            if resolved is not None:
                return resolved
        return None

    if not isinstance(payload, dict):
        return None

    for key in VIDEO_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            if value.startswith("bos://"):
                resolved = _resolve_bos_runtime_input(value, bos_context)
            else:
                resolved = _search_path_or_json(value, base_dir, bos_context, visited_json_files, visited_dirs)
            if resolved is not None:
                return resolved

    for key in DIR_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            if value.startswith("bos://"):
                resolved = _resolve_bos_runtime_input(value, bos_context)
            else:
                resolved = _search_path_or_json(value, base_dir, bos_context, visited_json_files, visited_dirs)
            if resolved is not None:
                return resolved

    for key in (
        "data",
        "payload",
        "input",
        "result",
        "metadata",
        "item",
        "items",
        "entries",
        "videos",
        "files",
        "artifacts",
        "dataRef",
        "manifest",
        "partitions",
        "collections",
        "itemsByPartition",
        "entriesByPartition",
    ):
        if key in payload:
            resolved = _search_for_runtime_input(payload[key], bos_context, base_dir, visited_json_files, visited_dirs)
            if resolved is not None:
                return resolved

    for key in ("payloadPath", "manifestPath", "partitionDir", "rootPath", "root_path"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            if value.startswith("bos://"):
                resolved = _resolve_bos_runtime_input(value, bos_context)
            else:
                resolved = _search_path_or_json(value, base_dir, bos_context, visited_json_files, visited_dirs)
            if resolved is not None:
                return resolved

    for value in payload.values():
        resolved = _search_for_runtime_input(value, bos_context, base_dir, visited_json_files, visited_dirs)
        if resolved is not None:
            return resolved
    return None


def _select_primary_artifact(artifacts: list[dict]) -> dict:
    if not artifacts:
        raise RuntimeError("No upstream artifacts found in INPUT_ARTIFACTS_PATH")

    for artifact in artifacts:
        artifact_type = str(artifact.get("type", "")).strip().lower()
        port_name = str(artifact.get("portName", "")).strip().lower()
        port_id = str(artifact.get("portId", "")).strip().lower()
        if artifact_type in {"video", "json"} or port_name in {"input", "video"} or port_id == "in-1":
            return artifact
    return artifacts[0]


def _describe_candidate(candidate: object) -> str:
    if not isinstance(candidate, str) or not candidate:
        return "<empty>"
    if candidate.startswith("bos://"):
        return f"{candidate} [remote]"
    path = Path(candidate)
    exists = path.exists()
    kind = "dir" if path.is_dir() else "file" if path.is_file() else "missing"
    return f"{candidate} [{kind}, exists={exists}]"


def _resolve_runtime_input(artifact: dict, hyperparams: dict) -> tuple[str, Path]:
    artifact_data = artifact.get("data")
    payload_endpoint = artifact_data.get("endpoint") if isinstance(artifact_data, dict) else None
    bos_context = _build_bos_context(hyperparams, payload_endpoint=payload_endpoint if isinstance(payload_endpoint, str) else None)

    artifact_path = artifact.get("path")
    if isinstance(artifact_path, str) and artifact_path and not artifact_path.startswith("bos://"):
        resolved = _search_path_or_json(Path(artifact_path), None, bos_context, set(), set())
        if resolved is not None:
            return resolved
    elif isinstance(artifact_path, str) and artifact_path.startswith("bos://"):
        resolved = _resolve_bos_runtime_input(artifact_path, bos_context)
        if resolved is not None:
            return resolved

    resolved = _search_for_runtime_input(artifact_data, bos_context)
    if resolved is not None:
        return resolved

    data_ref = artifact.get("dataRef") or {}
    for candidate in (data_ref.get("payloadPath"), data_ref.get("manifestPath"), data_ref.get("partitionDir")):
        if isinstance(candidate, str) and candidate:
            if candidate.startswith("bos://"):
                resolved = _resolve_bos_runtime_input(candidate, bos_context)
            else:
                resolved = _search_path_or_json(Path(candidate), None, bos_context, set(), set())
            if resolved is not None:
                return resolved

    candidate_details = [
        f"artifact.path={_describe_candidate(artifact.get('path'))}",
        f"artifact.dataRef.payloadPath={_describe_candidate(data_ref.get('payloadPath'))}",
        f"artifact.dataRef.manifestPath={_describe_candidate(data_ref.get('manifestPath'))}",
        f"artifact.dataRef.partitionDir={_describe_candidate(data_ref.get('partitionDir'))}",
    ]

    nested_data_ref = artifact_data.get("dataRef") if isinstance(artifact_data, dict) else {}
    if isinstance(nested_data_ref, dict):
        candidate_details.extend(
            [
                f"artifact.data.dataRef.payloadPath={_describe_candidate(nested_data_ref.get('payloadPath'))}",
                f"artifact.data.dataRef.manifestPath={_describe_candidate(nested_data_ref.get('manifestPath'))}",
                f"artifact.data.dataRef.partitionDir={_describe_candidate(nested_data_ref.get('partitionDir'))}",
            ]
        )

    raise RuntimeError(
        "Caption custom operator requires a local video file or episode directory "
        "from artifact.path, artifact.data, artifact JSON payloads, or dataRef payload/manifests/partitions. "
        + "Checked: "
        + "; ".join(candidate_details)
    )


def _build_args(kind: str, input_path: Path, output_path: Path, hyperparams: dict) -> argparse.Namespace:
    method = _normalize_method(_pick_hyperparam(hyperparams, "method", env_key="METHOD", default="task"))
    vlm_provider = _pick_hyperparam(hyperparams, "vlm_api_provider", env_key="VLM_API_PROVIDER")
    no_batch_value = _pick_hyperparam(hyperparams, "no_batch", env_key="NO_BATCH")
    no_batch = _as_bool(no_batch_value, default=True)
    if vlm_provider.strip().lower() == "volcengine_ark":
        no_batch = True
    segment_granularity = _pick_hyperparam(
        hyperparams,
        "segment_granularity",
        env_key="SEGMENT_GRANULARITY",
        default="task",
    ).lower()

    return argparse.Namespace(
        method=method if method in {"task", "atomic_action"} else "task",
        video=input_path if kind == "video" else None,
        episode=input_path if kind == "episode" else None,
        output=output_path,
        preview=_as_bool(_pick_hyperparam(hyperparams, "preview", env_key="PREVIEW"), default=False),
        max_workers=_as_int(_pick_hyperparam(hyperparams, "max_workers", env_key="MAX_WORKERS"), 8),
        no_batch=no_batch,
        segment_cut=_as_bool(
            _pick_hyperparam(hyperparams, "segment_cut", env_key="SEGMENT_CUT"),
            default=False,
        ),
        segment_granularity=segment_granularity,
        segment_output_dir=None,
        window_sec=_as_float(_pick_hyperparam(hyperparams, "window_sec", env_key="WINDOW_SEC"), 10.0),
        step_sec=_as_float(_pick_hyperparam(hyperparams, "step_sec", env_key="STEP_SEC"), 5.0),
        frames_per_window=_as_int(
            _pick_hyperparam(hyperparams, "frames_per_window", env_key="FRAMES_PER_WINDOW"),
            12,
        ),
        task_name=_pick_hyperparam(hyperparams, "task_name", env_key="TASK_NAME"),
        task_window_sec=_as_float(
            _pick_hyperparam(hyperparams, "task_window_sec", env_key="TASK_WINDOW_SEC"),
            12.0,
        ),
        task_step_sec=_as_float(
            _pick_hyperparam(hyperparams, "task_step_sec", env_key="TASK_STEP_SEC"),
            6.0,
        ),
        task_frames_per_window=_as_int(
            _pick_hyperparam(hyperparams, "task_frames_per_window", env_key="TASK_FRAMES_PER_WINDOW"),
            12,
        ),
        action_window_sec=_as_float(
            _pick_hyperparam(hyperparams, "action_window_sec", env_key="ACTION_WINDOW_SEC"),
            6.0,
        ),
        action_step_sec=_as_float(
            _pick_hyperparam(hyperparams, "action_step_sec", env_key="ACTION_STEP_SEC"),
            3.0,
        ),
        action_frames_per_window=_as_int(
            _pick_hyperparam(hyperparams, "action_frames_per_window", env_key="ACTION_FRAMES_PER_WINDOW"),
            8,
        ),
        vlm_provider=vlm_provider,
        vlm_api_key=_pick_hyperparam(hyperparams, "vlm_api_key", env_key="VLM_API_KEY"),
        dashscope_api_key=_pick_hyperparam(hyperparams, "dashscope_api_key", env_key="DASHSCOPE_API_KEY"),
        ark_api_key=_pick_hyperparam(hyperparams, "ark_api_key", env_key="ARK_API_KEY"),
        vlm_model=_pick_hyperparam(hyperparams, "vlm_model", env_key="VLM_MODEL"),
    )


def _normalize_method(method: str | None) -> str:
    normalized = str(method or "task").strip().lower()
    aliases = {
        "segment_v2t": "task",
        "task_v2t": "task",
        "task_action_v2t": "atomic_action",
        "atomic": "atomic_action",
        "atomic-action": "atomic_action",
    }
    return aliases.get(normalized, normalized)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_platform_video_inputs(artifact: dict, hyperparams: dict) -> list[tuple[str, Path]]:
    videos = resolve_local_video_inputs(
        artifact,
        read_json_file=_read_json_file,
        video_keys=VIDEO_KEYS,
        video_suffixes=VIDEO_SUFFIXES,
        direct_video_names=DIRECT_VIDEO_NAMES,
    )
    if videos:
        return [("video", path) for path in videos]
    return [_resolve_runtime_input(artifact, hyperparams)]


def _caption_output_name(input_path: Path, index: int, total: int) -> str:
    if total == 1:
        return "caption.json"
    return f"{input_path.stem}_caption.json"


def _log_runtime_args(args: argparse.Namespace) -> None:
    print(
        "[egox] resolved runtime args "
        f"version={os.getenv('CAPTION_DOCKER_VERSION', '<unset>')} "
        f"method={args.method} "
        f"no_batch={args.no_batch} "
        f"max_workers={args.max_workers} "
        f"preview={args.preview} "
        f"segment_cut={args.segment_cut} "
        f"segment_granularity={args.segment_granularity} "
        f"vlm_provider={args.vlm_provider or '<empty>'} "
        f"vlm_model={args.vlm_model or '<empty>'}",
        file=sys.stderr,
        flush=True,
    )


def _runtime_args_summary(args: argparse.Namespace) -> dict:
    return {
        "method": args.method,
        "noBatch": args.no_batch,
        "maxWorkers": args.max_workers,
        "preview": args.preview,
        "segmentCut": args.segment_cut,
        "segmentGranularity": args.segment_granularity,
        "taskName": args.task_name or None,
        "windowSec": args.window_sec,
        "stepSec": args.step_sec,
        "framesPerWindow": args.frames_per_window,
        "taskWindowSec": args.task_window_sec,
        "taskStepSec": args.task_step_sec,
        "taskFramesPerWindow": args.task_frames_per_window,
        "actionWindowSec": args.action_window_sec,
        "actionStepSec": args.action_step_sec,
        "actionFramesPerWindow": args.action_frames_per_window,
        "vlmProvider": args.vlm_provider or None,
        "vlmModel": args.vlm_model or None,
    }


def _caption_artifact_data(summary: dict, output_path: Path) -> dict:
    return {
        "method": summary.get("method"),
        "instruction": summary.get("instruction"),
        "scene": summary.get("scene"),
        "numTasks": summary.get("numTasks"),
        "numActions": summary.get("numActions"),
        "previewGenerated": summary.get("previewGenerated"),
        "payloadPath": str(output_path),
    }


def main() -> None:
    output_dir = Path(os.environ["OUTPUT_DIR"])
    result_output_path = Path(os.environ["RESULT_OUTPUT_PATH"])
    stage_manifest_path = Path(os.environ["STAGE_MANIFEST_PATH"])
    node_label = os.getenv("NODE_LABEL", "caption")

    output_dir.mkdir(parents=True, exist_ok=True)
    node_data = _load_node_data()
    hyperparams = node_data.get("hyperparams") if isinstance(node_data.get("hyperparams"), dict) else {}

    artifacts = _load_input_artifacts()
    source_artifact = _select_primary_artifact(artifacts)
    video_inputs = _resolve_platform_video_inputs(source_artifact, hyperparams)
    _write_json(stage_manifest_path, {"dataRefs": []})

    output_artifacts = []
    summaries = []
    first_args = None
    for index, (input_kind, input_path) in enumerate(video_inputs, start=1):
        output_filename = _caption_output_name(input_path, index, len(video_inputs))
        output_name = _pick_hyperparam(hyperparams, "output_name", env_key="OUTPUT_NAME").strip()
        if output_name and len(video_inputs) == 1:
            output_filename = output_name if output_name.endswith(".json") else f"{output_name}.json"
        artifact_path = output_dir / output_filename
        args = _build_args(input_kind, input_path, artifact_path, hyperparams)
        if args.segment_cut:
            args.segment_output_dir = output_dir / f"{artifact_path.stem}_segments"
        if index == 1:
            _log_runtime_args(args)
            first_args = args

        runtime_result = run_caption(args)
        runtime_args = _runtime_args_summary(args)
        caption_result = runtime_result.get("result") or {}
        summary = runtime_result.get("summary") or {}
        summaries.append(_caption_artifact_data(summary, runtime_result["output_path"]))
        _write_json(runtime_result["output_path"], caption_result)
        segment_result = runtime_result.get("segment_result")
        segment_metrics = segment_result.metrics if segment_result is not None else {}
        output_artifacts.append({
            "name": artifact_path.name,
            "type": "json",
            "path": str(runtime_result["output_path"]),
            "portId": "out-1",
            "portName": "result",
            "data": _caption_artifact_data(summary, runtime_result["output_path"]),
            "dataRef": {"payloadPath": str(runtime_result["output_path"])},
            "metadata": {
                "method": runtime_result["method"],
                "captionDockerVersion": os.getenv("CAPTION_DOCKER_VERSION", "<unset>"),
                "runtimeArgs": runtime_args,
                "inputKind": input_kind,
                "inputPath": str(input_path),
                "inputArtifactName": source_artifact.get("name"),
                "inputPortId": source_artifact.get("portId"),
                "inputPortName": source_artifact.get("portName"),
                "inputIndex": index,
                "inputCount": len(video_inputs),
                "scene": summary.get("scene"),
                "previewPath": str(runtime_result["preview_path"]) if runtime_result.get("preview_path") else None,
                "segmentCut": bool(segment_result),
                "segmentMetrics": segment_metrics or None,
            },
        })
        if segment_result is not None:
            manifest_path = Path(segment_metrics.get("manifest_path", ""))
            failure_samples = [
                {
                    "segmentId": failure.get("segmentId"),
                    "reason": failure.get("reason"),
                    "message": failure.get("message"),
                    "frameInterval": failure.get("frameInterval"),
                }
                for failure in segment_metrics.get("failures", [])[:3]
                if isinstance(failure, dict)
            ]
            requested_segments = int(segment_metrics.get("requested_segments", 0) or 0)
            total_segments = int(segment_metrics.get("total_segments", 0) or 0)
            failed_segments = int(segment_metrics.get("failed_segments", 0) or 0)
            if requested_segments and total_segments == 0:
                segment_cut_status = "failed"
            elif failed_segments:
                segment_cut_status = "partial"
            else:
                segment_cut_status = "ok"
            segment_video_artifacts = []
            for segment_idx, segment_dir_value in enumerate(segment_metrics.get("segment_dirs", []), start=1):
                segment_dir = Path(segment_dir_value)
                segment_video_path = segment_dir / "rgb.mp4"
                if not segment_video_path.exists():
                    continue
                segment_info_path = segment_dir / "segment_info.json"
                segment_video_artifacts.append({
                    "name": f"{artifact_path.stem}_{segment_dir.name}.mp4",
                    "type": "video",
                    "path": str(segment_video_path),
                    "portId": "out-3",
                    "portName": "segment_video",
                    "data": {
                        "segmentId": segment_dir.name,
                        "videoPath": str(segment_video_path),
                        "infoPath": str(segment_info_path) if segment_info_path.exists() else None,
                        "manifestPath": str(manifest_path),
                        "captionPath": str(runtime_result["output_path"]),
                    },
                    "metadata": {
                        "runtimeArgs": runtime_args,
                        "inputKind": input_kind,
                        "inputPath": str(input_path),
                        "inputIndex": index,
                        "inputCount": len(video_inputs),
                        "segmentIndex": segment_idx,
                        "segmentCount": segment_metrics.get("total_segments", 0),
                        "granularity": segment_metrics.get("granularity"),
                    },
                })
            output_artifacts.append({
                "name": manifest_path.name or "segments_manifest.json",
                "type": "json",
                "path": str(manifest_path),
                "portId": "out-2",
                "portName": "segments",
                "data": {
                    "payloadPath": str(manifest_path),
                    "segmentsDir": str(manifest_path.parent),
                    "status": segment_cut_status,
                    "requestedSegments": requested_segments,
                    "totalSegments": total_segments,
                    "videoArtifacts": len(segment_video_artifacts),
                    "failedSegments": failed_segments,
                    "failureSamples": failure_samples,
                    "granularity": segment_metrics.get("granularity"),
                },
                "dataRef": {"payloadPath": str(manifest_path)},
                "metadata": {
                    "runtimeArgs": runtime_args,
                    "inputKind": input_kind,
                    "inputPath": str(input_path),
                    "inputIndex": index,
                    "inputCount": len(video_inputs),
                    "captionPath": str(runtime_result["output_path"]),
                },
            })
            output_artifacts.extend(segment_video_artifacts)

    if len(video_inputs) > 1:
        aggregate_path = output_dir / "caption_manifest.json"
        aggregate_data = {"inputCount": len(video_inputs), "captions": summaries}
        _write_json(aggregate_path, aggregate_data)
        output_artifacts.append({
            "name": aggregate_path.name,
            "type": "json",
            "path": str(aggregate_path),
            "portId": "out-1",
            "portName": "result",
            "data": aggregate_data,
            "dataRef": {"payloadPath": str(aggregate_path)},
            "metadata": {"inputCount": len(video_inputs)},
        })

    result_output = {
        "logs": [
            {
                "timestamp": _now_iso(),
                "level": "INFO",
                "message": (
                    "caption resolved runtime args: "
                    f"version={os.getenv('CAPTION_DOCKER_VERSION', '<unset>')}, "
                    f"method={first_args.method if first_args else '<empty>'}, "
                    f"no_batch={first_args.no_batch if first_args else '<empty>'}, "
                    f"max_workers={first_args.max_workers if first_args else '<empty>'}, "
                    f"preview={first_args.preview if first_args else '<empty>'}, "
                    f"segment_cut={first_args.segment_cut if first_args else '<empty>'}, "
                    f"segment_granularity={first_args.segment_granularity if first_args else '<empty>'}, "
                    f"vlm_provider={(first_args.vlm_provider or '<empty>') if first_args else '<empty>'}, "
                    f"vlm_model={(first_args.vlm_model or '<empty>') if first_args else '<empty>'}, "
                    f"input_count={len(video_inputs)}"
                ),
                "operator": node_label,
            },
            {
                "timestamp": _now_iso(),
                "level": "INFO",
                "message": "caption custom operator completed successfully",
                "operator": node_label,
            },
        ],
        "artifacts": output_artifacts,
        "resourceSeries": [],
    }
    _write_json(result_output_path, result_output)

    print(f"DROBOTICFLOW_RESULT_PATH={result_output_path}")


if __name__ == "__main__":
    main()
