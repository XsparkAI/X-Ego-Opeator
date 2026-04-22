from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from .cache_utils import PROFILE_VLM, ensure_cached_frame_paths


@dataclass
class FrameProvider:
    episode_dir: Path
    profile: str = PROFILE_VLM
    _path_cache: dict[int, str] = field(default_factory=dict)
    _b64_cache: dict[int, str] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    def ensure_profile(self, frame_ids: list[int]) -> list[str]:
        unique_ids = sorted({int(fid) for fid in frame_ids})
        with self._lock:
            missing_ids = [fid for fid in unique_ids if fid not in self._path_cache]
            if missing_ids:
                paths = ensure_cached_frame_paths(
                    self.episode_dir,
                    missing_ids,
                    profile=self.profile,
                )
                if paths is None:
                    return []
                for fid, path in zip(sorted(missing_ids), paths):
                    self._path_cache[fid] = path
            return [self._path_cache[fid] for fid in unique_ids if fid in self._path_cache]

    def get_paths(self, frame_ids: list[int]) -> list[str]:
        unique_ids = sorted({int(fid) for fid in frame_ids})
        paths = self.ensure_profile(unique_ids)
        if len(paths) != len(unique_ids):
            return []
        return paths

    def get_b64(self, frame_ids: list[int]) -> list[str]:
        unique_ids = sorted({int(fid) for fid in frame_ids})
        paths = self.get_paths(unique_ids)
        if len(paths) != len(unique_ids):
            return []
        with self._lock:
            for fid, path in zip(unique_ids, paths):
                if fid not in self._b64_cache:
                    self._b64_cache[fid] = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            return [self._b64_cache[fid] for fid in unique_ids]
