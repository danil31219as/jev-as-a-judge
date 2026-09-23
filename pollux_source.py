"""Download the pinned POLLUX Parquet shards without listing the Hub repository."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DATASET_ID = "ai-forever/POLLUX"
DATASET_REVISION = "461c80411f8dac5ff82f4f363ee7db967305f7eb"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/data"


@dataclass(frozen=True)
class ParquetShard:
    filename: str
    size: int
    sha256: str


# Sizes and SHA-256 hashes are from this revision's Git LFS metadata.
POLLUX_SHARDS = (
    ParquetShard("test-00000-of-00003.parquet", 132570000,
                 "d5a2fd7e09e48f14aad154151ae386d06bd95137026e297e1b14239c1e4250ec"),
    ParquetShard("test-00001-of-00003.parquet", 132105524,
                 "0dee3be0f5dacd210a87575e3e697a9df06dc067480acdba994dafb5527348c0"),
    ParquetShard("test-00002-of-00003.parquet", 133777815,
                 "3682af7da443a72ff3f2f8ddd82f16f81e1db0fa8b13a9ca3e10c82e49d2fa98"),
)


def _matches_shard(path: Path, shard: ParquetShard) -> bool:
    if not path.is_file() or path.stat().st_size != shard.size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == shard.sha256


def ensure_pollux_shards(
    source_dir: Path,
    *,
    shards: tuple[ParquetShard, ...] = POLLUX_SHARDS,
    open_url: Callable = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Cache each verified shard; retry transient download errors, including HTTP 429."""
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for shard in shards:
        target = source_dir / shard.filename
        if _matches_shard(target, shard):
            paths.append(target)
            continue
        url = f"{BASE_URL}/{shard.filename}"
        partial = source_dir / f".{shard.filename}.{os.getpid()}.part"
        for attempt in range(5):
            partial.unlink(missing_ok=True)
            try:
                print(f"Downloading POLLUX shard {shard.filename} ({shard.size} bytes)", flush=True)
                request = Request(url, headers={"User-Agent": "jev-as-a-judge/1"})
                digest = hashlib.sha256()
                size = 0
                with open_url(request, timeout=60) as response, partial.open("wb") as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != shard.size or digest.hexdigest() != shard.sha256:
                    raise ValueError(
                        f"POLLUX shard {shard.filename} failed size/SHA-256 verification"
                    )
                partial.replace(target)
                paths.append(target)
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                partial.unlink(missing_ok=True)
                retryable = not isinstance(exc, HTTPError) or exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == 4:
                    raise RuntimeError(
                        f"Could not download {url}; place the three pinned Parquet shards "
                        f"in {source_dir} and rerun"
                    ) from exc
                retry_after = (
                    exc.headers.get("Retry-After")
                    if isinstance(exc, HTTPError) and exc.headers else None
                )
                delay = min(120, int(retry_after)) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
                print(f"POLLUX download error ({exc}); retrying in {delay}s", flush=True)
                sleep(delay)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
    return paths


def load_pollux_source(source_dir: Path):
    """Stream local Parquet shards, avoiding the rate-limited Hub tree API."""
    from datasets import load_dataset

    paths = ensure_pollux_shards(source_dir)
    return load_dataset(
        "parquet", data_files={"test": [str(path) for path in paths]},
        split="test", streaming=True,
    )
