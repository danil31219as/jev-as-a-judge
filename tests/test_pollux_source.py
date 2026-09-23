import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError

from pollux_source import ParquetShard, ensure_pollux_shards


class PolluxSourceTests(unittest.TestCase):
    def test_download_retries_429_and_reuses_verified_cache(self):
        payload = b"pinned parquet test contents"
        shard = ParquetShard(
            "test-00000-of-00001.parquet", len(payload), hashlib.sha256(payload).hexdigest()
        )
        requests = []
        delays = []

        def open_url(request, timeout):
            requests.append((request.full_url, timeout))
            if len(requests) == 1:
                raise HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "1"}, None)
            return BytesIO(payload)

        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory)
            paths = ensure_pollux_shards(
                source_dir, shards=(shard,), open_url=open_url, sleep=delays.append
            )
            self.assertEqual(paths, [source_dir / shard.filename])
            self.assertEqual(paths[0].read_bytes(), payload)
            self.assertEqual(len(requests), 2)
            self.assertEqual(delays, [1])
            self.assertEqual(
                ensure_pollux_shards(source_dir, shards=(shard,), open_url=open_url), paths
            )
            self.assertEqual(len(requests), 2)


if __name__ == "__main__":
    unittest.main()
