"""Unit tests for pawpaw.runtime — no real Llama model required."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import llama_cpp
from pawpaw.runtime import (
    PLACEHOLDER,
    _MAX_PAW_SIZE,
    _Session,
    _auto_n_gpu_layers,
    _paw_cache_dir,
    _resolve_bundle,
    _unpack_paw,
    clear_session_cache,
)
from pawpaw.format import MAGIC, VERSION


class TestAutoNGpuLayers:
    def test_returns_zero_on_os_error(self):
        with patch.object(llama_cpp, "llama_supports_gpu_offload", side_effect=OSError):
            assert _auto_n_gpu_layers() == 0


class TestUnpackPaw:
    def test_rejects_oversized_file(self, tmp_path):
        big = tmp_path / "big.paw"
        big.write_bytes(MAGIC + struct.pack("<I", VERSION) + struct.pack("<I", 0))
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=_MAX_PAW_SIZE + 1)
            with pytest.raises(ValueError, match="too large"):
                _unpack_paw(big)

    def test_reuses_cached_bundle(self, tmp_path):
        paw = tmp_path / "test.paw"
        meta = {"interpreter_model": "Qwen/Qwen3-0.6B"}
        meta_bytes = json.dumps(meta).encode("utf-8")
        paw.write_bytes(MAGIC + struct.pack("<I", VERSION) + struct.pack("<I", len(meta_bytes)) + meta_bytes + b"\x00" * 16)

        with patch("pawpaw.runtime._paw_cache_dir", return_value=tmp_path / "cache"):
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()

            file_hash = "abc123"
            bundle_dir = cache_dir / file_hash
            bundle_dir.mkdir()
            (bundle_dir / "meta.json").write_text("{}")

            with patch("hashlib.sha256") as mock_sha:
                mock_sha.return_value.hexdigest.return_value = file_hash
                result = _unpack_paw(paw)
                assert result == bundle_dir


class TestResolveBundle:
    def test_raises_on_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Could not find"):
            _resolve_bundle(tmp_path / "nonexistent")

    def test_accepts_bundle_directory(self, tmp_path):
        d = tmp_path / "myprog"
        d.mkdir()
        (d / "meta.json").write_text('{"spec": "test"}')
        (d / "prompt_template.txt").write_text("hello")
        bundle, meta, template = _resolve_bundle(d)
        assert bundle == d
        assert meta["spec"] == "test"
        assert template == "hello"

    def test_accepts_paw_file(self, tmp_path):
        paw = tmp_path / "test.paw"
        paw.write_bytes(MAGIC + struct.pack("<I", VERSION) + struct.pack("<I", 0))
        with patch("pawpaw.runtime._unpack_paw") as mock_unpack:
            mock_dir = tmp_path / "unpacked"
            mock_dir.mkdir()
            (mock_dir / "meta.json").write_text('{"spec": "x"}')
            (mock_dir / "prompt_template.txt").write_text("y")
            mock_unpack.return_value = mock_dir
            bundle, meta, template = _resolve_bundle(paw)
            assert bundle == mock_dir


class TestClearSessionCache:
    def test_clears_sessions(self):
        from pawpaw.runtime import _sessions
        _sessions[("test", 0, 0)] = MagicMock()
        clear_session_cache()
        assert len(_sessions) == 0
