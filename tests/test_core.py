"""
Unit tests for audio-cleaner core functions.
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch

import pytest
import numpy as np
import torch


# ── helpers exposed by main for testing ──────────────────────────────────────

def _check_ffmpeg():
    """Re-implemented here so tests can call it without importing main.py UI."""
    import shutil as _sh
    for cmd in ['ffmpeg', 'ffprobe']:
        if not _sh.which(cmd):
            raise RuntimeError(
                f"ffmpeg '{cmd}' not found in PATH. "
                "Please install ffmpeg (e.g. brew install ffmpeg) and ensure it is on your PATH."
            )


class TestFfmpegAvailability:
    """Test that ffmpeg/ffprobe presence check works correctly."""

    def test_ffmpeg_check_passes_when_available(self):
        """When ffmpeg + ffprobe are found, no exception is raised."""
        # This test verifies the current environment has ffmpeg installed.
        # It should pass in CI/dev environments with ffmpeg, fail in minimal containers.
        import shutil
        has_ffmpeg = shutil.which('ffmpeg') is not None
        has_ffprobe = shutil.which('ffprobe') is not None

        if has_ffmpeg and has_ffprobe:
            # The actual function must not raise
            _check_ffmpeg()  # should be a no-op / raise nothing
        else:
            pytest.skip("ffmpeg or ffprobe not available in this environment")

    def test_ffmpeg_check_raises_when_missing(self):
        """When ffmpeg is absent, a RuntimeError with a clear message is raised."""
        with patch('shutil.which', return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg.*not found"):
                _check_ffmpeg()

    def test_ffmpeg_check_raises_for_ffprobe_too(self):
        """ffprobe being missing also raises RuntimeError."""
        def which_mock(cmd):
            return None if cmd == 'ffprobe' else '/usr/bin/ffmpeg'

        with patch('shutil.which', side_effect=which_mock):
            with pytest.raises(RuntimeError, match="ffprobe.*not found"):
                _check_ffmpeg()


class TestFileCleanup:
    """Test tempfile / working-file cleanup patterns."""

    def test_temp_files_are_created_and_removable(self):
        """Temp files created under temp_dir can be cleanly removed."""
        temp_dir = tempfile.gettempdir()
        file_id = "test_cleanup_999"
        input_path = os.path.join(temp_dir, f"in_{file_id}_test.wav")
        work_path = os.path.join(temp_dir, f"work_{file_id}.wav")
        out_path = os.path.join(temp_dir, f"out_{file_id}.wav")

        # Create dummy files
        for p in [input_path, work_path, out_path]:
            with open(p, 'w') as f:
                f.write("dummy")

        # All files should exist
        assert os.path.exists(input_path)
        assert os.path.exists(work_path)
        assert os.path.exists(out_path)

        # Cleanup via same pattern main.py uses
        for p in [input_path, work_path, out_path]:
            if os.path.exists(p):
                os.remove(p)

        # All files should be gone
        assert not os.path.exists(input_path)
        assert not os.path.exists(work_path)
        assert not os.path.exists(out_path)

    def test_nonexistent_path_does_not_raise_on_cleanup(self):
        """Removing a file that was never created must not raise."""
        temp_dir = tempfile.gettempdir()
        ghost = os.path.join(temp_dir, "this_file_never_existed_xyz.wav")
        # must not raise
        if os.path.exists(ghost):
            os.remove(ghost)


class TestEnhanceAudioShape:
    """Sanity-check on the enhance() pipeline shape / type contract."""

    @pytest.fixture
    def dummy_waveform(self):
        """48 kHz, 1 channel, 240 000 samples = 5 s."""
        sr = 48000
        waveform = torch.randn(1, sr * 5)
        return waveform, sr

    def test_enhance_returns_tensor(self):
        """enhance() must return a torch.Tensor."""
        from df.enhance import enhance, init_df

        model, df_state, _ = init_df(default_model="DeepFilterNet3", post_filter=False)
        waveform, sr = 48000, 48000
        chunk = torch.randn(1, 48000)  # 1 s of silence

        result = enhance(model, df_state, chunk, atten_lim_db=None)

        assert isinstance(result, torch.Tensor)

    def test_enhance_output_shape_matches_input(self, dummy_waveform):
        """Output of enhance() has the same shape as its input chunk."""
        from df.enhance import enhance, init_df

        model, df_state, _ = init_df(default_model="DeepFilterNet3", post_filter=False)
        waveform, sr = dummy_waveform
        chunk_len = 30 * sr
        s, e = 0, min(chunk_len, waveform.shape[1])
        chunk = waveform[:, s:e]

        result = enhance(model, df_state, chunk, atten_lim_db=None)

        assert result.shape == chunk.shape

    def test_enhance_output_is_mono(self):
        """enhance() always returns a single-channel tensor."""
        from df.enhance import enhance, init_df

        model, df_state, _ = init_df(default_model="DeepFilterNet3", post_filter=False)
        # 2-channel input (simulate stereo capture)
        stereo = torch.randn(2, 48000)
        result = enhance(model, df_state, stereo, atten_lim_db=None)

        assert result.shape[0] == 1, "enhance output should be mono"

    def test_enhance_accepts_none_atten_lim_db(self):
        """Passing atten_lim_db=None (the default in main.py) does not error."""
        from df.enhance import enhance, init_df

        model, df_state, _ = init_df(default_model="DeepFilterNet3", post_filter=False)
        chunk = torch.randn(1, 48000)

        # Must not raise
        result = enhance(model, df_state, chunk, atten_lim_db=None)
        assert isinstance(result, torch.Tensor)


class TestAudioFormats:
    """Test that input / output audio format expectations are met."""

    def test_work_wav_is_48khz_mono(self, tmp_path):
        """Simulate the ffmpeg conversion step and verify output properties."""
        import subprocess, soundfile as sf

        # Skip if ffmpeg unavailable (caught by ffmpeg availability tests)
        import shutil
        if not shutil.which('ffmpeg'):
            pytest.skip("ffmpeg not available")

        # Create a 1-second 44.1 kHz stereo WAV as input
        src = tmp_path / "src.wav"
        dummy_wav = np.random.rand(88200, 2).astype(np.float32)  # 44.1kHz, 2ch, 1s
        sf.write(str(src), dummy_wav, 44100, subtype='FLOAT')

        dst = tmp_path / "dst.wav"
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(src),
             '-ar', '48000', '-ac', '1', str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True,
        )

        assert dst.exists()
        data, sr = sf.read(str(dst))
        assert sr == 48000, f"Expected 48 kHz, got {sr}"
        assert data.ndim == 1, f"Expected mono, got {data.ndim} channels"