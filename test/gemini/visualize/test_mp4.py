"""Validation of the shared MP4 export options."""

import pytest
from matplotlib import animation

from bloqade.gemini.visualize._mp4 import mp4_writer, hold_frames


def test_mp4_writer_rejects_invalid_fps(tmp_path):
    with pytest.raises(ValueError, match="fps"):
        mp4_writer(tmp_path / "moves.mp4", fps=0)


def test_mp4_writer_rejects_non_mp4_path(tmp_path):
    with pytest.raises(ValueError, match=".mp4"):
        mp4_writer(tmp_path / "moves.gif", fps=30)


def test_mp4_writer_rejects_existing_file(tmp_path):
    output = tmp_path / "moves.mp4"
    output.write_bytes(b"original")

    with pytest.raises(FileExistsError, match="already exists"):
        mp4_writer(output, fps=30)

    assert output.read_bytes() == b"original"


def test_mp4_writer_reports_missing_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(animation.writers, "is_available", lambda _name: False)

    with pytest.raises(RuntimeError, match="FFmpeg"):
        mp4_writer(tmp_path / "moves.mp4", fps=30)


def test_hold_frames_rejects_negative_pause():
    with pytest.raises(ValueError, match="pause_time"):
        hold_frames(-0.1, 30)


def test_hold_frames_keeps_zero_duration_step_visible():
    assert hold_frames(0, 30) == 1
