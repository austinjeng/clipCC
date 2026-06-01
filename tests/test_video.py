import subprocess
import threading
import pytest
from app.services.video import (
    FrameExtractor, FrameSample, VideoInfo,
    probe_video, validate_video_constraints,
)

class TestProbeVideo:
    def test_probe_valid_video(self, small_video):
        info = probe_video(small_video, timeout=30)
        assert info.duration > 0
        assert info.width > 0
        assert info.height > 0
        assert info.video_stream_count == 1

    def test_probe_nonexistent_file(self, temp_dir):
        with pytest.raises(RuntimeError, match="ffprobe"):
            probe_video(temp_dir / "nonexistent.mp4", timeout=5)

class TestValidateVideoConstraints:
    def test_rejects_too_long(self, settings):
        info = VideoInfo(duration=999, width=320, height=240, video_stream_count=1, format_name="mov,mp4")
        with pytest.raises(Exception, match="minute"):
            validate_video_constraints(info, settings, fps=1.0)

    def test_rejects_too_many_frames(self, settings):
        info = VideoInfo(duration=29, width=320, height=240, video_stream_count=1, format_name="mov,mp4")
        with pytest.raises(Exception, match="frame"):
            validate_video_constraints(info, settings, fps=5.0)

    def test_rejects_high_resolution(self, settings):
        info = VideoInfo(duration=5, width=7680, height=4320, video_stream_count=1, format_name="mov,mp4")
        with pytest.raises(Exception, match="3840x2160"):
            validate_video_constraints(info, settings, fps=1.0)

    def test_rejects_multiple_streams(self, settings):
        info = VideoInfo(duration=5, width=320, height=240, video_stream_count=2, format_name="mov,mp4")
        with pytest.raises(Exception, match="video streams"):
            validate_video_constraints(info, settings, fps=1.0)

    def test_accepts_valid_video(self, settings):
        info = VideoInfo(duration=5, width=320, height=240, video_stream_count=1, format_name="mov,mp4")
        validate_video_constraints(info, settings, fps=1.0)

class TestFrameExtractor:
    def test_extract_frames(self, small_video, temp_dir):
        extractor = FrameExtractor(ffmpeg_timeout=30)
        frame_dir = temp_dir / "frames"
        frame_dir.mkdir()
        cancel = threading.Event()
        frames = extractor.extract(
            video_path=small_video, fps=1.0, max_frames=30,
            frame_dir=frame_dir, cancel_event=cancel,
        )
        assert len(frames) > 0
        assert all(isinstance(f, FrameSample) for f in frames)
        assert all(f.path.exists() for f in frames)
        assert frames[0].sample_index == 0
        assert frames[0].approx_timestamp_seconds == 0.0

    def test_extract_respects_cancel(self, small_video, temp_dir):
        extractor = FrameExtractor(ffmpeg_timeout=30)
        frame_dir = temp_dir / "frames"
        frame_dir.mkdir()
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(RuntimeError, match="cancel"):
            extractor.extract(
                video_path=small_video, fps=1.0, max_frames=30,
                frame_dir=frame_dir, cancel_event=cancel,
            )

    def test_extract_registers_process_with_runner(self, small_video, temp_dir):
        """extract() must register its ffmpeg Popen with the runner so a request
        timeout can kill it, and unregister it when done."""

        class _FakeRunner:
            def __init__(self):
                self.registered = []
                self.unregister_count = 0

            def register_process(self, proc):
                self.registered.append(proc)

            def unregister_process(self):
                self.unregister_count += 1

        extractor = FrameExtractor(ffmpeg_timeout=30)
        frame_dir = temp_dir / "frames"
        frame_dir.mkdir()
        cancel = threading.Event()
        runner = _FakeRunner()
        extractor.extract(
            video_path=small_video, fps=1.0, max_frames=30,
            frame_dir=frame_dir, cancel_event=cancel, runner=runner,
        )
        assert len(runner.registered) == 1
        assert isinstance(runner.registered[0], subprocess.Popen)
        assert runner.unregister_count == 1
