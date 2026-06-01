from __future__ import annotations

from fastapi import HTTPException


class FileTooLargeError(HTTPException):
    def __init__(self, size_mb: float, max_mb: float) -> None:
        super().__init__(
            status_code=413,
            detail=f"File size {size_mb:.1f} MB exceeds the maximum allowed size of {max_mb:.1f} MB.",
        )


class DurationTooLongError(HTTPException):
    def __init__(self, duration_seconds: float, max_seconds: float) -> None:
        max_minutes = max_seconds / 60
        super().__init__(
            status_code=422,
            detail=(
                f"Video duration {duration_seconds:.1f}s exceeds the maximum allowed duration "
                f"of {max_seconds:.1f}s ({max_minutes:.1f} minute(s))."
            ),
        )


class TooManyFramesError(HTTPException):
    def __init__(
        self,
        frame_count: int,
        max_frames: int,
        duration: float,
        fps: float,
    ) -> None:
        super().__init__(
            status_code=422,
            detail=(
                f"Extracting {frame_count} frames (duration={duration:.1f}s, fps={fps}) "
                f"exceeds the maximum of {max_frames} frames. "
                f"Reduce fps or use a shorter video."
            ),
        )


class ResolutionTooHighError(HTTPException):
    def __init__(self, width: int, height: int) -> None:
        super().__init__(
            status_code=422,
            detail=f"Video resolution {width}x{height} exceeds the maximum supported resolution of 3840x2160.",
        )


class MultipleVideoStreamsError(HTTPException):
    def __init__(self, stream_count: int) -> None:
        super().__init__(
            status_code=422,
            detail=f"Video contains {stream_count} video streams; only single-stream videos are supported.",
        )


class InvalidLabelsError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=422, detail=detail)


class InvalidPromptTemplateError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=422, detail=detail)


class InvalidFpsError(HTTPException):
    def __init__(self, fps: float) -> None:
        super().__init__(
            status_code=422,
            detail=f"FPS value {fps} is invalid. Must be between 0.1 and 5.0.",
        )


class InvalidAggregationError(HTTPException):
    def __init__(self, aggregation: str) -> None:
        super().__init__(
            status_code=422,
            detail=f"Aggregation method '{aggregation}' is not supported.",
        )


class UnsupportedFormatError(HTTPException):
    def __init__(self, extension: str) -> None:
        super().__init__(
            status_code=415,
            detail=f"File format '{extension}' is not supported.",
        )


class NoFramesExtractedError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=422,
            detail=(
                "No frames could be extracted from the video. The clip may be "
                "too short for the requested fps, empty, or corrupt."
            ),
        )


class TokenTruncationError(HTTPException):
    def __init__(self, prompt: str, token_count: int) -> None:
        super().__init__(
            status_code=422,
            detail=f"Prompt '{prompt}' has {token_count} tokens and will be truncated by the CLIP tokenizer.",
        )


class DuplicateTokensError(HTTPException):
    def __init__(self, label_a: str, label_b: str) -> None:
        super().__init__(
            status_code=422,
            detail=f"Labels '{label_a}' and '{label_b}' produce identical token sequences and cannot be distinguished.",
        )


class InferenceTimeoutError(HTTPException):
    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(
            status_code=504,
            detail=f"Inference timed out after {timeout_seconds:.1f}s.",
        )


class UploadConcurrencyError(Exception):
    pass


class InferenceConcurrencyError(Exception):
    pass


class InvalidTemporalParamsError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=422,
            detail="Parameters 'threshold', 'gap_tolerance', 'min_duration' are only valid with aggregation='temporal'.",
        )


class InvalidContrastParamsError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=422, detail=detail)
