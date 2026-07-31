"""Deterministic action schedule for the quantitative PICO recording."""

from __future__ import annotations

from dataclasses import asdict, dataclass


QUANTITATIVE_ACTIONS: tuple[str, ...] = (
    "fist",
    "thumb_index_pinch",
    "thumb_middle_pinch",
    "thumb_ring_pinch",
    "thumb_little_pinch",
    "tripod_pinch",
    "thumb_opposition",
)


@dataclass(frozen=True, slots=True)
class ProtocolDurations:
    prepare_s: float = 1.0
    transition_s: float = 1.0
    hold_s: float = 1.5
    release_s: float = 1.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0")


@dataclass(frozen=True, slots=True)
class ProtocolEpisode:
    action: str
    repetition: int
    start_frame: int
    transition_start_frame: int
    hold_start_frame: int
    hold_end_frame_exclusive: int
    end_frame_exclusive: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def build_protocol_episodes(
    *,
    sample_rate_hz: int,
    repetitions: int = 5,
    durations: ProtocolDurations = ProtocolDurations(),
) -> tuple[ProtocolEpisode, ...]:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if repetitions <= 0:
        raise ValueError("repetitions must be > 0")
    durations.validate()

    prepare_frames = max(1, round(sample_rate_hz * durations.prepare_s))
    transition_frames = max(1, round(sample_rate_hz * durations.transition_s))
    hold_frames = max(1, round(sample_rate_hz * durations.hold_s))
    release_frames = max(1, round(sample_rate_hz * durations.release_s))

    episodes: list[ProtocolEpisode] = []
    cursor = 0
    for action in QUANTITATIVE_ACTIONS:
        for repetition in range(1, repetitions + 1):
            transition_start = cursor + prepare_frames
            hold_start = transition_start + transition_frames
            hold_end = hold_start + hold_frames
            end = hold_end + release_frames
            episodes.append(
                ProtocolEpisode(
                    action=action,
                    repetition=repetition,
                    start_frame=cursor,
                    transition_start_frame=transition_start,
                    hold_start_frame=hold_start,
                    hold_end_frame_exclusive=hold_end,
                    end_frame_exclusive=end,
                )
            )
            cursor = end
    return tuple(episodes)


def protocol_total_frames(episodes: tuple[ProtocolEpisode, ...]) -> int:
    if not episodes:
        return 0
    return episodes[-1].end_frame_exclusive


def protocol_phase_at_frame(
    frame_index: int,
    episodes: tuple[ProtocolEpisode, ...],
) -> tuple[ProtocolEpisode, str]:
    if frame_index < 0:
        raise ValueError("frame_index must be >= 0")
    episode = next(
        (candidate for candidate in episodes if frame_index < candidate.end_frame_exclusive),
        None,
    )
    if episode is None or frame_index < episode.start_frame:
        raise IndexError(f"frame_index {frame_index} is outside the protocol")
    if frame_index < episode.transition_start_frame:
        phase = "prepare"
    elif frame_index < episode.hold_start_frame:
        phase = "transition"
    elif frame_index < episode.hold_end_frame_exclusive:
        phase = "hold"
    else:
        phase = "release"
    return episode, phase
