"""Causal uniform-grid windows with explicit missing samples and track resets."""

import math
from collections import defaultdict
from collections.abc import Sequence

from affective_video.config import DataConfig
from affective_video.data.manifest import SequenceExample, TimestampedObservation


def _segments(
    observations: list[TimestampedObservation],
) -> list[list[TimestampedObservation]]:
    segments: list[list[TimestampedObservation]] = [[]]
    active_track = None
    for observation in observations:
        if (
            observation.track_id is not None
            and active_track is not None
            and observation.track_id != active_track
        ):
            segments.append([])
        if observation.track_id is not None:
            active_track = observation.track_id
        segments[-1].append(observation)
    return segments


def build_sequences(
    observations: Sequence[TimestampedObservation], config: DataConfig
) -> list[SequenceExample]:
    """Use the latest source observation at/before each grid point, at most half a step old.

    Grid origin is the segment's first timestamp. Missing bins keep their positions;
    annotations are not interpolated. Stride counts grid endpoints, starting at zero.
    No padding is materialized: a future batch collator must mask padding separately.
    """
    groups: dict[tuple[str, str], list[TimestampedObservation]] = defaultdict(list)
    for observation in observations:
        groups[(observation.participant_id, observation.video_id)].append(observation)
    result = []
    step = 1.0 / config.sample_rate_hz
    length = round(config.sample_rate_hz * config.sequence_seconds)
    for (participant, video), group in sorted(groups.items()):
        group.sort(key=lambda item: item.timestamp)
        if any(a.timestamp == b.timestamp for a, b in zip(group, group[1:], strict=False)):
            raise ValueError(f"duplicate timestamp in participant {participant}, video {video}")
        for segment in _segments(group):
            origin = segment[0].timestamp
            count = math.floor((segment[-1].timestamp - origin) / step + 1e-9) + 1
            grid: list[TimestampedObservation] = []
            cursor = -1
            for index in range(count):
                timestamp = origin + index * step
                while cursor + 1 < len(segment) and segment[cursor + 1].timestamp <= timestamp:
                    cursor += 1
                source = segment[cursor] if cursor >= 0 else None
                if source is not None and timestamp - source.timestamp <= step / 2 + 1e-9:
                    grid.append(source.model_copy(update={"timestamp": timestamp}))
                else:
                    grid.append(
                        TimestampedObservation(
                            participant_id=participant, video_id=video, timestamp=timestamp
                        )
                    )
            for endpoint in range(0, count, config.stride):
                context = grid[max(0, endpoint - length + 1) : endpoint + 1]
                target = grid[endpoint]
                result.append(
                    SequenceExample(
                        participant_id=participant,
                        video_id=video,
                        track_id=segment[0].track_id,
                        timestamps=tuple(item.timestamp for item in context),
                        observation_mask=tuple(item.face_valid for item in context),
                        target_timestamp=target.timestamp,
                        target_valence=target.valence,
                        target_arousal=target.arousal,
                        target_valid_mask=(target.valence_valid, target.arousal_valid),
                    )
                )
    return result
