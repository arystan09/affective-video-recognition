"""Seeded infrastructure fixtures. Trajectories are not human affect annotations."""

import numpy as np

from affective_video.data.manifest import ParticipantRecord, Partition, TimestampedObservation


def generate_synthetic(
    seed: int, *, irregular: bool = False
) -> tuple[list[ParticipantRecord], list[TimestampedObservation]]:
    """Six independent participants, two videos each, gaps and a constant-label case."""
    rng = np.random.default_rng(seed)
    manifest = []
    observations = []
    for participant_index in range(6):
        participant = f"P{participant_index:02d}"
        partition: Partition = ("train", "train", "validation", "validation", "test", "test")[
            participant_index
        ]
        for video_index in range(2):
            video = f"{participant}_V{video_index}"
            manifest.append(
                ParticipantRecord(
                    participant_id=participant,
                    video_id=video,
                    session_id=f"S{participant_index}",
                    dyad_id=f"D{participant_index}",
                    partition=partition,
                    source_checksum=f"synthetic-{video}",
                )
            )
            phase = float(rng.uniform(-1, 1))
            for index in range(40):
                timestamp = index / 5
                if irregular and index > 0:
                    timestamp += float(rng.uniform(-0.025, 0.025))
                constant = participant_index == 5
                valence = 0.2 if constant else float(0.7 * np.sin(timestamp / 2 + phase))
                arousal = -0.1 if constant else float(0.6 * np.cos(timestamp / 3 + phase))
                valence_valid, arousal_valid = index % 13 != 5, index % 11 != 4
                observations.append(
                    TimestampedObservation(
                        participant_id=participant,
                        video_id=video,
                        timestamp=timestamp,
                        valence=valence if valence_valid else None,
                        arousal=arousal if arousal_valid else None,
                        valence_valid=valence_valid,
                        arousal_valid=arousal_valid,
                        face_valid=index % 9 != 3,
                        track_id=f"{video}_track0",
                    )
                )
    return manifest, observations
