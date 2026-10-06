"""Exercise M1 on CPU. Fabricated predictions are not scientific results."""

import argparse
import json
from pathlib import Path

import numpy as np

from affective_video.config import load_config
from affective_video.data.manifest import PredictionRecord
from affective_video.data.sequences import build_sequences
from affective_video.data.splits import validate_splits
from affective_video.data.synthetic import generate_synthetic
from affective_video.evaluation.metrics import evaluate_predictions
from affective_video.utils.reproducibility import set_global_seed
from affective_video.utils.run_metadata import create_run, save_json


def run(config_path: Path, *, irregular: bool = False) -> Path:
    config = load_config(config_path)
    if config.data.dataset_name != "synthetic":
        raise ValueError("synthetic_run requires data.dataset_name=synthetic")
    set_global_seed(config.training.seed, deterministic=config.training.deterministic)
    manifest, observations = generate_synthetic(config.training.seed, irregular=irregular)
    validate_splits(manifest)
    sequences = build_sequences(observations, config.data)
    partitions = {(item.participant_id, item.video_id): item.partition for item in manifest}
    rng = np.random.default_rng(config.training.seed)
    predictions = []
    for sequence in sequences:
        predictions.append(
            PredictionRecord(
                participant_id=sequence.participant_id,
                video_id=sequence.video_id,
                timestamp=sequence.target_timestamp,
                target_valence=sequence.target_valence,
                target_arousal=sequence.target_arousal,
                target_valid_mask=sequence.target_valid_mask,
                predicted_valence=float(
                    np.clip((sequence.target_valence or 0) + rng.normal(0, 0.1), -1, 1)
                ),
                predicted_arousal=float(
                    np.clip((sequence.target_arousal or 0) + rng.normal(0, 0.1), -1, 1)
                ),
                valid=sequence.observation_mask[-1],
            )
        )
    metrics = {
        "infrastructure_only": True,
        "prediction_method": "reference labels plus seeded noise; not a model",
        "synthetic": {"irregular": irregular, "observations": len(observations)},
        "sequence_count": len(sequences),
        "partitions": {
            partition: evaluate_predictions(
                [
                    item
                    for item in predictions
                    if partitions[(item.participant_id, item.video_id)] == partition
                ],
                config.data.target_dimensions,
                participant_macro_average=config.evaluation.participant_macro_average,
            ).to_dict()
            for partition in ("train", "validation", "test")
        },
    }
    directory, metadata = create_run(config, manifest, Path(__file__).resolve().parents[1])
    save_json(directory / "metrics.json", metrics)
    save_json(
        directory / "predictions.json", [item.model_dump(mode="json") for item in predictions]
    )
    print(
        json.dumps(
            {
                "infrastructure_only": True,
                "run_id": metadata.run_id,
                "output_dir": str(directory),
                "participants": len({r.participant_id for r in manifest}),
                "videos": len(manifest),
                "observations": len(observations),
                "sequences": len(sequences),
                "test_valid_counts": metrics["partitions"]["test"]["valid_counts"],
            },
            indent=2,
        )
    )
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--irregular", action="store_true")
    arguments = parser.parse_args()
    run(arguments.config, irregular=arguments.irregular)


if __name__ == "__main__":
    main()
