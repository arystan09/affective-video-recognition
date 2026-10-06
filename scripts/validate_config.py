"""Validate YAML and display the fully resolved configuration."""

import argparse
import json
from pathlib import Path

from affective_video.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(load_config(arguments.config).model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
