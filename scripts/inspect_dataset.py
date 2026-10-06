"""Inventory an authorized local release; audit only explicitly documented mappings."""

import argparse
import json
import os
from pathlib import Path

import yaml

from affective_video.config import load_config
from affective_video.data.adapters.recola import RecolaDatasetAdapter
from affective_video.data.adapters.recola_provenance import ChecksumCache
from affective_video.data.adapters.recola_schema import ReleaseMapping
from affective_video.data.recola_reporting import dataset_summary, plot_dataset, write_manifest
from affective_video.data.recola_splits import audit_split, freeze_split, manifest_hash
from affective_video.utils.run_metadata import canonical_hash, save_json


def inspect(
    config_path: Path,
    root: Path | None,
    mapping_path: Path | None = None,
    *,
    make_split: bool = False,
    report: bool = False,
    rehash: bool = False,
) -> tuple[int, dict]:
    config = load_config(config_path)
    if config.data.dataset_name != "recola":
        raise ValueError("use configs/recola.yaml; M1 synthetic configuration remains unchanged")
    manifest_path, split_path = config.data.manifest_path, config.data.split_path
    if manifest_path is None or split_path is None:
        raise ValueError("manifest_path and split_path are required")
    output = manifest_path.parent
    output.mkdir(parents=True, exist_ok=True)
    if root is None or not root.is_dir():
        result = {
            "status": "BLOCKED_ON_RECOLA_ACCESS",
            "real_release_inspected": False,
            "reason": "No authorized local release directory was supplied or found",
            "eligible_participants": None,
            "eligible_recordings": None,
            "manifest_generated": False,
            "split_generated": False,
        }
        save_json(output / "recola_inspection.json", result)
        return 2, result
    root = root.resolve()
    repository = Path(__file__).resolve().parents[1]
    figure_dir = repository / "reports/figures/dataset"
    for destination in (output.resolve(), split_path.parent.resolve(), figure_dir.resolve()):
        if destination.is_relative_to(root):
            raise ValueError("inspection outputs must be outside the release root")
    mapping = (
        ReleaseMapping.model_validate(yaml.safe_load(mapping_path.read_text(encoding="utf-8")))
        if mapping_path
        else None
    )
    cache = ChecksumCache(root, output / "recola_checksums.json", rehash=rehash)
    adapter = RecolaDatasetAdapter(root, mapping, cache)
    rows = adapter.build_manifest()
    validation = adapter.validate_release()
    split = None
    if make_split and not validation["critical_failure"]:
        split = freeze_split(split_path, rows, config.training.seed)
    elif split_path.exists() and mapping is not None and not validation["critical_failure"]:
        split = freeze_split(split_path, rows, config.training.seed)
    summary = dataset_summary(adapter, rows, split)
    if mapping is not None:
        write_manifest(manifest_path, rows)
    result = {
        "status": "AUDITED"
        if rows and summary["eligible_recordings"] and split and not validation["critical_failure"]
        else "INCOMPLETE_RELEASE_AUDIT",
        "real_release_inspected": True,
        "root": str(root),
        "manifest_generated": mapping is not None,
        "manifest_hash": manifest_hash(rows) if mapping is not None else None,
        "mapping_hash": canonical_hash(mapping.model_dump(mode="json")) if mapping else None,
        "split_generated": split is not None,
        "validation": validation,
        "summary": summary,
        "video_metadata_method": "supplied frame timestamps; no video-container decode/probe",
        "target_construction": "official consensus only; individual averaging is not approved",
    }
    if split:
        audit = audit_split(split, rows)
        save_json(output / "recola_split_audit.json", audit)
        lines = [
            "# Private internal split audit",
            "",
            "Available-identifier leakage audit: PASS",
            "",
        ]
        for partition, statistics in audit["partitions"].items():
            lines.extend(
                [f"## {partition}", "", "```json", json.dumps(statistics, indent=2), "```", ""]
            )
        (output / "recola_split_audit.md").write_text("\n".join(lines), encoding="utf-8")
    if report:
        result["figure_files"] = plot_dataset(adapter, rows, summary, figure_dir, split)
    save_json(output / "recola_inspection.json", result)
    save_json(output / "recola_statistics.json", summary)
    return (0 if result["status"] == "AUDITED" else 2), result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/recola.yaml"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--make-split", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--rehash", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root or (
        Path(os.environ["RECOLA_ROOT"]) if os.environ.get("RECOLA_ROOT") else None
    )
    try:
        code, result = inspect(
            arguments.config,
            root,
            arguments.mapping,
            make_split=arguments.make_split,
            report=arguments.report,
            rehash=arguments.rehash,
        )
        summary = result.get("summary", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "eligible_participants": summary.get("eligible_participants"),
                    "eligible_recordings": summary.get("eligible_recordings"),
                    "manifest_generated": result["manifest_generated"],
                    "split_generated": result["split_generated"],
                },
                indent=2,
            )
        )
        return code
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(json.dumps({"status": "INSPECTION_FAILED", "reason": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
