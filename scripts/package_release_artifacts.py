import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PATTERNS = (
    "repeated_experiments/summary.csv",
    "repeated_experiments/summary.md",
    "repeated_experiments/all_seed_metrics.csv",
    "repeated_experiments/benchmark/benchmark.csv",
    "repeated_experiments/benchmark/benchmark.md",
    "repeated_experiments/manifest.json",
    "repeated_experiments/execution_manifest.json",
    "repeated_experiments/internal_data_report.json",
    "repeated_experiments/external_data_report.json",
    "repeated_experiments/preflight/sanity_check/dataset_check_report.md",
    "repeated_experiments/preflight/sanity_check/*.png",
    "repeated_experiments/seed_*/outputs/curves/*.png",
    "repeated_experiments/seed_*/outputs/*metrics.csv",
    "repeated_experiments/seed_*/outputs/experiment_results.csv",
    "repeated_experiments/seed_*/outputs/samples/*overlay.png",
    "posthoc_analysis/**/*.csv",
    "posthoc_analysis/**/*.json",
    "posthoc_analysis/**/*.md",
    "posthoc_analysis/**/*.png",
    "research_v1_2/execution_manifest.json",
    "research_v1_2/preflight/sanity_check/dataset_check_report.md",
    "research_v1_2/preflight/sanity_check/*.png",
    "research_v1_2/cross_validation/*.csv",
    "research_v1_2/cross_validation/*.json",
    "research_v1_2/cross_validation/*.md",
    "research_v1_2/cross_validation/fold_*/outputs/curves/*.png",
    "research_v1_2/cross_validation/fold_*/outputs/*metrics.csv",
    "research_v1_2/cross_validation/fold_*/outputs/experiment_results.csv",
    "research_v1_2/cross_validation/fold_*/outputs/training_history.csv",
    "research_v1_2/cross_validation/fold_*/outputs/samples/*overlay.png",
    "research_v1_2/encoder_comparison/*.csv",
    "research_v1_2/encoder_comparison/*.json",
    "research_v1_2/encoder_comparison/*.md",
    "research_v1_2/encoder_comparison/*/outputs/curves/*.png",
    "research_v1_2/encoder_comparison/*/outputs/*metrics.csv",
    "research_v1_2/encoder_comparison/*/outputs/experiment_results.csv",
    "research_v1_2/encoder_comparison/*/outputs/training_history.csv",
    "research_v1_2/encoder_comparison/*/outputs/samples/*overlay.png",
    "research_v1_2/threshold_search/*.csv",
    "research_v1_2/threshold_search/*.json",
    "research_v1_2/threshold_search/*.md",
    "research_v1_2/subgroup_analysis_*/*.csv",
    "research_v1_2/subgroup_analysis_*/*.json",
    "research_v1_2/subgroup_analysis_*/*.md",
    "research_v1_2/statistics_*/*.csv",
    "research_v1_2/statistics_*/*.json",
    "research_v1_2/statistics_*/*.md",
    "research_v1_3_low_contrast/execution_manifest.json",
    "research_v1_3_low_contrast/kaggle_low_contrast_v1_3_runtime_base.yaml",
    "research_v1_3_low_contrast/comparison/*.csv",
    "research_v1_3_low_contrast/comparison/*.json",
    "research_v1_3_low_contrast/comparison/*.md",
    "research_v1_3_low_contrast/variants/*/runtime_config.yaml",
    "research_v1_3_low_contrast/variants/*/completed.json",
    "research_v1_3_low_contrast/variants/*/train.log",
    "research_v1_3_low_contrast/variants/*/outputs/curves/*.png",
    "research_v1_3_low_contrast/variants/*/outputs/training_history.csv",
    "research_v1_3_low_contrast/variants/*/outputs/experiment_results.csv",
    "research_v1_3_low_contrast/variants/*/outputs/samples/*overlay.png",
    "research_v1_3_low_contrast/variants/*/threshold_search/*.csv",
    "research_v1_3_low_contrast/variants/*/threshold_search/*.json",
    "research_v1_3_low_contrast/variants/*/threshold_search/*.md",
    "research_v1_3_low_contrast/variants/*/evaluation_*/*.csv",
    "research_v1_3_low_contrast/variants/*/subgroup_*/*.csv",
    "research_v1_3_low_contrast/variants/*/subgroup_*/*.json",
    "research_v1_3_low_contrast/variants/*/subgroup_*/*.md",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifact_files(source_root, patterns=DEFAULT_PATTERNS):
    source_root = Path(source_root)
    files = []
    for pattern in patterns:
        files.extend(path for path in source_root.glob(pattern) if path.is_file())
    return sorted(set(files))


def package_release_artifacts(source_root, output_dir, package_name="medical-segmentation-experiment-artifacts"):
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = collect_artifact_files(source_root)
    if not files:
        raise FileNotFoundError(f"No release artifact files found under {source_root}")
    staging_dir = output_dir / package_name
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    manifest = []
    for path in files:
        relative_path = path.relative_to(source_root)
        destination = staging_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest.append(
            {
                "path": str(relative_path),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    manifest_path = staging_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive_path = output_dir / f"{package_name}.zip"
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(staging_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(output_dir))
    checksum_path = output_dir / f"{archive_path.name}.sha256"
    checksum_path.write_text(f"{sha256_file(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    return {
        "archive": archive_path,
        "checksum": checksum_path,
        "manifest": manifest_path,
        "file_count": len(manifest),
    }


def main():
    parser = argparse.ArgumentParser(description="Package experiment reports and visual artifacts for GitHub Release.")
    parser.add_argument("--source-root", default="kaggle_outputs/repeated_experiment")
    parser.add_argument("--output-dir", default="release_artifacts")
    parser.add_argument("--package-name", default="medical-segmentation-experiment-artifacts")
    args = parser.parse_args()
    result = package_release_artifacts(args.source_root, args.output_dir, package_name=args.package_name)
    print(f"Packaged {result['file_count']} files")
    print(f"Archive: {result['archive']}")
    print(f"Checksum: {result['checksum']}")
    print(f"Manifest: {result['manifest']}")


if __name__ == "__main__":
    main()
