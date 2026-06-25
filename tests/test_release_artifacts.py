from scripts.package_release_artifacts import package_release_artifacts


def test_package_release_artifacts_writes_archive_and_manifest(tmp_path):
    source = tmp_path / "source"
    summary_dir = source / "repeated_experiments"
    benchmark_dir = summary_dir / "benchmark"
    benchmark_dir.mkdir(parents=True)
    (summary_dir / "summary.csv").write_text("metric,value\n", encoding="utf-8")
    (summary_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")
    (benchmark_dir / "benchmark.csv").write_text("device,latency\n", encoding="utf-8")

    result = package_release_artifacts(source, tmp_path / "release", package_name="artifacts")

    assert result["file_count"] == 3
    assert result["archive"].exists()
    assert result["checksum"].exists()
    assert result["manifest"].exists()
    assert "artifacts.zip" in result["checksum"].read_text(encoding="utf-8")


def test_package_release_artifacts_excludes_v1_3_checkpoints(tmp_path):
    source = tmp_path / "source"
    variant = source / "research_v1_3_low_contrast" / "variants" / "control_bce_dice"
    comparison = source / "research_v1_3_low_contrast" / "comparison"
    comparison.mkdir(parents=True)
    (comparison / "low_contrast_comparison.csv").write_text("variant,dice\n", encoding="utf-8")
    (variant / "checkpoints").mkdir(parents=True)
    (variant / "checkpoints" / "best_model.pth").write_bytes(b"checkpoint")
    (variant / "outputs" / "curves").mkdir(parents=True)
    (variant / "outputs" / "curves" / "training_curves.png").write_bytes(b"png")

    result = package_release_artifacts(source, tmp_path / "release", package_name="artifacts")
    manifest_text = result["manifest"].read_text(encoding="utf-8")

    assert "low_contrast_comparison.csv" in manifest_text
    assert "training_curves.png" in manifest_text
    assert "best_model.pth" not in manifest_text
