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
