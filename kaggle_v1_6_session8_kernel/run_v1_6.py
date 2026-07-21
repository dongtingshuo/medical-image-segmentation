import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"
STATE_SOURCE_COMMIT = "0d584ff80ba4b41b69e17e6ebfa3422675b3b422"


def run(command, cwd=None):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


working_root = Path.cwd().resolve()
source_root = Path(__file__).resolve().parent
output_root = working_root / "research_v1_6"
for source_name, target_name in (
    ("locked_decision.json", "selection/locked_decision.json"),
    ("evaluation_complete.json", "final/evaluation_complete.json"),
    ("state_lineage.json", "state_lineage.json"),
):
    source = source_root / "recovery" / source_name
    target = output_root / target_name
    if not source.exists():
        raise FileNotFoundError(f"Missing locked Session 8 recovery artifact: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
repository = working_root / "medical-image-segmentation"
run(["git", "clone", "--depth", "1", REPOSITORY_URL, repository])
run(["git", "-C", repository, "fetch", "--depth", "1", "origin", STATE_SOURCE_COMMIT])
run(["git", "-C", repository, "checkout", "--detach", STATE_SOURCE_COMMIT])
run(
    [
        sys.executable,
        "notebooks/kaggle_v1_6.py",
        "--use-existing-repo",
        "--allow-state-mismatch",
    ],
    cwd=repository,
)
