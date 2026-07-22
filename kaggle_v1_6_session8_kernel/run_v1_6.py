import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"
STATE_SOURCE_COMMIT = "1b24795bdf74664b24e5eb05b4bc12a8f21e104e"


def run(command, cwd=None):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


working_root = Path.cwd().resolve()
repository = working_root / "medical-image-segmentation"
run(["git", "clone", "--depth", "1", REPOSITORY_URL, repository])
run(["git", "-C", repository, "fetch", "--depth", "1", "origin", STATE_SOURCE_COMMIT])
run(["git", "-C", repository, "checkout", "--detach", STATE_SOURCE_COMMIT])

source_root = repository / "kaggle_v1_6_session8_kernel/recovery"
output_root = working_root / "research_v1_6"
for source_name, target_name in (
    ("locked_decision.json", "selection/locked_decision.json"),
    ("evaluation_complete.json", "final/evaluation_complete.json"),
    ("state_lineage.json", "state_lineage.json"),
):
    source = source_root / source_name
    target = output_root / target_name
    if not source.exists():
        raise FileNotFoundError(f"Missing locked Session 8 recovery artifact: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
run(
    [
        sys.executable,
        "notebooks/kaggle_v1_6.py",
        "--use-existing-repo",
        "--allow-state-mismatch",
    ],
    cwd=repository,
)
