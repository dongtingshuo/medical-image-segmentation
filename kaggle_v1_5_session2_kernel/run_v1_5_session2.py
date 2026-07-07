import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = WORKING_ROOT / "medical-image-segmentation"


def run(command, cwd=None):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    if REPOSITORY_ROOT.exists():
        run(["rm", "-rf", REPOSITORY_ROOT])
    run(["git", "clone", "--depth", "1", REPOSITORY_URL, REPOSITORY_ROOT])
    run(
        [
            sys.executable,
            "notebooks/kaggle_v1_5.py",
            "--use-existing-repo",
            "--allow-state-mismatch",
        ],
        cwd=REPOSITORY_ROOT,
    )


if __name__ == "__main__":
    main()
