import subprocess
import sys
from pathlib import Path


REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = WORKING_ROOT / "medical-image-segmentation"


def run(command, cwd=None):
    print(">>>", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def main():
    if REPOSITORY_ROOT.exists():
        run(["rm", "-rf", REPOSITORY_ROOT])
    run(["git", "clone", "--depth", "1", REPOSITORY_URL, REPOSITORY_ROOT])
    run([sys.executable, "notebooks/kaggle_aggressive_v1_4.py"], cwd=REPOSITORY_ROOT)


if __name__ == "__main__":
    main()
