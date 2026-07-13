import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"
STATE_SOURCE_COMMIT = "8158479b7d78bddcd1a7c5d660cc6ac356985b19"


def run(command, cwd=None):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


working_root = Path.cwd().resolve()
repository = working_root / "medical-image-segmentation"
run(["git", "clone", "--depth", "1", REPOSITORY_URL, repository])
run(["git", "-C", repository, "fetch", "--depth", "1", "origin", STATE_SOURCE_COMMIT])
run(["git", "-C", repository, "checkout", "--detach", STATE_SOURCE_COMMIT])
run([sys.executable, "notebooks/kaggle_v1_6.py", "--use-existing-repo"], cwd=repository)
