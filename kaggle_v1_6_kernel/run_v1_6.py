import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"


def run(command, cwd=None):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


working_root = Path.cwd().resolve()
repository = working_root / "medical-image-segmentation"
if repository.exists():
    run(["git", "-C", repository, "fetch", "--depth", "1", "origin", "main"])
    run(["git", "-C", repository, "reset", "--hard", "FETCH_HEAD"])
else:
    run(["git", "clone", "--depth", "1", REPOSITORY_URL, repository])
run([sys.executable, "notebooks/kaggle_v1_6.py", "--use-existing-repo"], cwd=repository)
