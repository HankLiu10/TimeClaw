"""Install the exact external MTBench code dependency, separately from datasets."""
from pathlib import Path
import argparse
import subprocess
from tempfile import TemporaryDirectory

URL = "https://github.com/Graph-and-Geometric-Learning/MTBench.git"
REVISION = "394a9d274403e93a08b24c2a0b7af461615ce302"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("mtbench_official"))
    args = parser.parse_args()
    target = args.destination.resolve()
    if target.exists():
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != REVISION:
            raise RuntimeError(f"Existing MTBench has revision {actual}; expected {REVISION}. No files changed.")
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        if dirty.strip():
            raise RuntimeError("Existing MTBench has local changes. No files changed.")
    else:
        # A failed fetch must not publish a partial installation. Keep all
        # new files in an owned sibling staging directory until checkout ends.
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{target.name}.setup-", dir=target.parent) as stage:
            checkout = Path(stage) / "checkout"
            subprocess.run(["git", "init", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth=1", URL, REVISION], check=True)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", REVISION], check=True)
            checkout.rename(target)
    print(f"MTBench {REVISION}: {target}")
    print(f"For a custom destination, export TIMECLAW_MTBENCH_ROOT={target}")

if __name__ == "__main__":
    main()
