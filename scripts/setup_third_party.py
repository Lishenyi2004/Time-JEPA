"""Fetch only source dependencies, at recorded revisions; no data or weights."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="VideoMAEv2, iVideoGPT, PSG-JEPA; omitted means all three")
    args = parser.parse_args()
    items = json.loads((ROOT / "third-party.json").read_text())
    sources = {item["name"]: item for item in items if item["source_required"]}
    unknown = set(args.names) - sources.keys()
    if unknown:
        parser.error(f"Unknown source dependency: {sorted(unknown)}")
    for name in args.names or list(sources):
        item = sources[name]
        target = ROOT / item["directory"]
        if target.exists():
            revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
            if revision != item["revision"]:
                raise RuntimeError(f"Existing {target} has revision {revision}; expected {item['revision']}")
            print(f"Already present: {name} @ {revision}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", item["url"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", item["revision"]], check=True)
        print(f"Prepared: {name} @ {item['revision']}")

if __name__ == "__main__":
    main()
