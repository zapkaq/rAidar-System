import csv
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "H1merka/TIGAS_dataset"
SEED = 42
ROOT = Path.cwd()
DEST = ROOT / "datasets" / "diverse_balanced"

SPLITS = {
    "train": {"source_split": "train", "per_class": 1600},
    "valid": {"source_split": "val", "per_class": 200},
    "test": {"source_split": "test", "per_class": 200},
}

# Do not overwrite an existing destination.
if DEST.exists() and any(DEST.iterdir()):
    raise SystemExit(
        f"STOP: {DEST} already contains files. "
        "Choose a new destination before proceeding."
    )

random.seed(SEED)
manifest_path = DEST / "manifest.csv"
manifest_rows = []
failures = []
downloaded = 0

# Read and balance each official split independently.
selections = {}

for local_split, config in SPLITS.items():
    source_split = config["source_split"]
    target = config["per_class"]

    csv_path = hf_hub_download(
        repo_id=REPO,
        filename=f"{source_split}/annotations01.csv",
        repo_type="dataset",
    )

    grouped = defaultdict(lambda: {"real": [], "fake": []})

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            image_path = row["image_path"].replace("\\", "/")
            parts = image_path.split("/")
            source = parts[1] if len(parts) > 2 else "unknown"
            label = row["label"].strip()

            if label == "0":
                grouped[source]["real"].append(image_path)
            elif label == "1":
                grouped[source]["fake"].append(image_path)

    sources = sorted(
        s for s, values in grouped.items()
        if values["real"] and values["fake"]
    )

    # Allocate an equal target across sources, then redistribute
    # unused slots from small sources to sources with spare capacity.
    allocation = {s: target // len(sources) for s in sources}
    remainder = target % len(sources)

    for s in sources[:remainder]:
        allocation[s] += 1

    for label in ("real", "fake"):
        capacities = {
            s: min(len(grouped[s][label]), target)
            for s in sources
        }
        counts = {s: min(allocation[s], capacities[s]) for s in sources}
        left = target - sum(counts.values())

        while left > 0:
            candidates = [
                s for s in sources if counts[s] < capacities[s]
            ]
            if not candidates:
                raise RuntimeError(
                    f"Not enough {label} images for {local_split}"
                )
            for s in candidates:
                if left == 0:
                    break
                counts[s] += 1
                left -= 1

        for s in sources:
            random.shuffle(grouped[s][label])
            chosen = grouped[s][label][:counts[s]]
            selections.setdefault(local_split, {}).setdefault(label, [])
            selections[local_split][label].extend(
                (s, image_path) for image_path in chosen
            )

# Download to a separate destination, retaining source/split metadata.
for local_split, labels in selections.items():
    for label, entries in labels.items():
        folder = DEST / local_split / label
        folder.mkdir(parents=True, exist_ok=True)

        for index, (source, image_path) in enumerate(entries, start=1):
            remote_path = f"{SPLITS[local_split]['source_split']}/{image_path}"
            filename = f"{index:05d}{Path(image_path).suffix.lower()}"
            target_path = folder / filename

            try:
                cached_path = hf_hub_download(
                    repo_id=REPO,
                    filename=remote_path,
                    repo_type="dataset",
                )
                shutil.copy2(cached_path, target_path)
                manifest_rows.append({
                    "local_split": local_split,
                    "label": label,
                    "source": source,
                    "original_path": image_path,
                    "local_path": str(target_path.relative_to(ROOT)),
                })
                downloaded += 1

                if downloaded % 100 == 0:
                    print(f"Downloaded {downloaded}/4000")
            except Exception as exc:
                failures.append((remote_path, str(exc)))
                print(f"FAILED: {remote_path}: {exc}")

# Write a manifest so each renamed file can be traced to its source.
DEST.mkdir(parents=True, exist_ok=True)
with open(manifest_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "local_split", "label", "source",
            "original_path", "local_path"
        ],
    )
    writer.writeheader()
    writer.writerows(manifest_rows)

print("\n===== DOWNLOAD SUMMARY =====")
print(f"Destination: {DEST}")
print(f"Successfully copied: {downloaded}")
print(f"Failures: {len(failures)}")
print(f"Manifest: {manifest_path}")

for split in SPLITS:
    for label in ("real", "fake"):
        count = sum(
            1 for row in manifest_rows
            if row["local_split"] == split and row["label"] == label
        )
        print(f"{split}/{label}: {count}")

if failures:
    print("\nSome images failed to download. Do not train yet.")
    print("Review the failure messages and rerun after fixing the cause.")
