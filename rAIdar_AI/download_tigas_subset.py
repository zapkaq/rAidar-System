import csv
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "H1merka/TIGAS_dataset"

# Target count per class in each split.
TARGETS = {
    "train": {"0": 1600, "1": 1600},
    "val":   {"0": 200,  "1": 200},
    "test":  {"0": 200,  "1": 200},
}

ROOT = Path("datasets/diverse")
LABEL_NAMES = {"0": "real", "1": "fake"}

def get_rows(csv_path, label, limit):
    selected = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["label"].strip() == label:
                selected.append(row["image_path"].strip())
                if len(selected) == limit:
                    break
    return selected

def main():
    total_saved = 0
    total_skipped = 0
    total_failed = 0

    for split, label_targets in TARGETS.items():
        print(f"\n===== {split.upper()} SPLIT =====")
        csv_path = hf_hub_download(
            repo_id=REPO,
            filename=f"{split}/annotations01.csv",
            repo_type="dataset"
        )

        for label, target_count in label_targets.items():
            category = LABEL_NAMES[label]
            destination = ROOT / ("valid" if split == "val" else split) / category
            destination.mkdir(parents=True, exist_ok=True)

            paths = get_rows(csv_path, label, target_count)
            if len(paths) < target_count:
                print(f"ERROR: Only found {len(paths)} rows for {split}/{category}.")
                return 1

            saved = skipped = failed = 0

            for index, image_path in enumerate(paths, start=1):
                normalized = image_path.replace("\\", "/")
                remote_path = f"{split}/{normalized}"
                suffix = Path(normalized).suffix or ".img"
                target = destination / f"{index:04d}_{category}{suffix}"

                if target.exists() and target.stat().st_size > 0:
                    skipped += 1
                    continue

                try:
                    cached_file = hf_hub_download(
                        repo_id=REPO,
                        filename=remote_path,
                        repo_type="dataset"
                    )
                    shutil.copy2(cached_file, target)
                    saved += 1
                except Exception as error:
                    failed += 1
                    print(f"\nFAILED: {remote_path}")
                    print(f"Reason: {error}")
                    if failed >= 5:
                        print("Stopping this class after 5 failures.")
                        break

                completed = saved + skipped + failed
                if completed % 50 == 0 or completed == target_count:
                    print(
                        f"{split}/{category}: {completed}/{target_count} "
                        f"(new={saved}, existing={skipped}, failed={failed})"
                    )

            total_saved += saved
            total_skipped += skipped
            total_failed += failed
            print(
                f"Finished {split}/{category}: "
                f"new={saved}, existing={skipped}, failed={failed}"
            )

    print("\n===== DOWNLOAD SUMMARY =====")
    print("New files copied:", total_saved)
    print("Existing files skipped:", total_skipped)
    print("Download failures:", total_failed)
    print("Destination:", ROOT.resolve())

    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
