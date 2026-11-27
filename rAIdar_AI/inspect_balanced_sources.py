import csv
from collections import defaultdict
from huggingface_hub import hf_hub_download

REPO = "H1merka/TIGAS_dataset"

for split in ["train", "val", "test"]:
    csv_path = hf_hub_download(
        repo_id=REPO,
        filename=f"{split}/annotations01.csv",
        repo_type="dataset"
    )

    sources = defaultdict(lambda: {"real": 0, "fake": 0})

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            parts = row["image_path"].replace("\\", "/").split("/")
            source = parts[1] if len(parts) > 2 else "unknown"
            label = row["label"].strip()

            if label == "0":
                sources[source]["real"] += 1
            elif label == "1":
                sources[source]["fake"] += 1

    print(f"\n===== {split.upper()}: AVAILABLE BALANCED IMAGES =====")
    print(f"{'Source':<20} {'Real':>8} {'Fake':>8}")
    print("-" * 38)

    for source, counts in sorted(sources.items()):
        available = min(counts["real"], counts["fake"])
        print(
            f"{source:<20} {counts['real']:>8} "
            f"{counts['fake']:>8}  "
            f"balanced pairs available: {available}"
        )
