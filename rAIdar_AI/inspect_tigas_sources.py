import csv
from collections import defaultdict
from huggingface_hub import hf_hub_download

repo = "H1merka/TIGAS_dataset"

for split in ["train", "val", "test"]:
    csv_path = hf_hub_download(
        repo_id=repo,
        filename=f"{split}/annotations01.csv",
        repo_type="dataset"
    )

    counts = defaultdict(lambda: {"real": 0, "fake": 0})

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            parts = row["image_path"].replace("\\", "/").split("/")
            source = parts[1] if len(parts) > 2 else "unknown"
            label = row["label"].strip()

            if label == "0":
                counts[source]["real"] += 1
            elif label == "1":
                counts[source]["fake"] += 1

    print(f"\n===== {split.upper()} SOURCE DISTRIBUTION =====")
    print(f"{'Source':<20} {'Real':>10} {'Fake':>10}")
    print("-" * 42)

    for source, values in sorted(
        counts.items(),
        key=lambda item: item[1]["real"] + item[1]["fake"],
        reverse=True
    ):
        print(f"{source:<20} {values['real']:>10,} {values['fake']:>10,}")
