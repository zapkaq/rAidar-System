import csv
from collections import Counter
from huggingface_hub import hf_hub_download

repo = "H1merka/TIGAS_dataset"

for split in ["train", "val", "test"]:
    path = hf_hub_download(
        repo_id=repo,
        filename=f"{split}/annotations01.csv",
        repo_type="dataset"
    )

    counts = Counter()
    samples = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip()
            counts[label] += 1
            samples.setdefault(label, row["image_path"])

    print(f"\n=== {split.upper()} ===")
    print("Real (0):", f"{counts['0']:,}")
    print("Fake (1):", f"{counts['1']:,}")
    print("Real sample:", samples.get("0", "NOT FOUND"))
    print("Fake sample:", samples.get("1", "NOT FOUND"))
