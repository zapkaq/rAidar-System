import csv
from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="H1merka/TIGAS_dataset",
    filename="train/annotations01.csv",
    repo_type="dataset"
)

counts = Counter()
examples = defaultdict(list)

with open(path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    print("Columns:", reader.fieldnames)

    for row in reader:
        label = row["label"].strip()
        image_path = row["image_path"].strip()
        counts[label] += 1

        if len(examples[label]) < 3:
            examples[label].append(image_path)

print("\n=== IMAGE COUNTS BY LABEL ===")
for label, count in sorted(counts.items()):
    print(f"Label {label}: {count:,} images")

print("\n=== SAMPLE PATHS BY LABEL ===")
for label in sorted(examples):
    print(f"\nLabel {label}:")
    for image_path in examples[label]:
        print(" ", image_path)
