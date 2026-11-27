
from pathlib import Path
from collections import defaultdict, Counter
import csv

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "datasets" / "diverse_balanced" / "manifest.csv"

SPLITS = {"train", "valid", "test"}

source_counts = defaultdict(Counter)
source_splits = defaultdict(set)
original_path_splits = defaultdict(set)
original_path_records = defaultdict(list)

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    required = {
        "local_split",
        "label",
        "source",
        "original_path",
        "local_path"
    }

    missing = required - set(reader.fieldnames or [])

    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")

    for row in reader:
        split = row["local_split"].strip().lower()
        label = row["label"].strip().lower()
        source = row["source"].strip()
        original = row["original_path"].strip().replace("\\", "/")

        if split not in SPLITS:
            print("Unexpected split:", split, row["local_path"])
            continue

        source_counts[source][(split, label)] += 1
        source_splits[source].add(split)

        original_path_splits[original].add(split)
        original_path_records[original].append(row)

print("\n========== SOURCE DISTRIBUTION ==========")

with (ROOT / "source_overlap_summary.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "source",
        "train_fake", "train_real",
        "valid_fake", "valid_real",
        "test_fake", "test_real",
        "number_of_splits"
    ])

    for source in sorted(source_counts):
        counts = source_counts[source]

        values = [
            counts[("train", "fake")],
            counts[("train", "real")],
            counts[("valid", "fake")],
            counts[("valid", "real")],
            counts[("test", "fake")],
            counts[("test", "real")]
        ]

        writer.writerow([
            source,
            *values,
            len(source_splits[source])
        ])

        print(
            f"{source}: "
            f"train={values[0] + values[1]}, "
            f"valid={values[2] + values[3]}, "
            f"test={values[4] + values[5]}, "
            f"splits={len(source_splits[source])}"
        )

print("\n========== ORIGINAL PATH OVERLAP ==========")

cross_split_paths = {
    path: records
    for path, records in original_path_records.items()
    if len(original_path_splits[path]) > 1
}

with (ROOT / "original_path_cross_split.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "original_path",
        "source",
        "label",
        "local_split",
        "local_path"
    ])

    for original, records in sorted(cross_split_paths.items()):
        for row in records:
            writer.writerow([
                original,
                row["source"],
                row["label"],
                row["local_split"],
                row["local_path"]
            ])

print("Total manifest records:", sum(
    sum(counts.values()) for counts in source_counts.values()
))

print("Number of sources:", len(source_counts))
print("Sources appearing in all splits:", sum(
    1 for splits in source_splits.values()
    if splits == SPLITS
))
print("Original paths repeated across splits:", len(cross_split_paths))

print("\nReports saved:")
print(ROOT / "source_overlap_summary.csv")
print(ROOT / "original_path_cross_split.csv")
