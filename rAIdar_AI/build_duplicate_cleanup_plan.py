
from pathlib import Path
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "duplicate_report.csv"
OUTPUT = ROOT / "duplicate_cleanup_plan.csv"

groups = defaultdict(list)

with REPORT.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        groups[row["duplicate_group"]].append(row)

# Preserve evaluation images before training images.
priority = {
    "test": 0,
    "valid": 1,
    "train": 2
}

plan = []

for group_id, files in sorted(groups.items()):
    ordered = sorted(
        files,
        key=lambda x: (
            priority[x["split"]],
            x["path"]
        )
    )

    keep = ordered[0]

    for item in ordered:
        if item["path"] == keep["path"]:
            action = "KEEP"
            reason = "Preserve this copy"
        else:
            action = "REMOVE_CANDIDATE"
            reason = (
                "Duplicate of retained image; "
                "exclude from future cleaned dataset"
            )

        plan.append({
            "duplicate_group": group_id,
            "action": action,
            "split": item["split"],
            "class": item["class"],
            "path": item["path"],
            "retained_path": keep["path"],
            "reason": reason
        })

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "duplicate_group",
            "action",
            "split",
            "class",
            "path",
            "retained_path",
            "reason"
        ]
    )

    writer.writeheader()
    writer.writerows(plan)

print("========== CLEANUP PLAN ==========")
print("Duplicate groups:", len(groups))
print("Images reviewed:", len(plan))
print("Keep:", sum(x["action"] == "KEEP" for x in plan))
print("Remove candidates:", sum(
    x["action"] == "REMOVE_CANDIDATE" for x in plan
))
print("Report saved:", OUTPUT)
print("No images were deleted or modified.")
