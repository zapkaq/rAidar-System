
from pathlib import Path
import csv
import shutil

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "datasets" / "diverse_balanced"
DESTINATION = ROOT / "datasets" / "diverse_balanced_cleaned"

MANIFEST = SOURCE / "manifest.csv"
CLEAN_MANIFEST = DESTINATION / "manifest.csv"
PLAN = ROOT / "duplicate_cleanup_plan.csv"

if DESTINATION.exists():
    raise FileExistsError(
        f"{DESTINATION} already exists. "
        "Do not overwrite it. Check the folder first."
    )

# Read paths marked for exclusion.
remove_paths = set()

with PLAN.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["action"] == "REMOVE_CANDIDATE":
            remove_paths.add(
                row["path"].replace("\\", "/")
            )

print("Files marked for exclusion:", len(remove_paths))

copied = 0
skipped = 0

try:
    with MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as source_file:

        reader = csv.DictReader(source_file)
        fieldnames = reader.fieldnames

        if not fieldnames or "local_path" not in fieldnames:
            raise ValueError("Manifest has no local_path column.")

        DESTINATION.mkdir(parents=True)

        with CLEAN_MANIFEST.open(
            "w", encoding="utf-8-sig", newline=""
        ) as output_file:

            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames
            )
            writer.writeheader()

            for row in reader:
                relative_path = row["local_path"].replace("\\", "/")

                if relative_path in remove_paths:
                    skipped += 1
                    continue

                old_path = ROOT / Path(relative_path)

                if not old_path.is_file():
                    raise FileNotFoundError(old_path)

                relative_to_dataset = Path(relative_path).relative_to(
                    Path("datasets/diverse_balanced")
                )

                new_path = DESTINATION / relative_to_dataset
                new_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(old_path, new_path)

                row["local_path"] = (
                    Path("datasets/diverse_balanced_cleaned")
                    / relative_to_dataset
                ).as_posix()

                writer.writerow(row)
                copied += 1

    print("\n========== CLEANING RESULTS ==========")
    print("Images copied:", copied)
    print("Images excluded:", skipped)
    print("Total original records:", copied + skipped)
    print("Cleaned dataset:", DESTINATION)
    print("Updated manifest:", CLEAN_MANIFEST)
    print("Original dataset was not modified.")

except Exception:
    shutil.rmtree(DESTINATION, ignore_errors=True)
    raise
