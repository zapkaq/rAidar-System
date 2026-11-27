import csv
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = "H1merka/TIGAS_dataset"
output = Path("datasets/pilot/train")
csv_path = hf_hub_download(
    repo_id=repo,
    filename="train/annotations01.csv",
    repo_type="dataset"
)

selected = {"0": [], "1": []}

with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        label = row["label"].strip()
        if label in selected and len(selected[label]) < 5:
            selected[label].append(row["image_path"])
        if all(len(items) == 5 for items in selected.values()):
            break

for label, paths in selected.items():
    category = "real" if label == "0" else "fake"
    destination = output / category
    destination.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(paths, start=1):
        normalized_path = image_path.replace("\\", "/")
        remote_path = f"train/{normalized_path}"

        print(f"Downloading {category} image {index}/5: {normalized_path}")
        cached_file = hf_hub_download(
            repo_id=repo,
            filename=remote_path,
            repo_type="dataset"
        )

        suffix = Path(normalized_path).suffix
        target = destination / f"{index:02d}_{category}{suffix}"
        shutil.copy2(cached_file, target)
        print(f"Saved: {target}")

print("\nPilot download complete.")
print("Real images:", len(list((output / "real").glob("*"))))
print("Fake images:", len(list((output / "fake").glob("*"))))
