import csv
from huggingface_hub import hf_hub_download

repo = "H1merka/TIGAS_dataset"
filename = "train/annotations01.csv"

print("Downloading annotation CSV only...")
path = hf_hub_download(
    repo_id=repo,
    filename=filename,
    repo_type="dataset"
)

print("\nCSV location:", path)

with open(path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        print(row)
        if i >= 5:
            break
