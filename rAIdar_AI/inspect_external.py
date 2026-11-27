from PIL import Image
import os
import numpy as np

EXTERNAL_PATH = r"C:\Users\Adrian\source\repos\rAIdar_AI\external_test"

for category in ["fake", "real"]:

    folder = os.path.join(EXTERNAL_PATH, category)

    print("\n" + "=" * 80)
    print(category.upper())
    print("=" * 80)

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])

    widths = []
    heights = []
    means = []
    stds = []

    for filename in files:

        path = os.path.join(folder, filename)

        try:
            image = Image.open(path).convert("RGB")
            array = np.array(image)

            width, height = image.size

            widths.append(width)
            heights.append(height)
            means.append(array.mean())
            stds.append(array.std())

            print(
                f"{filename:35} "
                f"{width:5}x{height:<5} "
                f"mean={array.mean():7.2f} "
                f"std={array.std():7.2f}"
            )

        except Exception as e:
            print(filename, "ERROR:", e)

    print("\nSUMMARY")
    print("Images:", len(files))
    print("Average width:", np.mean(widths))
    print("Average height:", np.mean(heights))
    print("Average pixel mean:", np.mean(means))
    print("Average pixel std:", np.mean(stds))