import os
import shutil
import pandas as pd

# Paths
base_dir = "audience_data/AdienceBenchmarkGenderAndAgeClassification"
faces_dir = os.path.join(base_dir, "faces")
output_dir = "audience_data/organized"

# Read all fold files
dfs = []
for i in range(5):
    fold_file = os.path.join(base_dir, f"fold_{i}_data.txt")
    df = pd.read_csv(fold_file, sep="\t")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Total samples: {len(data)}")
print(f"Columns: {data.columns.tolist()}")
print(data.head())

# Create output folders for gender classification
for gender in ["m", "f"]:
    os.makedirs(os.path.join(output_dir, gender), exist_ok=True)

copied = 0
skipped = 0

for _, row in data.iterrows():
    user_id = str(row["user_id"])
    original_image = str(row["original_image"])
    gender = str(row["gender"]).strip().lower()

    if gender not in ["m", "f"]:
        skipped += 1
        continue

    # Image is inside faces/user_id/coarse_tilt_aligned_face.age.original_image
    img_folder = os.path.join(faces_dir, user_id)
    if not os.path.exists(img_folder):
        skipped += 1
        continue

    # Find matching image file
    for fname in os.listdir(img_folder):
        if original_image in fname:
            src = os.path.join(img_folder, fname)
            dst = os.path.join(output_dir, gender, f"{user_id}_{fname}")
            shutil.copy2(src, dst)
            copied += 1
            break
    else:
        skipped += 1

print(f"Copied: {copied} | Skipped: {skipped}")
print(f"Done! Organized dataset at: {output_dir}")
