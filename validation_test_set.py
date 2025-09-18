import os, shutil, glob
from sklearn.model_selection import train_test_split

base_path = "processed_data/train"
output_val = "processed_data/validation"

# Collect real and fake images
real_images = glob.glob(os.path.join(base_path, "real", "*.jpg"))
fake_images = glob.glob(os.path.join(base_path, "fake", "*.jpg"))

# Split (20% validation)
train_real, val_real = train_test_split(real_images, test_size=0.2, random_state=42)
train_fake, val_fake = train_test_split(fake_images, test_size=0.2, random_state=42)

def move_files(file_list, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for f in file_list:
        shutil.move(f, dest_folder)

# Move validation images
move_files(val_real, os.path.join(output_val, "real"))
move_files(val_fake, os.path.join(output_val, "fake"))
