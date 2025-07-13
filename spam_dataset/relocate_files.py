import os
import shutil

autosplit_train = "autosplit_train.txt"
autosplit_val = "autosplit_val.txt"

# source_images_dir = "coco128/images"
# source_labels_dir = "coco128/labels"

dest_images_dir_train = "../dataset/train/images"
dest_labels_dir_train = "../dataset/train/labels"

dest_images_dir_val = "../dataset/val/images"
dest_labels_dir_val = "../dataset/val/labels"

def relocate_files(dest_dir, list_file, is_img=True):
    os.makedirs(dest_dir, exist_ok=True)

    with open(list_file, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]
        if not is_img:
            converted = []
            for name in filenames:
                replaced_folder = name.replace('images', 'labels')
                base, ext = os.path.splitext(replaced_folder)
                new_path = base + '.txt'
                converted.append(new_path)
            filenames = converted
    for filename in filenames:
        dest_path = os.path.join(dest_dir, filename.split("/")[-1])

        if os.path.isfile(filename):
            shutil.copy(filename, dest_path)
            print(f"Copied: {filename}")
        else:
            print(f"File not found: {filename}")

relocate_files(dest_images_dir_train, autosplit_train)
relocate_files(dest_labels_dir_train, autosplit_train, is_img=False)
relocate_files(dest_images_dir_val, autosplit_val)
relocate_files(dest_labels_dir_val, autosplit_val, is_img=False)