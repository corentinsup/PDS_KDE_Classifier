import os
import shutil
from pathlib import Path

def split_by_labels(data_dir: str, labels_file: str, dest_dir: str):

    data_dir = Path(data_dir)
    labels_file = Path(labels_file)
    dest_dir = Path(dest_dir)

    # Define mapping between label symbols and folder names
    label_map = {
        'S': 'Single',
        'M': 'Multi',
        'Gbg': 'Garbage'
    }
# --- Create target folders if needed ---
    for folder in label_map.values():
        (dest_dir / folder).mkdir(exist_ok=True)
# --- Read labels file ---
    file_to_label = {}
    current_label = None

    with open(labels_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # if line is one of the labels
            if stripped in label_map:
                current_label = stripped
            # otherwise it's a filename
            elif current_label and stripped.endswith('.pcd'):
                file_to_label[stripped] = current_label

    print(f"Found {len(file_to_label)} labeled files.")

    # --- Move only labeled files ---
    moved_count = 0
    for filename, label in file_to_label.items():
        src_path = data_dir / filename
        if not src_path.exists():
            # if your files are nested (e.g., in subfolders), we can search recursively:
            found = list(data_dir.rglob(filename))
            if found:
                src_path = found[0]
            else:
                print(f"Not found: {filename}")
                continue

        destination_dir = dest_dir / label_map[label]
        destination_path = destination_dir / src_path.name

        try:
            shutil.copy2(src_path, destination_path)  # atomic + fast
            moved_count += 1
        except Exception as e:
            print(f"Failed to move {filename}: {e}")

    print(f"Done! Moved {moved_count} files into labeled folders.")


if __name__ == "__main__":

    data_dir = input("Enter path to data directory: ").strip()
    labels_file = input("Enter path to .labels file: ").strip()
    dest_dir = input("Enter path to destination directory (will create subfolders here): ").strip()
    split_by_labels(data_dir, labels_file, dest_dir)