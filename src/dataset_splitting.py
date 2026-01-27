import os
import shutil
from pathlib import Path

def split_by_labels(data_dir: str, labels_file: str, dest_dir: str, inference_dir: str):

    data_dir = Path(data_dir)
    labels_file = Path(labels_file)
    dest_dir = Path(dest_dir)
    inference_dir = Path(inference_dir)

    # --- Label mapping ---
    label_map = {
        'S': 'Single',
        'M': 'Multi',
        'Gbg': 'Garbage'
    }

    # Folder for unlabeled files
    unlabeled_folder_name = "test"

    # --- Create target folders ---
    for folder in label_map.values():
        (dest_dir / folder).mkdir(parents=True, exist_ok=True)

    (inference_dir / unlabeled_folder_name).mkdir(parents=True, exist_ok=True)

    # --- Parse labels file ---
    file_to_label = {}
    current_label = None

    with open(labels_file, 'r') as f:
        for line in f:
            stripped = line.strip()

            if not stripped:
                continue

            # If it's one of the label blocks (S, M, Gbg)
            if stripped in label_map:
                current_label = stripped

            # Accept only .pcd filenames, ignore dotfiles like ".skipped_clouds"
            elif current_label and stripped.endswith('.pcd') and not stripped.startswith('.'):
                file_to_label[stripped] = current_label

    print(f"Found {len(file_to_label)} labeled files.")

    # --- Move labeled files ---
    moved_count = 0
    labeled_filenames = set(file_to_label.keys())

    for filename, label in file_to_label.items():
        src_path = data_dir / filename

        # If file is not found, search recursively in subfolders
        if not src_path.exists():
            found = list(data_dir.rglob(filename))
            if found:
                src_path = found[0]
            else:
                print(f"Not found: {filename}")
                continue

        destination_dir = dest_dir / label_map[label]
        destination_path = destination_dir / src_path.name

        # Skip if already exists
        if destination_path.exists():
            continue

        try:
            shutil.copy2(src_path, destination_path)
            moved_count += 1
        except Exception as e:
            print(f"Failed to copy {filename}: {e}")

    print(f"Copied {moved_count} labeled files.")

    # --- Copy unlabeled files ---
    unlabeled_count = 0

    for pcd_file in data_dir.rglob("*.pcd"):

        # Skip labeled files
        if pcd_file.name in labeled_filenames:
            continue

        # Correct: use inference_dir, not dest_dir
        dest_path = inference_dir / unlabeled_folder_name / pcd_file.name

        # Create directory if missing
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists
        if dest_path.exists():
            continue

        try:
            shutil.copy2(pcd_file, dest_path)
            unlabeled_count += 1
        except Exception as e:
            print(f"Failed to copy unlabeled file {pcd_file}: {e}")

    print(f"Copied {unlabeled_count} unlabeled files to '{unlabeled_folder_name}'.")
    print("Done.")


if __name__ == "__main__":

    data_dir = input("Enter path to data directory: ").strip()
    labels_file = input("Enter path to .labels file: ").strip()
    dest_dir = input("Enter path to destination directory: ").strip()
    inference_dir = input("Enter path to inference directory: ").strip()

    split_by_labels(data_dir, labels_file, dest_dir, inference_dir)
