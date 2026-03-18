import cv2
import numpy as np
import csv
import random
import os

# ----------------------------
# CONFIG
# ----------------------------
csv_path = "balanced_tree_patch_labels.csv"
rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE

GRID_ROWS = 4
GRID_COLS = 3

NUM_SAMPLES = 30   # how many random patches to show

# ----------------------------
# HELPERS
# ----------------------------
def split_into_grid(image, rows, cols):
    h, w = image.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    grid = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            row_cells.append(image[y1:y2, x1:x2])
        grid.append(row_cells)
    return grid

# ----------------------------
# MAIN
# ----------------------------
def main():
    # Load CSV entries
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)

    if not samples:
        print("CSV is empty.")
        return

    # Randomly sample
    chosen = random.sample(samples, min(NUM_SAMPLES, len(samples)))

    for entry in chosen:
        frame_path = entry["frame_path"]
        cell_idx = int(entry["cell_index"])
        label = int(entry["tree_present"])

        if "row_index" in entry and "col_index" in entry and entry["row_index"] and entry["col_index"]:
            row_idx = int(entry["row_index"])
            col_idx = int(entry["col_index"])
        else:
            row_idx = cell_idx // GRID_COLS
            col_idx = cell_idx % GRID_COLS

        img = cv2.imread(frame_path)
        if img is None:
            print(f"Failed to load {frame_path}")
            continue

        # Rotate to match training orientation
        img = cv2.rotate(img, rotation_code)

        # Split into grid
        grid = split_into_grid(img, GRID_ROWS, GRID_COLS)

        # Extract patch from full grid
        patch = grid[row_idx][col_idx]

        # Display
        display = patch.copy()
        text = (
            f"Label: {'TREE' if label == 1 else 'NO TREE'} | "
            f"Cell {cell_idx} (r{row_idx}, c{col_idx})"
        )
        cv2.putText(display, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cv2.imshow("Random Patch Viewer", display)
        print(f"Showing: {frame_path} | Cell {cell_idx} | Label {label}")

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
