import argparse
import csv
import os

import cv2
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
GRID_ROWS = 4
GRID_COLS = 3
TARGET_ROW_INDEX = 2  # 3rd row in a 4-row grid (0-based index)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manual patch labeling for only the 3rd row of a 4x3 grid."
    )
    parser.add_argument(
        "--folder",
        default="./datasets/Organised/20260306-handFlying-2",
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output",
        default="manual_row3_labels.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Disable default 90 degrees counterclockwise rotation before labeling.",
    )
    return parser.parse_args()


def get_image_files(folder):
    return sorted(
        f
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    )


def load_existing_labels(csv_path):
    labels = {}
    if not os.path.exists(csv_path):
        return labels

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_path = row.get("frame_path", "")
            row_idx = int(row.get("row_index", -1))
            col_idx = int(row.get("col_index", -1))
            label = int(row.get("tree_present", -1))

            if row_idx != TARGET_ROW_INDEX or col_idx not in (0, 1, 2):
                continue

            if frame_path not in labels:
                labels[frame_path] = [-1, -1, -1]
            labels[frame_path][col_idx] = label

    return labels


def save_labels_csv(output_path, image_paths, labels_by_path):
    rows = []
    for frame_path in image_paths:
        frame_labels = labels_by_path.get(frame_path, [-1, -1, -1])
        for col_idx in range(3):
            cell_index = TARGET_ROW_INDEX * GRID_COLS + col_idx
            rows.append(
                [
                    frame_path,
                    cell_index,
                    TARGET_ROW_INDEX,
                    col_idx,
                    frame_labels[col_idx],
                ]
            )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_path",
            "cell_index",
            "row_index",
            "col_index",
            "tree_present",
        ])
        writer.writerows(rows)


def draw_grid(img):
    h, w = img.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS

    for r in range(1, GRID_ROWS):
        y = r * cell_h
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)

    for c in range(1, GRID_COLS):
        x = c * cell_w
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)


def overlay_target_row_patches(img, labels):
    h, w = img.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS

    y1 = TARGET_ROW_INDEX * cell_h
    y2 = (TARGET_ROW_INDEX + 1) * cell_h

    overlay = img.copy()

    for col_idx in range(3):
        x1 = col_idx * cell_w
        x2 = (col_idx + 1) * cell_w

        label = labels[col_idx]
        if label == 1:
            color = (0, 180, 0)
            text = f"P{col_idx + 1}: TREE"
        elif label == 0:
            color = (0, 0, 180)
            text = f"P{col_idx + 1}: NO"
        else:
            color = (0, 180, 180)
            text = f"P{col_idx + 1}: ?"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            img,
            text,
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.22, img, 0.78, 0.0, dst=img)


def draw_help_text(img, frame_idx, frame_total, image_name, labels):
    all_labeled = all(v in (0, 1) for v in labels)
    status = "READY (all 3 labeled)" if all_labeled else "INCOMPLETE"

    lines = [
        f"Frame {frame_idx + 1}/{frame_total}: {image_name}",
        "Label 3rd row patches only (left->right):",
        "1/2/3 = TREE for patch 1/2/3",
        "4/5/6 = NO-TREE for patch 1/2/3",
        "r = reset frame labels | n = next | b = previous",
        "s = save CSV now | ESC = save and quit",
        f"Status: {status}",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            img,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28


def prepare_display_image(image, rotate_ccw_90):
    out = image.copy()
    if rotate_ccw_90:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return out


def main():
    args = parse_args()

    if not os.path.isdir(args.folder):
        print(f"Folder not found: {args.folder}")
        return

    image_files = get_image_files(args.folder)
    if not image_files:
        print("No images found in folder.")
        return

    first = cv2.imread(os.path.join(args.folder, image_files[0]))
    if first is None:
        print("Failed to read first image.")
        return

    base_h, base_w = first.shape[:2]

    image_paths = [os.path.join(args.folder, f) for f in image_files]
    labels_by_path = load_existing_labels(args.output)

    for p in image_paths:
        if p not in labels_by_path:
            labels_by_path[p] = [-1, -1, -1]

    current_idx = 0
    window_name = "Manual Labeling - 3rd Row of 4x3 Grid"

    def move_next_if_complete(frame_path, idx):
        labels = labels_by_path[frame_path]
        if not all(v in (0, 1) for v in labels):
            return idx, False, False

        if idx < len(image_paths) - 1:
            return idx + 1, True, False

        save_labels_csv(args.output, image_paths, labels_by_path)
        print(f"Saved labels to: {args.output}")
        print("Reached last image.")
        return idx, True, True

    while True:
        image_path = image_paths[current_idx]
        image_name = os.path.basename(image_path)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping unreadable image: {image_path}")
            current_idx = min(current_idx + 1, len(image_paths) - 1)
            if current_idx == len(image_paths) - 1:
                break
            continue

        img = cv2.resize(img, (base_w, base_h))
        display = prepare_display_image(img, not args.no_rotate)

        labels = labels_by_path[image_path]

        draw_grid(display)
        overlay_target_row_patches(display, labels)
        draw_help_text(display, current_idx, len(image_paths), image_name, labels)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            save_labels_csv(args.output, image_paths, labels_by_path)
            print(f"Saved labels to: {args.output}")
            break

        auto_advanced = False
        finished_last = False
        if key == ord("1"):
            labels[0] = 1
            current_idx, auto_advanced, finished_last = move_next_if_complete(image_path, current_idx)
        elif key == ord("2"):
            labels[1] = 1
            current_idx, auto_advanced, finished_last = move_next_if_complete(image_path, current_idx)
        elif key == ord("3"):
            labels[2] = 1
            current_idx, auto_advanced, finished_last = move_next_if_complete(image_path, current_idx)
        elif key == ord("4"):
            labels[0] = 0
            current_idx, auto_advanced, finished_last = move_next_if_complete(image_path, current_idx)
        elif key == ord("5"):
            labels[1] = 0
            current_idx, auto_advanced, finished_last = move_next_if_complete(image_path, current_idx)
        elif key == ord("6"):
            labels[2] = 0
            current_idx, auto_advanced, finished_last = move_next_if_complete(image_path, current_idx)
        elif key == ord("r"):
            labels_by_path[image_path] = [-1, -1, -1]
        elif key == ord("s"):
            save_labels_csv(args.output, image_paths, labels_by_path)
            print(f"Saved labels to: {args.output}")
        elif key == ord("b"):
            if current_idx > 0:
                current_idx -= 1
        elif key == ord("n"):
            if all(v in (0, 1) for v in labels):
                if current_idx < len(image_paths) - 1:
                    current_idx += 1
                else:
                    save_labels_csv(args.output, image_paths, labels_by_path)
                    print(f"Saved labels to: {args.output}")
                    print("Reached last image.")
                    break
            else:
                print("Label all 3 target patches before moving to next frame.")

        if finished_last:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
