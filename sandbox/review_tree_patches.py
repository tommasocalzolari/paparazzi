import argparse
import csv
import os

import cv2
import numpy as np


GRID_ROWS = 4
GRID_COLS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Review tree-labeled patches and correct labels with keyboard input."
    )
    parser.add_argument(
        "--csv",
        default="balanced_tree_patch_labels.csv",
        help="Input CSV with patch labels.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Default overwrites --csv.",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Disable default 90 degrees CCW rotation before patch extraction.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Review all patches instead of only patches currently labeled tree.",
    )
    return parser.parse_args()


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def load_csv_records(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def save_csv_records(csv_path, fields, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_frame_path(raw_path, csv_path):
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(os.path.join(os.path.dirname(csv_path), raw_path))


def get_patch(image, row_idx, col_idx, rows=GRID_ROWS, cols=GRID_COLS):
    h, w = image.shape[:2]

    y_edges = np.linspace(0, h, rows + 1, dtype=np.int32)
    x_edges = np.linspace(0, w, cols + 1, dtype=np.int32)

    y1 = int(y_edges[row_idx])
    y2 = int(y_edges[row_idx + 1])
    x1 = int(x_edges[col_idx])
    x2 = int(x_edges[col_idx + 1])

    if y2 <= y1 or x2 <= x1:
        return None
    return image[y1:y2, x1:x2]


def build_display_image(patch, info_lines):
    view_h, view_w = 720, 960
    canvas = np.zeros((view_h, view_w, 3), dtype=np.uint8)

    ph, pw = patch.shape[:2]
    scale = min((view_w - 40) / max(1, pw), (view_h - 200) / max(1, ph))
    nw = max(1, int(round(pw * scale)))
    nh = max(1, int(round(ph * scale)))
    patch_resized = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_CUBIC)

    x = (view_w - nw) // 2
    y = 20
    canvas[y:y + nh, x:x + nw] = patch_resized
    cv2.rectangle(canvas, (x, y), (x + nw, y + nh), (0, 255, 255), 2)

    text_y = y + nh + 35
    for line in info_lines:
        cv2.putText(
            canvas,
            line,
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        text_y += 30

    return canvas


def review_labels(records, fields, csv_path, output_path, rotate_ccw_90, show_all):
    required = ["frame_path", "row_index", "col_index", "tree_present"]
    for key in required:
        if key not in fields:
            raise RuntimeError(f"CSV missing required column: {key}")

    if show_all:
        review_indices = list(range(len(records)))
    else:
        review_indices = [
            i for i, rec in enumerate(records)
            if safe_int(rec.get("tree_present", 0), 0) == 1
        ]

    if not review_indices:
        print("No patches to review.")
        return

    win = "Tree Patch Review"
    pos = 0
    yes_count = 0
    no_count = 0
    skip_count = 0

    while pos < len(review_indices):
        rec_idx = review_indices[pos]
        rec = records[rec_idx]

        frame_path = resolve_frame_path(rec.get("frame_path", ""), csv_path)
        row_idx = safe_int(rec.get("row_index", -1), -1)
        col_idx = safe_int(rec.get("col_index", -1), -1)

        if row_idx < 0 or row_idx >= GRID_ROWS or col_idx < 0 or col_idx >= GRID_COLS:
            print(f"Skipping invalid row/col at record {rec_idx}: row={row_idx}, col={col_idx}")
            pos += 1
            skip_count += 1
            continue

        image = cv2.imread(frame_path)
        if image is None:
            print(f"Skipping unreadable frame: {frame_path}")
            pos += 1
            skip_count += 1
            continue

        if rotate_ccw_90:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        patch = get_patch(image, row_idx, col_idx)
        if patch is None:
            print(f"Skipping empty patch at record {rec_idx}")
            pos += 1
            skip_count += 1
            continue

        current_label = safe_int(rec.get("tree_present", 0), 0)
        info = [
            f"Patch {pos + 1}/{len(review_indices)} | record={rec_idx}",
            f"Frame: {os.path.basename(frame_path)} | row={row_idx}, col={col_idx} | current={current_label}",
            "Enter = YES(tree), Space = NO(not tree), q = save and quit",
        ]

        display = build_display_image(patch, info)
        cv2.imshow(win, display)

        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):
            rec["tree_present"] = "1"
            yes_count += 1
            pos += 1
        elif key == 32:
            rec["tree_present"] = "0"
            no_count += 1
            pos += 1
        elif key in (ord("q"), 27):
            break

        if (yes_count + no_count) % 25 == 0 and (yes_count + no_count) > 0:
            save_csv_records(output_path, fields, records)
            print(f"Autosaved after {yes_count + no_count} reviewed patches -> {output_path}")

    save_csv_records(output_path, fields, records)
    cv2.destroyAllWindows()

    print(f"Saved updated labels to: {output_path}")
    print(f"Reviewed patches: {yes_count + no_count}")
    print(f"Set YES: {yes_count}, Set NO: {no_count}, Skipped unreadable/invalid: {skip_count}")


def main():
    args = parse_args()
    csv_path = os.path.abspath(args.csv)
    output_path = os.path.abspath(args.output) if args.output else csv_path

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    records, fields = load_csv_records(csv_path)
    if not records:
        print("CSV is empty.")
        return

    review_labels(
        records=records,
        fields=fields,
        csv_path=csv_path,
        output_path=output_path,
        rotate_ccw_90=(not args.no_rotate),
        show_all=args.show_all,
    )


if __name__ == "__main__":
    main()
