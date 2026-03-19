import cv2
import numpy as np
import os
import csv
import random

# ----------------------------
# CONFIG
# ----------------------------
folder_path = "./datasets/Organised/20260306-handFlying-2"
output_csv = "balanced_tree_patch_labels.csv"
rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "0"))

green_lower = np.array([25, 30, 60], dtype=np.uint8)
green_upper = np.array([100, 255, 255], dtype=np.uint8)

GRID_ROWS = 4
GRID_COLS = 3
TARGET_ROW_INDEX = 2  # 3rd row (0-based)

# Thresholds (tuned to suppress floor)
GREEN_RATIO_THR = 0.10
FAST_DENSITY_THR = 0.0015
GFTT_DENSITY_THR = 0.0012
VERT_EDGE_THR = 0.008
GRAD_MAG_DENSITY_THR = 0.010
CANNY_DENSITY_THR = 0.012
ENTROPY_THR = 20.0

# Floor-rejection heuristics for each grid cell
FLOOR_BOTTOM_BAND = 0.25
FLOOR_TOP_BAND = 0.25
FLOOR_BOTTOM_TO_TOP_RATIO = 2.2
FLOOR_WIDE_RATIO = 0.65
FLOOR_FLAT_RATIO = 0.60

# Frame-level floor detection (captures large green regions connected to bottom)
FRAME_FLOOR_MIN_WIDTH_RATIO = 0.22
FRAME_FLOOR_MIN_AREA_RATIO = 0.01
FLOOR_OVERLAP_THR = 0.30

# Majority-tree override: if vegetation fills most of a patch and has strong
# tree-like structure, force a tree label even when base thresholds are borderline.
MAJORITY_TREE_GREEN_RATIO_THR = 0.55
MAJORITY_TREE_FAST_DENSITY_THR = 0.0025
MAJORITY_TREE_GFTT_DENSITY_THR = 0.0018
MAJORITY_TREE_VERT_EDGE_THR = 0.020
MAJORITY_TREE_GRAD_MAG_DENSITY_THR = 0.020
MAJORITY_TREE_CANNY_DENSITY_THR = 0.025
MAJORITY_TREE_ENTROPY_THR = 120.0
MAJORITY_TREE_COMPONENT_RATIO_THR = 0.40
MAJORITY_TREE_HEIGHT_RATIO_THR = 0.65
MAJORITY_TREE_WIDTH_RATIO_MAX = 0.95
MAJORITY_TREE_VERT_HOR_RATIO_THR = 0.90

# ----------------------------
# HELPERS
# ----------------------------
def get_image_files(folder):
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    )

def keep_only_green_pixels(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=green_mask)
    return masked, green_mask

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


def clean_green_mask(green_mask):
    # Remove isolated pixels and fill small holes before shape analysis.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    return cleaned


def is_floor_like_region(mask):
    h, w = mask.shape[:2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    _, y, bw, bh = cv2.boundingRect(largest)

    touches_bottom = (y + bh) >= (h - 1)
    wide_region = (bw / w) >= FLOOR_WIDE_RATIO
    flat_region = (bh / h) <= FLOOR_FLAT_RATIO

    top_h = max(1, int(h * FLOOR_TOP_BAND))
    bottom_h = max(1, int(h * FLOOR_BOTTOM_BAND))
    top_ratio = np.count_nonzero(mask[:top_h, :]) / (top_h * w)
    bottom_ratio = np.count_nonzero(mask[h - bottom_h:, :]) / (bottom_h * w)
    bottom_heavy = bottom_ratio > ((top_ratio + 1e-6) * FLOOR_BOTTOM_TO_TOP_RATIO)

    return touches_bottom and wide_region and flat_region and bottom_heavy


def build_bottom_connected_floor_mask(cleaned_green_mask):
    h, w = cleaned_green_mask.shape[:2]
    total = h * w

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_green_mask, connectivity=8)
    floor_mask = np.zeros_like(cleaned_green_mask)

    for label in range(1, n_labels):
        x, y, bw, bh, area = stats[label]

        touches_bottom = (y + bh) >= (h - 2)
        wide_region = (bw / w) >= FRAME_FLOOR_MIN_WIDTH_RATIO
        large_region = (area / total) >= FRAME_FLOOR_MIN_AREA_RATIO

        if touches_bottom and wide_region and large_region:
            floor_mask[labels == label] = 255

    return floor_mask


def largest_component_stats(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0

    h, w = mask.shape[:2]
    total = h * w
    largest = max(contours, key=cv2.contourArea)
    area_ratio = float(cv2.contourArea(largest)) / max(1, total)
    _, _, bw, bh = cv2.boundingRect(largest)
    height_ratio = float(bh) / max(1, h)
    width_ratio = float(bw) / max(1, w)
    return area_ratio, height_ratio, width_ratio

# ----------------------------
# MAIN
# ----------------------------
def main():
    image_files = get_image_files(folder_path)
    if not image_files:
        print("No images found.")
        return

    if MAX_FRAMES > 0:
        image_files = image_files[:MAX_FRAMES]

    fast = cv2.FastFeatureDetector_create()
    gftt_params = dict(
        maxCorners=150,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )

    # Read first image for size
    first = cv2.imread(os.path.join(folder_path, image_files[0]))
    h0, w0 = first.shape[:2]

    samples = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        if img is None:
            continue

        img = cv2.resize(img, (w0, h0))
        img = cv2.rotate(img, rotation_code)

        # Build a frame-level floor prior once, then evaluate overlap per cell.
        _, frame_green_mask = keep_only_green_pixels(img)
        frame_clean_green = clean_green_mask(frame_green_mask)
        frame_floor_mask = build_bottom_connected_floor_mask(frame_clean_green)
        frame_h, frame_w = frame_floor_mask.shape[:2]
        cell_h = frame_h // GRID_ROWS
        cell_w = frame_w // GRID_COLS

        # Split into 4x3 grid
        grid = split_into_grid(img, GRID_ROWS, GRID_COLS)

        if TARGET_ROW_INDEX >= len(grid):
            continue

        row_idx = TARGET_ROW_INDEX
        row_cells = grid[row_idx]
        for col_idx, patch in enumerate(row_cells):
                cell_idx = row_idx * GRID_COLS + col_idx
                total_pixels = patch.shape[0] * patch.shape[1]

                # Green mask
                masked, green_mask = keep_only_green_pixels(patch)
                cleaned_green_mask = clean_green_mask(green_mask)
                green_pixels = np.count_nonzero(cleaned_green_mask)
                green_ratio = green_pixels / total_pixels

                # FAST keypoints
                gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
                kps = fast.detect(gray, None)
                fast_density = len(kps) / total_pixels

                # Additional corner-based texture cue.
                gftt_pts = cv2.goodFeaturesToTrack(gray, mask=None, **gftt_params)
                gftt_count = 0 if gftt_pts is None else len(gftt_pts)
                gftt_density = gftt_count / total_pixels

                # Vertical structure (x-gradient catches left/right boundaries of trunks/foliage clumps).
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                vert_edges = np.abs(sobel_x) > 50
                vert_edge_density = np.count_nonzero(vert_edges) / total_pixels
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                hor_edges = np.abs(sobel_y) > 50
                hor_edge_density = np.count_nonzero(hor_edges) / total_pixels
                vert_hor_ratio = vert_edge_density / (hor_edge_density + 1e-6)
                grad_mag = cv2.magnitude(sobel_x.astype(np.float32), sobel_y.astype(np.float32))
                grad_mag_density = np.count_nonzero(grad_mag > 70.0) / total_pixels
                canny = cv2.Canny(gray, 50, 120)
                canny_density = np.count_nonzero(canny) / total_pixels

                # Entropy (Laplacian variance)
                entropy = cv2.Laplacian(gray, cv2.CV_64F).var()
                comp_ratio, comp_height_ratio, comp_width_ratio = largest_component_stats(cleaned_green_mask)

                # Cell-level overlap with frame-level bottom-connected floor components.
                y1 = row_idx * cell_h
                y2 = (row_idx + 1) * cell_h
                x1 = col_idx * cell_w
                x2 = (col_idx + 1) * cell_w
                floor_overlap = np.count_nonzero(frame_floor_mask[y1:y2, x1:x2]) / total_pixels

                sem_tree_ratio = 0.0
                sem_floor_ratio = 0.0
                semantic_tree_majority = False
                semantic_floor_like = False

                depth_slope = -1.0
                depth_std = -1.0
                depth_edge_density = -1.0
                depth_floor_like = False

                local_floor_like = is_floor_like_region(cleaned_green_mask)
                feature_density = max(fast_density, gftt_density)
                majority_tree_like = (
                    green_ratio >= MAJORITY_TREE_GREEN_RATIO_THR
                    and fast_density >= MAJORITY_TREE_FAST_DENSITY_THR
                    and gftt_density >= MAJORITY_TREE_GFTT_DENSITY_THR
                    and vert_edge_density >= MAJORITY_TREE_VERT_EDGE_THR
                    and grad_mag_density >= MAJORITY_TREE_GRAD_MAG_DENSITY_THR
                    and canny_density >= MAJORITY_TREE_CANNY_DENSITY_THR
                    and entropy >= MAJORITY_TREE_ENTROPY_THR
                    and comp_ratio >= MAJORITY_TREE_COMPONENT_RATIO_THR
                    and comp_height_ratio >= MAJORITY_TREE_HEIGHT_RATIO_THR
                    and comp_width_ratio <= MAJORITY_TREE_WIDTH_RATIO_MAX
                    and vert_hor_ratio >= MAJORITY_TREE_VERT_HOR_RATIO_THR
                )

                floor_like = (
                    local_floor_like
                    or (floor_overlap >= FLOOR_OVERLAP_THR)
                ) and (not majority_tree_like)

                base_tree_like = (
                    green_ratio > GREEN_RATIO_THR
                    and feature_density > min(FAST_DENSITY_THR, GFTT_DENSITY_THR)
                    and vert_edge_density > VERT_EDGE_THR
                    and grad_mag_density > GRAD_MAG_DENSITY_THR
                    and canny_density > CANNY_DENSITY_THR
                    and entropy > ENTROPY_THR
                )

                # Classification rule
                if (not floor_like) and (base_tree_like or majority_tree_like):
                    label = 1
                else:
                    label = 0

                samples.append([
                    image_path,
                    cell_idx,
                    row_idx,
                    col_idx,
                    green_ratio,
                    fast_density,
                    gftt_density,
                    vert_edge_density,
                    hor_edge_density,
                    vert_hor_ratio,
                    grad_mag_density,
                    canny_density,
                    entropy,
                    comp_ratio,
                    comp_height_ratio,
                    comp_width_ratio,
                    sem_tree_ratio,
                    sem_floor_ratio,
                    floor_overlap,
                    depth_slope,
                    depth_std,
                    depth_edge_density,
                    int(semantic_tree_majority),
                    int(semantic_floor_like),
                    int(majority_tree_like),
                    int(floor_like),
                    label
                ])

        print(f"Processed: {image_file}")

    # ----------------------------
    # BALANCE THE DATASET
    # ----------------------------
    tree_samples = [s for s in samples if s[-1] == 1]
    no_tree_samples = [s for s in samples if s[-1] == 0]

    n_tree = len(tree_samples)
    n_no_tree = len(no_tree_samples)

    print(f"Trees: {n_tree}, No-trees: {n_no_tree}")

    if n_tree == 0 or n_no_tree == 0:
        print("Warning: one class is empty, cannot balance.")
        balanced = samples
    else:
        target_n = min(n_tree, n_no_tree)
        tree_balanced = random.sample(tree_samples, target_n)
        no_tree_balanced = random.sample(no_tree_samples, target_n)
        balanced = tree_balanced + no_tree_balanced
        random.shuffle(balanced)

    # ----------------------------
    # WRITE BALANCED CSV
    # ----------------------------
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_path",
            "cell_index",
            "row_index",
            "col_index",
            "green_ratio",
            "fast_density",
            "gftt_density",
            "vert_edge_density",
            "hor_edge_density",
            "vert_hor_ratio",
            "grad_mag_density",
            "canny_density",
            "entropy",
            "component_ratio",
            "component_height_ratio",
            "component_width_ratio",
            "semantic_tree_ratio",
            "semantic_floor_ratio",
            "floor_overlap",
            "depth_slope",
            "depth_std",
            "depth_edge_density",
            "semantic_tree_majority",
            "semantic_floor_like",
            "majority_tree_like",
            "floor_like",
            "tree_present"
        ])
        writer.writerows(balanced)

    print(f"Balanced dataset saved to {output_csv}")
    print(f"Final count per class: {len(balanced)//2}")

if __name__ == "__main__":
    main()
