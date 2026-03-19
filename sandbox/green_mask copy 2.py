import csv
import os
import random
from contextlib import nullcontext

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
except Exception as exc:
    raise RuntimeError(
        "Missing dependencies for semantic segmentation. "
        "Install with: pip install torch torchvision transformers pillow"
    ) from exc


# ----------------------------
# CONFIG
# ----------------------------
folder_path = "./datasets/Organised/20260306-handFlying-2"
output_csv = "balanced_tree_patch_labels.csv"
rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "0"))

GRID_ROWS = 4
GRID_COLS = 3

# Strong ADE20K model for semantic segmentation.
SEMANTIC_MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL_NAME",
    "nvidia/segformer-b5-finetuned-ade-640-640",
)

# GPU utilization knobs.
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
USE_ALL_GPUS = os.getenv("USE_ALL_GPUS", "1") == "1"

# Label keywords used to map semantic class IDs to "tree-like" classes.
SEMANTIC_TREE_KEYWORDS = ("tree", "palm")

# Keep only meaningful tree blobs to avoid noisy tiny boxes.
MIN_TREE_COMPONENT_AREA = int(os.getenv("MIN_TREE_COMPONENT_AREA", "500"))

# Occupancy rule requested by user: >50% patch coverage by one tree bbox.
PATCH_OCCUPANCY_THRESHOLD = float(os.getenv("PATCH_OCCUPANCY_THRESHOLD", "0.50"))


# ----------------------------
# HELPERS
# ----------------------------
def get_image_files(folder):
    return sorted(
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    )


def find_label_ids(id2label, keywords):
    ids = []
    for idx, name in id2label.items():
        label = str(name).lower()
        if any(word in label for word in keywords):
            ids.append(int(idx))
    return sorted(set(ids))


def load_semantic_segmenter(model_name):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if device.type == "cuda" and USE_ALL_GPUS and gpu_count > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    id2label = getattr(base_model.config, "id2label", {})
    tree_ids = find_label_ids(id2label, SEMANTIC_TREE_KEYWORDS)

    if not tree_ids:
        raise RuntimeError(
            "No tree-like class IDs found in model labels. "
            f"Model={model_name}, labels={id2label}"
        )

    return processor, model, device, tree_ids, gpu_count


def infer_tree_masks(frames_bgr, processor, model, device, tree_ids):
    images_pil = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames_bgr]
    inputs = processor(images=images_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():
        with autocast_ctx:
            outputs = model(**inputs)

    logits = outputs.logits
    tree_masks = []

    for i, frame_bgr in enumerate(frames_bgr):
        h, w = frame_bgr.shape[:2]
        one = logits[i : i + 1]
        one = F.interpolate(one, size=(h, w), mode="bilinear", align_corners=False)
        pred = one.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

        tree_mask = np.isin(pred, np.array(tree_ids, dtype=np.int32)).astype(np.uint8) * 255

        # Smooth boundaries and suppress tiny segmentation artifacts.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_OPEN, kernel_open)
        tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_CLOSE, kernel_close)
        tree_masks.append(tree_mask)

    return tree_masks


def extract_tree_bboxes(tree_mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tree_mask, connectivity=8)

    bboxes = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        bboxes.append((int(x), int(y), int(w), int(h), int(area)))

    return bboxes


def patch_rect(row_idx, col_idx, cell_h, cell_w):
    y1 = row_idx * cell_h
    y2 = (row_idx + 1) * cell_h
    x1 = col_idx * cell_w
    x2 = (col_idx + 1) * cell_w
    return x1, y1, x2, y2


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def compute_patch_occupancy(row_idx, col_idx, cell_h, cell_w, bboxes):
    patch = patch_rect(row_idx, col_idx, cell_h, cell_w)
    patch_area = cell_h * cell_w

    max_coverage = 0.0
    overlap_count = 0

    for x, y, w, h, _ in bboxes:
        bbox = (x, y, x + w, y + h)
        overlap = intersection_area(patch, bbox)
        if overlap <= 0:
            continue

        overlap_count += 1
        coverage = overlap / patch_area
        if coverage > max_coverage:
            max_coverage = coverage

    is_occupied = int(max_coverage > PATCH_OCCUPANCY_THRESHOLD)
    return is_occupied, max_coverage, overlap_count


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

    processor, model, device, tree_ids, gpu_count = load_semantic_segmenter(SEMANTIC_MODEL_NAME)
    print(
        "Semantic segmentation enabled: "
        f"{SEMANTIC_MODEL_NAME} (tree_ids={tree_ids})"
    )
    if device.type == "cuda":
        print(
            f"Using CUDA with {gpu_count} GPU(s), BATCH_SIZE={BATCH_SIZE}, "
            f"DataParallel={'ON' if (USE_ALL_GPUS and gpu_count > 1) else 'OFF'}"
        )
    else:
        print("CUDA not available, running on CPU.")

    first = cv2.imread(os.path.join(folder_path, image_files[0]))
    if first is None:
        print("Failed to read first image.")
        return

    h0, w0 = first.shape[:2]
    frame_h = w0
    frame_w = h0
    cell_h = frame_h // GRID_ROWS
    cell_w = frame_w // GRID_COLS

    samples = []

    batch_paths = []
    batch_frames = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        if img is None:
            continue

        img = cv2.resize(img, (w0, h0))
        img = cv2.rotate(img, rotation_code)

        batch_paths.append(image_path)
        batch_frames.append(img)

        if len(batch_frames) < max(1, BATCH_SIZE):
            continue

        tree_masks = infer_tree_masks(batch_frames, processor, model, device, tree_ids)
        for idx, tree_mask in enumerate(tree_masks):
            image_path_i = batch_paths[idx]
            bboxes = extract_tree_bboxes(tree_mask, MIN_TREE_COMPONENT_AREA)

            for row_idx in range(GRID_ROWS):
                for col_idx in range(GRID_COLS):
                    cell_idx = row_idx * GRID_COLS + col_idx
                    occupied, max_cov, overlap_count = compute_patch_occupancy(
                        row_idx,
                        col_idx,
                        cell_h,
                        cell_w,
                        bboxes,
                    )

                    samples.append(
                        [
                            image_path_i,
                            cell_idx,
                            row_idx,
                            col_idx,
                            max_cov,
                            overlap_count,
                            len(bboxes),
                            occupied,
                        ]
                    )

            print(f"Processed: {os.path.basename(image_path_i)} (tree_boxes={len(bboxes)})")

        batch_paths = []
        batch_frames = []

    if batch_frames:
        tree_masks = infer_tree_masks(batch_frames, processor, model, device, tree_ids)
        for idx, tree_mask in enumerate(tree_masks):
            image_path_i = batch_paths[idx]
            bboxes = extract_tree_bboxes(tree_mask, MIN_TREE_COMPONENT_AREA)

            for row_idx in range(GRID_ROWS):
                for col_idx in range(GRID_COLS):
                    cell_idx = row_idx * GRID_COLS + col_idx
                    occupied, max_cov, overlap_count = compute_patch_occupancy(
                        row_idx,
                        col_idx,
                        cell_h,
                        cell_w,
                        bboxes,
                    )

                    samples.append(
                        [
                            image_path_i,
                            cell_idx,
                            row_idx,
                            col_idx,
                            max_cov,
                            overlap_count,
                            len(bboxes),
                            occupied,
                        ]
                    )

            print(f"Processed: {os.path.basename(image_path_i)} (tree_boxes={len(bboxes)})")

    tree_samples = [s for s in samples if s[-1] == 1]
    no_tree_samples = [s for s in samples if s[-1] == 0]

    n_tree = len(tree_samples)
    n_no_tree = len(no_tree_samples)

    print(f"Occupied patches: {n_tree}, Empty patches: {n_no_tree}")

    if n_tree == 0 or n_no_tree == 0:
        print("Warning: one class is empty, cannot balance.")
        balanced = samples
    else:
        target_n = min(n_tree, n_no_tree)
        balanced = random.sample(tree_samples, target_n) + random.sample(no_tree_samples, target_n)
        random.shuffle(balanced)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_path",
                "cell_index",
                "row_index",
                "col_index",
                "max_bbox_patch_coverage",
                "overlapping_tree_bbox_count",
                "frame_tree_bbox_count",
                "tree_present",
            ]
        )
        writer.writerows(balanced)

    print(f"Balanced dataset saved to {output_csv}")
    if len(balanced) % 2 == 0 and len(balanced) > 0:
        print(f"Final count per class: {len(balanced) // 2}")
    else:
        print(f"Final sample count: {len(balanced)}")


if __name__ == "__main__":
    main()
