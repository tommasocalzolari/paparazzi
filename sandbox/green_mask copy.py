import cv2
import numpy as np
import os
import csv
import random
import urllib.request

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    HAS_SEMANTIC_DEPS = True
except Exception:
    HAS_SEMANTIC_DEPS = False

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

# Thresholds (tuned to suppress floor)
GREEN_RATIO_THR = 0.10
FAST_DENSITY_THR = 0.0015
VERT_EDGE_THR = 0.008
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

# Optional monocular depth (MiDaS ONNX) for stronger floor-vs-tree separation.
USE_MONOCULAR_DEPTH = True
DEPTH_MODELS_DIR = "./models"
HIGH_ACCURACY_MODEL_NAME = "midas_v21_large_384.onnx"
HIGH_ACCURACY_MODEL_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.onnx"
DEPTH_MODEL_PATH = os.getenv(
    "DEPTH_MODEL_PATH",
    os.path.join(DEPTH_MODELS_DIR, HIGH_ACCURACY_MODEL_NAME),
)
DEPTH_INPUT_SIZE = int(os.getenv("DEPTH_INPUT_SIZE", "384"))
DEPTH_FLOOR_SLOPE_THR = -0.10
DEPTH_FLOOR_STD_THR = 0.10
DEPTH_FLOOR_EDGE_THR = 0.020

# Optional semantic segmentation (ADE20K classes) for tree vs floor disambiguation.
USE_SEMANTIC_SEGMENTATION = True
SEMANTIC_MODEL_NAME = os.getenv("SEMANTIC_MODEL_NAME", "nvidia/segformer-b2-finetuned-ade-512-512")
SEMANTIC_TREE_KEYWORDS = ("tree", "plant", "palm", "bush")
SEMANTIC_FLOOR_KEYWORDS = ("grass", "field", "earth", "ground", "floor", "dirt", "soil")
SEMANTIC_TREE_RATIO_THR = 0.25
SEMANTIC_TREE_MAJORITY_THR = 0.55
SEMANTIC_FLOOR_RATIO_THR = 0.35
SEMANTIC_FLOOR_MAX_FOR_TREE = 0.25

# Majority-tree override: if vegetation fills most of a patch and has strong
# tree-like structure, force a tree label even when base thresholds are borderline.
MAJORITY_TREE_GREEN_RATIO_THR = 0.55
MAJORITY_TREE_FAST_DENSITY_THR = 0.0025
MAJORITY_TREE_VERT_EDGE_THR = 0.020
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


def load_semantic_segmenter(model_name):
    if not HAS_SEMANTIC_DEPS:
        print("Semantic segmentation dependencies missing. Continuing without semantic model.")
        return None, None, None, [], []

    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    except Exception as exc:
        print(f"Failed to load semantic model '{model_name}': {exc}")
        return None, None, None, [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", {})
    tree_ids = find_label_ids(id2label, SEMANTIC_TREE_KEYWORDS)
    floor_ids = find_label_ids(id2label, SEMANTIC_FLOOR_KEYWORDS)

    if not tree_ids:
        print("Warning: no tree-like class IDs found in semantic model labels.")
    if not floor_ids:
        print("Warning: no floor-like class IDs found in semantic model labels.")

    return processor, model, device, tree_ids, floor_ids


def find_label_ids(id2label, keywords):
    ids = []
    for idx, name in id2label.items():
        label = str(name).lower()
        if any(word in label for word in keywords):
            ids.append(int(idx))
    return sorted(set(ids))


def infer_semantic_masks(frame_bgr, processor, model, device, tree_ids, floor_ids):
    if processor is None or model is None:
        return None, None

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    inputs = processor(images=image_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    h, w = frame_bgr.shape[:2]
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

    tree_mask = np.isin(pred, np.array(tree_ids, dtype=np.int32)) if tree_ids else np.zeros((h, w), dtype=bool)
    floor_mask = np.isin(pred, np.array(floor_ids, dtype=np.int32)) if floor_ids else np.zeros((h, w), dtype=bool)
    return tree_mask, floor_mask


def resolve_depth_model_path(model_path):
    if os.path.isabs(model_path):
        return model_path
    return os.path.abspath(model_path)


def maybe_download_high_accuracy_model(resolved_model_path):
    if os.path.exists(resolved_model_path):
        return

    if os.path.basename(resolved_model_path) != HIGH_ACCURACY_MODEL_NAME:
        return

    model_dir = os.path.dirname(resolved_model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    try:
        print(f"Downloading high-accuracy depth model: {HIGH_ACCURACY_MODEL_NAME}")
        urllib.request.urlretrieve(HIGH_ACCURACY_MODEL_URL, resolved_model_path)
        print(f"Saved model to: {resolved_model_path}")
    except Exception as exc:
        print(f"Failed to download {HIGH_ACCURACY_MODEL_NAME}: {exc}")


def load_depth_net(model_path):
    resolved = resolve_depth_model_path(model_path)

    maybe_download_high_accuracy_model(resolved)

    if not os.path.exists(resolved):
        # Fallback to any available ONNX depth model to keep pipeline running.
        available_models = sorted(
            os.path.join(resolve_depth_model_path(DEPTH_MODELS_DIR), file_name)
            for file_name in os.listdir(resolve_depth_model_path(DEPTH_MODELS_DIR))
            if file_name.lower().endswith(".onnx")
        ) if os.path.isdir(resolve_depth_model_path(DEPTH_MODELS_DIR)) else []

        if available_models:
            print(
                f"Requested depth model not found at {resolved}. "
                f"Falling back to {available_models[0]}"
            )
            resolved = available_models[0]
        else:
            print(f"Depth model not found at {resolved}. Continuing without depth.")
            return None

    return cv2.dnn.readNet(resolved)


def preprocess_with_letterbox(frame_rgb, input_size):
    target_w, target_h = input_size
    src_h, src_w = frame_rgb.shape[:2]

    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(frame_rgb, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    padded = np.zeros((target_h, target_w, 3), dtype=np.float32)

    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    padded[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized

    return padded, pad_x, pad_y, resized_w, resized_h


def infer_normalized_distance_map(frame_bgr, net, input_size):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized, pad_x, pad_y, resized_w, resized_h = preprocess_with_letterbox(frame_rgb, input_size)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized - mean) / std

    blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    net.setInput(blob)
    depth = net.forward().squeeze()

    depth = depth[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
    depth = cv2.resize(depth, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    # MiDaS is inverse-depth-like; convert to a distance proxy where lower is closer.
    near_val = float(np.percentile(depth, 2.0))
    far_val = float(np.percentile(depth, 98.0))
    depth_span = max(far_val - near_val, 1e-6)
    normalized_depth = np.clip((depth - near_val) / depth_span, 0.0, 1.0)
    normalized_distance = 1.0 - normalized_depth
    return normalized_distance

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
    depth_net = load_depth_net(DEPTH_MODEL_PATH) if USE_MONOCULAR_DEPTH else None
    use_depth = depth_net is not None
    if use_depth:
        print(f"Monocular depth enabled with model: {resolve_depth_model_path(DEPTH_MODEL_PATH)}")
    else:
        print("Monocular depth disabled.")

    semantic_processor = None
    semantic_model = None
    semantic_device = None
    semantic_tree_ids = []
    semantic_floor_ids = []
    if USE_SEMANTIC_SEGMENTATION:
        (
            semantic_processor,
            semantic_model,
            semantic_device,
            semantic_tree_ids,
            semantic_floor_ids,
        ) = load_semantic_segmenter(SEMANTIC_MODEL_NAME)

    use_semantic = semantic_model is not None
    if use_semantic:
        print(
            f"Semantic segmentation enabled: {SEMANTIC_MODEL_NAME} "
            f"(tree_ids={semantic_tree_ids}, floor_ids={semantic_floor_ids})"
        )
    else:
        print("Semantic segmentation disabled.")

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

        frame_distance_map = None
        if use_depth:
            frame_distance_map = infer_normalized_distance_map(
                img,
                depth_net,
                (DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE),
            )

        frame_tree_sem_mask = None
        frame_floor_sem_mask = None
        if use_semantic:
            frame_tree_sem_mask, frame_floor_sem_mask = infer_semantic_masks(
                img,
                semantic_processor,
                semantic_model,
                semantic_device,
                semantic_tree_ids,
                semantic_floor_ids,
            )

        # Build a frame-level floor prior once, then evaluate overlap per cell.
        _, frame_green_mask = keep_only_green_pixels(img)
        frame_clean_green = clean_green_mask(frame_green_mask)
        frame_floor_mask = build_bottom_connected_floor_mask(frame_clean_green)
        frame_h, frame_w = frame_floor_mask.shape[:2]
        cell_h = frame_h // GRID_ROWS
        cell_w = frame_w // GRID_COLS

        # Split into 4x3 grid
        grid = split_into_grid(img, GRID_ROWS, GRID_COLS)

        for row_idx, row_cells in enumerate(grid):
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

                # Vertical structure (x-gradient catches left/right boundaries of trunks/foliage clumps).
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                vert_edges = np.abs(sobel_x) > 50
                vert_edge_density = np.count_nonzero(vert_edges) / total_pixels
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                hor_edges = np.abs(sobel_y) > 50
                hor_edge_density = np.count_nonzero(hor_edges) / total_pixels
                vert_hor_ratio = vert_edge_density / (hor_edge_density + 1e-6)

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
                if frame_tree_sem_mask is not None:
                    sem_tree_ratio = float(np.count_nonzero(frame_tree_sem_mask[y1:y2, x1:x2])) / total_pixels
                if frame_floor_sem_mask is not None:
                    sem_floor_ratio = float(np.count_nonzero(frame_floor_sem_mask[y1:y2, x1:x2])) / total_pixels

                semantic_tree_majority = (
                    sem_tree_ratio >= SEMANTIC_TREE_MAJORITY_THR
                    and sem_floor_ratio <= SEMANTIC_FLOOR_MAX_FOR_TREE
                )
                semantic_floor_like = (
                    sem_floor_ratio >= SEMANTIC_FLOOR_RATIO_THR
                    and sem_tree_ratio < SEMANTIC_TREE_RATIO_THR
                )

                depth_slope = -1.0
                depth_std = -1.0
                depth_edge_density = -1.0
                depth_floor_like = False
                if frame_distance_map is not None:
                    cell_distance = frame_distance_map[y1:y2, x1:x2]
                    band_h = max(1, cell_distance.shape[0] // 3)
                    top_mean = float(np.mean(cell_distance[:band_h, :]))
                    bottom_mean = float(np.mean(cell_distance[-band_h:, :]))
                    depth_slope = bottom_mean - top_mean
                    depth_std = float(np.std(cell_distance))

                    depth_gx = cv2.Sobel(cell_distance, cv2.CV_32F, 1, 0, ksize=3)
                    depth_gy = cv2.Sobel(cell_distance, cv2.CV_32F, 0, 1, ksize=3)
                    depth_mag = cv2.magnitude(depth_gx, depth_gy)
                    depth_edge_density = float(np.count_nonzero(depth_mag > 0.08)) / total_pixels

                    # Floor usually gets smoother and farther toward the top of the patch.
                    depth_floor_like = (
                        depth_slope <= DEPTH_FLOOR_SLOPE_THR
                        and depth_std <= DEPTH_FLOOR_STD_THR
                        and depth_edge_density <= DEPTH_FLOOR_EDGE_THR
                    )

                local_floor_like = is_floor_like_region(cleaned_green_mask)
                majority_tree_like = (
                    green_ratio >= MAJORITY_TREE_GREEN_RATIO_THR
                    and fast_density >= MAJORITY_TREE_FAST_DENSITY_THR
                    and vert_edge_density >= MAJORITY_TREE_VERT_EDGE_THR
                    and entropy >= MAJORITY_TREE_ENTROPY_THR
                    and comp_ratio >= MAJORITY_TREE_COMPONENT_RATIO_THR
                    and comp_height_ratio >= MAJORITY_TREE_HEIGHT_RATIO_THR
                    and comp_width_ratio <= MAJORITY_TREE_WIDTH_RATIO_MAX
                    and vert_hor_ratio >= MAJORITY_TREE_VERT_HOR_RATIO_THR
                    and not depth_floor_like
                )

                majority_tree_like = majority_tree_like or semantic_tree_majority

                floor_like = (
                    local_floor_like
                    or (floor_overlap >= FLOOR_OVERLAP_THR)
                    or depth_floor_like
                    or semantic_floor_like
                ) and (not majority_tree_like)

                base_tree_like = (
                    green_ratio > GREEN_RATIO_THR
                    and fast_density > FAST_DENSITY_THR
                    and vert_edge_density > VERT_EDGE_THR
                    and entropy > ENTROPY_THR
                ) or (sem_tree_ratio >= SEMANTIC_TREE_RATIO_THR and sem_floor_ratio < SEMANTIC_FLOOR_RATIO_THR)

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
                    vert_edge_density,
                    hor_edge_density,
                    vert_hor_ratio,
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
            "vert_edge_density",
            "hor_edge_density",
            "vert_hor_ratio",
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
