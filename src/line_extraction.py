import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from src.debug import (
    save_connected_components_by_label_temp,
    save_line_boxes_overlays_temp,
    save_line_images_temp,
    visualize_connected_components_temp,
)
from src.io import save_image_temp


def dilate_for_clustering(binary, kernel_size=(3, 3), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(binary, kernel, iterations=iterations)
    return dilated


def extract_connected_components(
    binary_image,
    max_aspect_ratio: float,
    connectivity: int = 8,
    min_area: int = 10,
    max_area: int | None = None,
):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=connectivity
    )

    # index 0 is background
    filtered_indices = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if h / w > max_aspect_ratio or w / h > max_aspect_ratio:
            continue
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        filtered_indices.append(i)

    new_stats = []
    new_centroids = []

    for old_idx in filtered_indices:
        new_stats.append(stats[old_idx])
        new_centroids.append(centroids[old_idx])

    new_stats = np.array(new_stats, dtype=np.int32)
    new_centroids = np.array(new_centroids, dtype=np.float32)

    return new_stats, new_centroids


def cluster_lines_dbscan(
    centroids,
    eps_vertical: float = 15.0,
    eps_horizontal: float = 50.0,
    min_samples: int = 3,
):
    cx = centroids[:, 0]
    cy = centroids[:, 1]

    # scale features so Euclidean distance approximates anisotropic metric
    features = np.stack([cx / eps_horizontal, cy / eps_vertical], axis=1)

    db = DBSCAN(
        eps=1.0,  # distance threshold in normalized feature space
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = db.fit_predict(features)
    return labels


def merge_nearby_clusters(centroids, labels, vertical_threshold=15):

    unique_labels = [l for l in np.unique(labels) if l != -1]
    cluster_means = {l: np.mean(centroids[labels == l, 1]) for l in unique_labels}

    # Sort clusters by vertical position
    sorted_labels = sorted(cluster_means.items(), key=lambda x: x[1])

    # Merge clusters that are close in y
    merged_map = {}
    current_group = 0
    prev_y = None

    for label, y_mean in sorted_labels:
        if prev_y is None or abs(y_mean - prev_y) > vertical_threshold:
            current_group += 1
        merged_map[label] = current_group
        prev_y = y_mean

    # Apply mapping
    merged_labels = np.array([merged_map.get(l, -1) for l in labels], dtype=np.int32)

    return merged_labels


def enforce_line_count_with_vert_clust(centroids, expected_lines):
    y = centroids[:, 1].reshape(-1, 1)

    kmeans = KMeans(n_clusters=expected_lines, random_state=42, n_init="auto")
    final_labels = kmeans.fit_predict(y)

    return final_labels


def merge_centroids_by_y_proximity(centroids, labels, y_distance_threshold: int = 5):
    unique_labels = np.unique(labels)

    # Calculate mean Y for each label
    label_y_means = {}
    for label in unique_labels:
        y_coords = centroids[labels == label, 1]
        label_y_means[label] = np.mean(y_coords)

    # Build merge mapping using union-find approach
    merge_map = {label: label for label in unique_labels}

    def find_root(label):
        if merge_map[label] != label:
            merge_map[label] = find_root(merge_map[label])
        return merge_map[label]

    # Sort labels by Y mean to process in order
    sorted_labels = sorted(unique_labels, key=lambda l: label_y_means[l])

    # Check adjacent labels and merge if close
    for i in range(len(sorted_labels) - 1):
        label_i = sorted_labels[i]
        label_j = sorted_labels[i + 1]

        y_mean_i = label_y_means[label_i]
        y_mean_j = label_y_means[label_j]

        if abs(y_mean_i - y_mean_j) <= y_distance_threshold:
            # Merge j into i
            root_i = find_root(label_i)
            root_j = find_root(label_j)
            merge_map[root_j] = root_i

    # Apply the merge map to all labels
    merged_labels = np.array(
        [find_root(labels[i]) for i in range(len(labels))], dtype=np.int32
    )

    return merged_labels


def sort_clusters_by_vertical_position(centroids, labels):
    unique_labels = np.unique(labels)
    cluster_means = []

    for lab in unique_labels:
        ys = centroids[labels == lab, 1]
        cluster_means.append((lab, np.mean(ys)))

    # Sort by mean y (ascending = top to bottom)
    sorted_pairs = sorted(cluster_means, key=lambda x: x[1])

    # Build mapping old_label -> new_label
    remap = {old: new for new, (old, _) in enumerate(sorted_pairs)}

    # Apply mapping
    new_labels = np.array([remap[l] for l in labels], dtype=np.int32)

    return new_labels


def remove_y_outliers(centroids, labels, method="iqr", threshold=1.5):
    inlier_mask = np.ones(len(labels), dtype=bool)
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        y_values = centroids[mask, 1]

        if method == "iqr":
            q1 = np.percentile(y_values, 25)
            q3 = np.percentile(y_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (y_values < lower_bound) | (y_values > upper_bound)
        elif method == "zscore":
            mean_y = np.mean(y_values)
            std_y = np.std(y_values)
            if std_y > 0:
                z_scores = np.abs((y_values - mean_y) / std_y)
                outlier_mask = z_scores > threshold
            else:
                outlier_mask = np.zeros(len(y_values), dtype=bool)
        else:
            raise ValueError(f"Unknown method: {method}")

        indices = np.where(mask)[0]
        inlier_mask[indices[outlier_mask]] = False

    return inlier_mask


def compute_line_bounding_boxes(stats, labels, image_width):
    line_boxes = {}

    for label, stat in zip(labels, stats):
        x, y, w, h, area = stat
        y2 = y + h

        if label not in line_boxes:
            line_boxes[label] = [y, y2]
        else:
            bx = line_boxes[label]
            bx[0] = min(bx[0], y)
            bx[1] = max(bx[1], y2)

    for label in line_boxes:
        y1, y2 = line_boxes[label]
        line_boxes[label] = (0, y1, image_width, y2)

    return line_boxes


def crop_lines_fixed_height(
    image,
    line_boxes,
    target_height: int,
    target_width: int,
    interpolation=cv2.INTER_LINEAR,
) -> list[np.ndarray]:
    line_images = []

    for label in sorted(line_boxes.keys()):
        (x1, y1, x2, y2) = line_boxes[label]
        crop = image[y1:y2, x1:x2]
        resized = cv2.resize(
            crop, (target_width, target_height), interpolation=interpolation
        )
        line_images.append(resized)

    return line_images


def segment_handwritten_lines(
    image,
    expected_lines: int,
    cc_params: dict | None = None,
    clustering_params: dict | None = None,
    crop_line_params: dict | None = None,
) -> list[np.ndarray]:
    cc_params = cc_params or {}
    clustering_params = clustering_params or {}
    crop_line_params = crop_line_params or {}

    # image = dilate_for_clustering(image)
    # save_image_temp(dilated, "dilated")

    image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 1. Connected components
    stats, centroids = extract_connected_components(image, **cc_params)
    # visualize_connected_components_temp("extract_connected_components",image, stats, centroids)

    # 2. DBSCAN clustering (initial rough line grouping)
    dbscan_labels = cluster_lines_dbscan(centroids=centroids, **clustering_params)
    # visualize_connected_components_temp(
    #     "dbscan_labels", image, stats, centroids, dbscan_labels
    # )

    # Optional: remove noise CCs before K-means
    # valid_mask = dbscan_labels != -1
    # valid_labels = dbscan_labels[valid_mask]
    # valid_centroids = centroids[valid_mask]
    # valid_stats = stats[valid_mask]

    # 3. Enforce exact number of lines using k-means on y
    labels = enforce_line_count_with_vert_clust(
        centroids=centroids, expected_lines=expected_lines
    )
    # visualize_connected_components_temp(
    #     "enforce_line_count", image, stats, centroids, labels
    # )

    # 3b. Merge centroids that are very close in Y
    # labels = merge_centroids_by_y_proximity(
    #     centroids=centroids, labels=labels, y_distance_threshold=y_distance_threshold
    # )

    # 4. Sort clusters vertically
    labels = sort_clusters_by_vertical_position(centroids=centroids, labels=labels)

    # 5. Remove Y-axis outliers from clusters
    # inlier_mask = remove_y_outliers(
    #     centroids=centroids, labels=labels, method="iqr", threshold=threshold_outlier
    # )
    # stats = stats[inlier_mask]
    # centroids = centroids[inlier_mask]
    # labels = labels[inlier_mask]
    # save_connected_components_by_label(image, stats, centroids, labels)

    # 6. Compute rectangular bounding boxes per line
    line_boxes = compute_line_bounding_boxes(
        stats=stats,
        labels=labels,
        image_width=image.shape[1],
    )

    # save_line_boxes_overlays(image, line_boxes)

    # 7. Crop fixed-height line images
    line_images = crop_lines_fixed_height(
        image=image, line_boxes=line_boxes, **crop_line_params
    )

    # save_line_images_temp(line_images)

    return line_images
