from pathlib import Path
from typing import List
import cv2
import numpy as np
import os

from src.io import save_image_temp


def draw_contours(image, contours):
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i, _ in enumerate(contours):
        x = overlay.copy()
        cv2.drawContours(x, contours, i, (255, 255, 0), 2)  # draw the i-th contour
        save_image_temp(x, str(i) + "_" + "contour")


def save_line_segments(
    lines: List[np.ndarray],
    output_dir: str,
    base_name: str = "line",
    format: str = "png",
) -> List[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if not format.startswith("."):
        format = "." + format

    for idx, line_image in enumerate(lines):
        filename = f"{base_name}_{idx:03d}{format}"
        file_path = output_path / filename

        try:
            success = cv2.imwrite(str(file_path), line_image)

            if success:
                saved_files.append(str(file_path))
                print(f"Saved: {file_path}")
            else:
                print(f"Error: Failed to save {file_path}")
        except Exception as e:
            print(f"Error saving {file_path}: {e}")

    print(f"\nTotal lines saved: {len(saved_files)}")
    return saved_files


def visualize_connected_components_temp(
    name,
    image,
    stats,
    centroids,
    labels=None,
    draw_boxes=True,
    draw_centroids=True,
    draw_ids=True,
    box_color=(0, 255, 0),
    centroid_color=(0, 0, 255),
    thickness=1,
    font_scale=0.5,
):
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for i, (stat, centroid) in enumerate(zip(stats, centroids)):
        x, y, w, h, area = stat
        cx, cy = centroid

        if draw_boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, thickness)

        if draw_centroids:
            cv2.circle(
                vis, (int(cx), int(cy)), radius=3, color=centroid_color, thickness=-1
            )

        if draw_ids:
            label_text = str(labels[i]) if labels is not None else str(i + 1)
            cv2.putText(
                vis,
                label_text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                box_color,
                1,
                cv2.LINE_AA,
            )

    save_image_temp(vis, name)


def save_connected_components_by_label_temp(
    image,
    stats,
    centroids,
    labels,
    output_dir="./temp",
    prefix="cc_label",
    draw_boxes=True,
    draw_centroids=True,
    draw_ids=True,
    box_color=(0, 255, 0),
    centroid_color=(0, 0, 255),
    thickness=1,
    font_scale=0.5,
):
    if len(image.shape) == 2:
        base_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base_image = image.copy()

    os.makedirs(output_dir, exist_ok=True)

    unique_labels = np.unique(labels)

    for label in unique_labels:
        vis = base_image.copy()

        for i, (stat, centroid) in enumerate(zip(stats, centroids)):
            if labels[i] != label:
                continue

            x, y, w, h, area = stat
            cx, cy = centroid

            if draw_boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, thickness)

            if draw_centroids:
                cv2.circle(
                    vis,
                    (int(cx), int(cy)),
                    radius=3,
                    color=centroid_color,
                    thickness=-1,
                )

            if draw_ids:
                label_text = str(labels[i])
                cv2.putText(
                    vis,
                    label_text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

        filename = f"{prefix}_{label}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, vis)


def save_line_images_temp(
    line_images: dict,
    output_dir="./temp",
    prefix: str = "line",
    overwrite: bool = True,
    ext: str = ".png",
):
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    # Sort by label for reproducibility
    for label in sorted(line_images.keys()):
        img = line_images[label]

        filename = f"{prefix}_{label+1}{ext}"
        filepath = os.path.join(output_dir, filename)

        if not overwrite and os.path.exists(filepath):
            continue

        # Ensure image is uint8
        if img.dtype != "uint8":
            img = img.astype("uint8")

        # Save image
        cv2.imwrite(filepath, img)
        saved_files.append(filepath)

    return saved_files


def visualize_line_boxes(
    image,
    line_boxes,
    color=(0, 255, 0),
    thickness=2,
    font_scale=0.7,
    text_color=(0, 0, 255),
    text_thickness=2,
):
    # Ensure we work in color
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for label, (y1, y2) in line_boxes.items():
        # Draw rectangle
        cv2.rectangle(vis, (0, int(y1)), (image.shape[1], int(y2)), color, thickness)

        # Draw label above the box
        text = f"Line {label}"
        cv2.putText(
            vis,
            text,
            (0 + 5, int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )

    return vis


def save_line_boxes_overlays_temp(
    image,
    line_boxes,
    output_dir="./temp",
    prefix: str = "line_overlay",
    color=(0, 255, 0),
    thickness=2,
    font_scale=0.7,
    text_color=(0, 0, 255),
    text_thickness=2,
):
    # Ensure we work in color
    if len(image.shape) == 2:
        base_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base_image = image.copy()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Sort by label for reproducibility
    for label in sorted(line_boxes.keys()):
        vis = base_image.copy()
        x1, y1, x2, y2 = line_boxes[label]

        # Draw rectangle for this label
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Draw label above the box
        text = f"Line {label}"
        cv2.putText(
            vis,
            text,
            (int(x1) + 5, int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )

        # Save the image
        filename = f"{prefix}_{label}{'.png'}"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, vis)
