import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

output_dir = os.path.join(BASE_DIR,"flask_server", "ActivePolyOut")
os.makedirs(output_dir, exist_ok=True)

video_path = os.path.join(BASE_DIR,"dynamic-master","output","segmentation","deeplabv3_resnet50_pretrained","videos","0X1A0A263B22CCD966.avi")
def build_full_wall(contour):
    apex_idx = np.argmin(contour[:, 1])  # Apex is visually at the top
    apex = contour[apex_idx]
    wall_path = np.concatenate([contour[apex_idx:], contour[:apex_idx]], axis=0)
    return wall_path, apex

def get_lv_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2)

def split_apex_boundary_with_base(contour):
    # --- Apex = top point ---
    apex_idx = np.argmin(contour[:, 1])
    apex = contour[apex_idx]

    # --- Base points (low Y) ---
    base_y_thresh = np.max(contour[:, 1]) - 10
    base_indices = np.where(contour[:, 1] >= base_y_thresh)[0]
    base_pts = contour[base_indices]

    if len(base_pts) < 2:
        raise ValueError("Insufficient base points.")

    # Farthest base points = start & end
    dists = np.linalg.norm(base_pts[:, None] - base_pts[None], axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    start = base_pts[i]
    end = base_pts[j]

    # --- Line from start to end: ax + by + c = 0 ---
    x1, y1 = start
    x2, y2 = end
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    # --- Filter points on same side as apex ---
    def line_side(p): return a * p[0] + b * p[1] + c
    apex_side = line_side(apex)
    filtered = np.array([pt for pt in contour if line_side(pt) * apex_side > 0])

    # --- Classify points as left/right based on line from apex to midpoint ---
    mid = (start + end) / 2
    x1, y1 = apex
    x2, y2 = mid
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    def side_fn(p):
        return a * p[0] + b * p[1] + c

    left = np.array([pt for pt in filtered if side_fn(pt) < 0])
    right = np.array([pt for pt in filtered if side_fn(pt) > 0])


    return left, right, apex, start, end


def clean_wall_by_distance(points, p_start, p_end, num_internal=4):
    """
    Retain `num_internal` equally spaced points between p_start and p_end (based on geometric distance),
    plus the start and end points themselves.
    """
    points = np.array(points)

    # Vector from start to end
    vec = p_end - p_start
    total_dist = np.linalg.norm(vec)
    if total_dist == 0:
        return np.array([p_start, p_end])

    vec_unit = vec / total_dist

    # Project all points onto the line segment to get distance along it
    projections = np.dot(points - p_start, vec_unit)

    # Keep points within segment [0, total_dist]
    mask = (projections >= 0) & (projections <= total_dist)
    valid_points = points[mask]
    valid_proj = projections[mask]

    # Determine target distances for sampling
    step = total_dist / (num_internal + 1)
    targets = [(i + 1) * step for i in range(num_internal)]

    # Select closest points to each target
    selected = [p_start]
    for t in targets:
        idx = np.argmin(np.abs(valid_proj - t))
        pt = valid_points[idx]
        if not any(np.allclose(pt, sel) for sel in selected):
            selected.append(pt)
    selected.append(p_end)

    return np.array(selected)


def point_exists(p, arr, tol=1e-3):
    """Check if point p already exists in arr (within a tolerance)."""
    return np.any(np.linalg.norm(arr - p, axis=1) < tol)

def fit_polynomial_arc(points, degree=3, lambda_reg=0.01):
    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    t = np.concatenate(([0], np.cumsum(dists)))  # Arc-length parameterization

    A = np.vander(t, degree + 1, increasing=True)
    x = points[:, 0]
    y = points[:, 1]

    ATA = A.T @ A
    reg = lambda_reg * np.eye(degree + 1)

    coeffs_x = np.linalg.solve(ATA + reg, A.T @ x)
    coeffs_y = np.linalg.solve(ATA + reg, A.T @ y)

    return coeffs_x, coeffs_y, t


def eval_poly_arc(coeffs_x, coeffs_y, t_vals):
    A = np.vander(t_vals, len(coeffs_x), increasing=True)
    x_vals = A @ coeffs_x
    y_vals = A @ coeffs_y
    return np.column_stack((x_vals, y_vals))


def extract_segment(contour, p_start, p_end):
    """Extract and order segment from p_start to p_end, preserving direction."""
    i_start = np.argmin(np.linalg.norm(contour - p_start, axis=1))
    i_end = np.argmin(np.linalg.norm(contour - p_end, axis=1))

    # Try forward direction
    if i_start <= i_end:
        segment = contour[i_start:i_end+1]
    else:
        segment = np.concatenate([contour[i_start:], contour[:i_end+1]])

    # Confirm segment starts at p_start and ends at p_end
    if (np.linalg.norm(segment[0] - p_start) > np.linalg.norm(segment[-1] - p_start) or
        np.linalg.norm(segment[-1] - p_end) > np.linalg.norm(segment[0] - p_end)):
        segment = segment[::-1]
    return segment

def get_arc_length(points):
    diffs = np.diff(points, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    return np.concatenate(([0], np.cumsum(dists)))

def split_segments(points, n_segments=3):
    arc = get_arc_length(points)
    seg_bounds = np.linspace(0, arc[-1], n_segments + 1)
    segments = []
    for i in range(n_segments):
        mask = (arc >= seg_bounds[i]) & (arc < seg_bounds[i+1]) if i < n_segments - 1 else (arc >= seg_bounds[i])
        segments.append(points[mask])
    return segments

def findActivePoly():
    print(BASE_DIR)
    sum=0
    count=0
    avg=0

    areas = []
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color range
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Threshold red
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        contour = get_lv_contour(red_mask)
        if contour is None:
            print(f"Frame {frame_idx}: No LV contour found.")
            frame_idx += 1
            continue

        try:
            left, right, apex, start, end = split_apex_boundary_with_base(contour)

            left = left[~np.all(left == start, axis=1)]
            left = left[~np.all(left == end, axis=1)]

            right = right[~np.all(right == start, axis=1)]
            right = right[~np.all(right == end, axis=1)]

            left = clean_wall_by_distance(left, start, apex)
            right = clean_wall_by_distance(right, apex, end)

            # Ensure start and end are included
            if not point_exists(start, left):
                left = np.vstack([start, left])  # Add start at the beginning

            if not point_exists(end, right):
                right = np.vstack([right, end])  # Add end at the end

            left_seg = extract_segment(left, start, apex)
            right_seg = extract_segment(right, apex, end)

            coeffs_lx, coeffs_ly, t_left = fit_polynomial_arc(left_seg)
            coeffs_rx, coeffs_ry, t_right = fit_polynomial_arc(right_seg)

            t_sample_l = np.linspace(t_left.min(), t_left.max(), 200)
            t_sample_r = np.linspace(t_right.min(), t_right.max(), 200)

            left_curve = eval_poly_arc(coeffs_lx, coeffs_ly, t_sample_l)
            right_curve = eval_poly_arc(coeffs_rx, coeffs_ry, t_sample_r)

            segments_left = split_segments(left_curve, 3)
            segments_right = split_segments(right_curve, 3)
            segments = segments_left + [np.array([apex])] + segments_right

            # Optional visualization per frame
            height, width = red_mask.shape
            black_bg = np.zeros((height, width))

            # Scale and thickness
            scale = 3
            thickness = 10  # Increase for thicker lines

            # Original shape
            h, w = red_mask.shape
            canvas_big = np.zeros((h * scale, w * scale), dtype=np.uint8)

            # Combine all wall points
            all_points = np.vstack(segments)
            scaled_points = (all_points * scale).astype(np.int32).reshape(-1, 1, 2)

            # Draw thicker, smoother line
            cv2.polylines(canvas_big, [scaled_points], isClosed=True, color=255, thickness=thickness)

            # Downscale to original size
            canvas = cv2.resize(canvas_big, (w, h), interpolation=cv2.INTER_AREA)

            # Crop around the drawn area
            s = 25  # margin
            ys, xs = np.where(canvas > 0)

            if ys.size > 0 and xs.size > 0:
                # Margin
                s = 25

                # Bounding box
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()

                # Center of the box
                cy = (y_min + y_max) // 2
                cx = (x_min + x_max) // 2

                # Half-size of the square box (whichever side is longer)
                half_size = max((y_max - y_min), (x_max - x_min)) // 2 + s

                # New square bounds
                y_min = max(0, cy - half_size)
                y_max = min(canvas.shape[0], cy + half_size)
                x_min = max(0, cx - half_size)
                x_max = min(canvas.shape[1], cx + half_size)


                cropped = canvas[y_min:y_max, x_min:x_max]

                # fig, ax = plt.subplots(figsize=(4, 4))
                # ax.imshow(cropped, cmap='gray')
                # all_points = np.vstack(segments)
                # ax.plot(all_points[:, 0], all_points[:, 1], 'w.', markersize=2)
                # ax.axis('off')
                # plt.tight_layout(pad=0)

                contour_area = np.sum(cropped > 0)
                sum=sum+contour_area
                count=count+1
                avg=sum/count

                # Get original dimensions
                # H_original, W_original = cropped.shape[:2]  # Should be height and width of original binary mask

                # Apply scaling to 224x224
                scale_factor =4.1 #(224 * 224) / (H_original * W_original)
                area_scaled = contour_area * scale_factor

                areas.append(area_scaled)


                #area_data.append((frame_idx, contour_area, area_scaled))
                # print(f"Frame {frame_idx}: Raw Area = {contour_area} px², Scaled Area = {area_scaled:.2f} px²")


                # output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
                # fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
                # plt.close(fig)
                output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(output_path, cropped)

            else:
                print(f"Frame {frame_idx}: No content to crop, skipping.")


        except Exception as e:
            print(f"Frame {frame_idx}: Error - {e}")

        frame_idx += 1

    cap.release()
    # print(f"Original canvas size: {W_original} x {H_original}")
    # print(f"Average area: {avg:.2f} px²")

    np.save(os.path.join(output_dir, "scaled_areas.npy"), np.array(areas))
    print(f"Saved scaled areas to {os.path.join(output_dir, 'scaled_areas.npy')}")


    return 1

# def findActivePoly(*args, **kwargs):

#     return 1
