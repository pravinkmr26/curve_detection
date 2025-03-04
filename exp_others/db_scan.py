import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN


def group_curves(curve_segments, threshold=20):
    """Groups curve segments based on proximity of start/end points."""
    start_points = [curve[0] for curve in curve_segments]
    end_points = [curve[-1] for curve in curve_segments]

    all_points = np.concatenate([start_points, end_points])

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=threshold, min_samples=1)  # Adjust eps
    clusters = dbscan.fit_predict(all_points)

    grouped_curves = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in grouped_curves:
            grouped_curves[cluster_id] = []
        if i < len(curve_segments):  # Start points
            grouped_curves[cluster_id].append({"type": "start", "index": i})
        else:  # End points
            grouped_curves[cluster_id].append(
                {"type": "end", "index": i - len(curve_segments)}
            )

    final_curves = []
    for cluster_id, points in grouped_curves.items():
        start_indices = [p["index"] for p in points if p["type"] == "start"]
        end_indices = [p["index"] for p in points if p["type"] == "end"]

        for start_index in start_indices:
            for end_index in end_indices:
                if start_index == end_index:
                    final_curves.append(curve_segments[start_index])

    return final_curves


# ... (Previous code for contour extraction and spline fitting)

image_path = "curves.png"
result_image, curve_segments = extract_curves_from_contours(image_path)

grouped_curves = group_curves(curve_segments, threshold=20)  # Adjust threshold

drawing = np.zeros_like(result_image)
for curve in grouped_curves:
    curve = curve.reshape((-1, 1, 2))
    cv2.polylines(drawing, [curve], False, (0, 255, 0), 2)

cv2.imshow("Grouped Curves", drawing)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"Number of grouped curves: {len(grouped_curves)}")
