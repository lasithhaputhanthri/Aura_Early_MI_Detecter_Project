import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# Load image in grayscale
image_path = "experiments/images/sample_echo.jpg"
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Threshold to isolate bright regions (heart walls)
_, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

# Step 2: Extract contours
contours = measure.find_contours(binary, level=0.8)
if not contours:
    raise RuntimeError("No contours found.")
contour = max(contours, key=len)  # pick largest

# Step 3: Separate top and bottom half for polynomial fitting
ys, xs = contour[:, 0], contour[:, 1]
mid_y = np.mean(ys)
top_indices = ys < mid_y
bottom_indices = ys >= mid_y

# Fit polynomials
top_poly = np.polyfit(xs[top_indices], ys[top_indices], deg=3)
bottom_poly = np.polyfit(xs[bottom_indices], ys[bottom_indices], deg=3)

# Step 4: Generate smooth fitted curves
x_vals = np.linspace(np.min(xs), np.max(xs), 500)
top_fit = np.polyval(top_poly, x_vals)
bottom_fit = np.polyval(bottom_poly, x_vals)

# Step 5: Plot everything
plt.figure(figsize=(8, 6))
plt.imshow(binary, cmap="gray")
plt.plot(xs, ys, '.', color="blue", alpha=0.4, label="Contour points")
plt.plot(x_vals, top_fit, 'r-', linewidth=2, label="Top Polynomial Fit")
plt.plot(x_vals, bottom_fit, 'g-', linewidth=2, label="Bottom Polynomial Fit")
plt.legend()
plt.title("Active Polynomials – Boundary Fitting")
plt.axis("off")
plt.show()
