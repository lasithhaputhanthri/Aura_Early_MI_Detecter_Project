import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian


# File path
MAT_FILE_PATH = "data/raw/Mask_ES0001 _4CH_1.mat"

# Load the .mat file
mat = sio.loadmat(MAT_FILE_PATH)
key = next(k for k in mat.keys() if not k.startswith("__") and isinstance(mat[k], np.ndarray))



# Get the first frame (2D binary image)
frame = mat[key][0]
image = frame.astype(float)  # active_contour expects float



# Initial circular contour in the middle of the image
s = np.linspace(0, 2 * np.pi, 500)
x_center, y_center = 115, 110
a, b = 50, 90  # horizontal (x) and vertical (y) radii for ellipse

x = x_center + a * np.cos(s)
y = y_center + b * np.sin(s)
init = np.array([y, x]).T  # active_contour expects (row, col)

smoothed = gaussian(image, sigma=3)

# Apply active contour (remove `coordinates` if scikit-image < 0.19)
snake = active_contour(
    smoothed,
    init,
    alpha=0.01,
    beta=0.1,
    gamma=0.01,
    max_num_iter=500
)

# Plotting the result
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.plot(init[:, 1], init[:, 0], '--r', label='Initial contour')
ax.plot(snake[:, 1], snake[:, 0], '-b', label='Active contour')
ax.set_title("Active Contour on Mask Frame")
ax.legend()
plt.show()
