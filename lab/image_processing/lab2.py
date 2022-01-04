import math

import cv2 as cv
import utils
import numpy as np

fn = "/Users/stone/PycharmProjects/image_processing/lab1_python/top_4.jpg"

# Read an image from file
img = cv.imread(fn, cv.IMREAD_COLOR)
utils.check_none(img)

# Show source image
cv.imshow("Source", img)

#
# Shift
#
"""
rows, cols = img.shape[0:2]
shift_x, shift_y = (50, 100)
T = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
img_shift = cv.warpAffine(img, T, (cols, rows))
cv.imshow("Shift", img_shift)
"""

#
# Flip
#
"""
rows, cols = img.shape[0:2]
# Flip around Ox
T = np.float32([[1, 0, 0], [0, -1, rows - 1]])
img_flip_x = cv.warpAffine(img, T, (cols, rows))
# img_flip_x = cv.flip(img, 0)
cv.imshow("Flip around Ox", img_flip_x)

# Flip around Oy
T = np.float32([[-1, 0, cols - 1], [0, 1, 0]])
img_flip_y = cv.warpAffine(img, T, (cols, rows))
# img_flip_y = cv.flip(img, 1)
cv.imshow("Flip around Oy", img_flip_y)
"""

#
# Scale
#
"""
rows, cols = img.shape[0:2]
scale_x, scale_y = (1.5, 0.5)
T = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
img_scale = cv.warpAffine(img, T, (int(cols * scale_x), int(rows * scale_y)))
# img_scale = cv.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_CUBIC)
cv.imshow("Scale", img_scale)
"""

#
# Rotation
#
"""
rows, cols = img.shape[0:2]
phi = 17.0
phi_rad = phi * math.pi / 180
# T = np.float32([[math.cos(phi_rad), -math.sin(phi_rad), 0], [math.sin(phi_rad), math.cos(phi_rad), 0]])
# T1 = np.float32([[1, 0, -(cols - 1) / 2.0], [0, 1, -(rows - 1) / 2.0], [0, 0, 1]])
# T2 = np.float32([[math.cos(phi_rad), -math.sin(phi_rad), 0], [math.sin(phi_rad), math.cos(phi_rad), 0], [0, 0, 1]])
# T3 = np.float32([[1, 0, (cols - 1) / 2.0], [0, 1, (rows - 1) / 2.0], [0, 0, 1]])
# T = np.matmul(T3, np.matmul(T2, T1))[0:2, :]
T = cv.getRotationMatrix2D(
    ((cols - 1) / 2.0, (rows - 1) / 2.0), -phi, 1
)
img_rotate = cv.warpAffine(img, T, (cols, rows))
cv.imshow("Rotate", img_rotate)
"""

#
# Affine mapping
#
"""
rows, cols = img.shape[0:2]
pts_src = np.float32([[50, 300], [150, 200], [50, 50]])
pts_dst = np.float32([[50, 200], [250, 200], [50, 100]])
T = cv.getAffineTransform(pts_src, pts_dst)
img_affine = cv.warpAffine(img, T, (cols, rows))

# Draw triangle on source and transformed images
cv.polylines(img, [pts_src.astype(np.int32)], True, (0, 0, 0), 1)
cv.polylines(img_affine, [pts_dst.astype(np.int32)], True, (0, 0, 0), 1)

# Show source and transformed images
cv.imshow("Source", img)
cv.imshow("Transform", img_affine)
"""

#
# Image Bevel
#
"""
rows, cols = img.shape[0:2]
s = 0.8
T = np.float32([[1, s, 0], (0, 1, 0)])
img_bevel = cv.warpAffine(img, T, (cols, rows))
cv.imshow("Bevel", img_bevel)
"""

#
# Piecewise-Linear mapping
#
"""
stretch = 2
T = np.float32([[stretch, 0, 0], [0, 1, 0]])
rows, cols = img.shape[0:2]

# img_left = img[:, 0:int(cols / 2), :]
# img_right = img[:, int(cols / 2):, :]
# img_right = cv.warpAffine(img_right, T, (img_right.shape[1]), rows)
# img_plm = np.concatenate((img_left, img_right), axis=1)

img_plm = img.copy()
img_plm[:, int(cols / 2):, :] = cv.warpAffine(img_plm[:, int(cols / 2):, :], T, (cols - int(cols / 2), rows))

# cv.imshow("Left", img_left)
# cv.imshow("Right", img_right)
cv.imshow("Piecewise-Linear", img_plm)
"""

#
# Projection mapping
#
"""
rows, cols = img.shape[0:2]
pts_src = np.float32([[50, 461], [461, 461], [461, 50], [50, 50]])
pts_dst = np.float32([[50, 461], [461, 440], [450, 10], [100, 50]])
# T = cv.getPerspectiveTransform(pts_src, pts_dst)
T = np.float32([[1.1, 0.2, 0.00075], [0.35, 1.1, 0.0005], [0, 0, 1]])
img_persp = cv.warpPerspective(img, T, (cols * 2, rows * 2))

# Draw rectangles on source and transformed images
cv.polylines(img, [pts_src.astype(np.int32)], True, (0, 0, 0), 1)
for i in range(4):
    src = np.float32([pts_src[i, 0], pts_src[i, 1], 1])
    pts_dst[i] = np.matmul(T[0:2, :], src) / np.matmul(T[2, :], src)
cv.polylines(img_persp, [pts_dst.astype(np.int32)], True, (0, 0, 0), 1)

# Show source and transformed images
cv.imshow("Source", img)
cv.imshow("Projection", img_persp)
"""

#
# Polynomial mapping
#
"""
# rows, cols = img.shape[0:2]
T = np.array([[0, 0], [1, 0], [0, 1], [0.00001, 0], [0.002, 0], [0.001, 0]])

# Create destination image
img_poly = np.zeros(img.shape, img.dtype)
"""

#
# Using numpy reindexing
#
"""
# Create mesh grid for X, Y coordinates
x, y = np.meshgrid(np.arange(cols), np.arange(rows))

# Calculate all new X, Y coordinates
x_new = np.round(T[0, 0] + x * T[1, 0] + y * T[2, 0] + x * x * T[3, 0] + x * y * T[4, 0] + y * y * T[5, 0]).astype(np.float32)
y_new = np.round(T[0, 1] + x * T[1, 1] + y * T[2, 1] + x * x * T[3, 1] + x * y * T[4, 1] + y * y * T[5, 1]).astype(np.float32)

# Create a mask where new coordinates are valid
mask = np.logical_and(np.logical_and(x_new >= 0, x_new < cols), np.logical_and(y_new >= 0, y_new < cols))

# Apply new coordinates
if img.ndim == 2:
    img_poly[y_new[mask].astype(int), x_new[mask].astype(int)] = img[y[mask], x[mask]]
else:
    img_poly[y_new[mask].astype(int), x_new[mask].astype(int), :] = img[y[mask], x[mask], :]

cv.imshow("Polynomial", img_poly)
"""


#
# Sinusoidal distortion
#
"""
# Create mesh grid for X, Y coordinates
rows, cols = img.shape[0:2]
x, y = np.meshgrid(np.arange(cols), np.arange(rows))

x = x + 20 * np.sin(2 * math.pi * y / 90)

img_sin = cv.remap(img, x.astype(np.float32), y.astype(np.float32), cv.INTER_LINEAR)

cv.imshow("Sinusoidal", img_sin)
"""

#
# Barrel distortion
#
"""
# Create mesh grid for X, Y coordinates
rows, cols = img.shape[0:2]
x, y = np.meshgrid(np.arange(cols), np.arange(rows))

# Shift and normalize grid
x_mid = cols / 2.0
y_mid = rows / 2.0
x = x - x_mid
y = y - y_mid

# Convert to polar and do transformation
r, theta = cv.cartToPolar(x / x_mid, y / y_mid)
F3 = 0.1
# F3 = -0.003
F5 = 0.12
r = r + F3 * r ** 3 + F5 * r ** 5
# r = r + F3 * r ** 2
# F1 = 0.0

# Undo conversion, normalization and shift
u, v = cv.polarToCart(r, theta)
u = u * x_mid + x_mid
v = v * y_mid + y_mid

img_dist = cv.remap(img, u.astype(np.float32), v.astype(np.float32), cv.INTER_LINEAR)

cv.imshow("Distorted", img_dist)
"""

#
# Correction
#
"""

# Calculate the background transformation coefficients
d = 5  # Maximum degree
max_r = 1.3  # Maximum value for the pts array
pts = (np.arange(d, dtype=np.float32) + 1) / d * max_r
pts2 = pts + F3 * pts ** 3 + F5 * pts ** 5

# Create and fill matrix
T = np.zeros((d, d))
for i in range(d):
    T[:, i] = pts2 ** (i + 1)

# Invert it and calculate coefficients
T = np.linalg.inv(T)
F = np.matmul(T, pts)

# Do transformation in polar CS
r, theta = cv.cartToPolar(x / x_mid, y / y_mid)
rt = 0
for i in range(d):
    rt = rt + F[i] * r ** (i + 1)
r = rt

u, v = cv.polarToCart(r, theta)
u = u * x_mid + x_mid
v = v * y_mid + y_mid

img_corr = cv.remap(img_dist, u.astype(np.float32), v.astype(np.float32), cv.INTER_LINEAR)

cv.imshow("Correct", img_corr)
"""


#
# Stitching
#
"""

# Manual
fn_top = "/Users/stone/PycharmProjects/image_processing/lab1_python/top_4.jpg"
fn_bot = "/Users/stone/PycharmProjects/image_processing/lab1_python/bot_4.jpg"

# Read image from files
img_top = cv.imread(fn_top, cv.IMREAD_COLOR)
utils.check_error(img_top, fn_top)
img_bot = cv.imread(fn_bot, cv.IMREAD_COLOR)
utils.check_error(img_bot, fn_bot)

# Show parts of image
cv.destroyAllWindows()
cv.imshow("Top", img_top)
cv.imshow("Bottom", img_bot)

# Match template
templ_size = 10
templ = img_top[-templ_size:, :, :]
res = cv.matchTemplate(img_bot, templ, cv.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

# Stitch images
# change to :1003 since the width has been changed by accident while splitting images
img_stitch = np.zeros((img_top.shape[0] + img_bot.shape[0] - max_loc[1] - templ_size, img_top.shape[1], img_top.shape[2]), dtype=np.uint8)
img_stitch[0:img_top.shape[0], :, :] = img_top
img_stitch[img_top.shape[0]:, :, :] = img_bot[max_loc[1] + templ_size:, :1003, :]

# Show stitched image
cv.imshow("Stitch", img_stitch)
"""

#
# Stitcher
#
"""
fn_top = "/Users/stone/PycharmProjects/image_processing/lab1_python/top_4.jpg"
fn_bot = "/Users/stone/PycharmProjects/image_processing/lab1_python/bot_4.jpg"

# Read image from files
img_top = cv.imread(fn_top, cv.IMREAD_COLOR)
utils.check_error(img_top, fn_top)
img_bot = cv.imread(fn_bot, cv.IMREAD_COLOR)
utils.check_error(img_bot, fn_bot)

# Show parts of image
cv.destroyAllWindows()
cv.imshow("Top", img_top)
cv.imshow("Bottom", img_bot)

# Stitch
stitcher = cv.Stitcher.create(cv.Stitcher_SCANS)
status, img_stitch = stitcher.stitch([img_top, img_bot])

# Show stitched image in case of success
if status is cv.Stitcher_OK:
    cv.imshow("Stitch", img_stitch)
else:
    print(status is cv.Stitcher_OK)
"""


utils.end()
