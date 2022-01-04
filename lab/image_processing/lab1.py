import cv2 as cv
import numpy as np


# Draw a plot in a given image drawing context
# @param[in, out] image image drawing context
# @param[in] data_array data to draw
# @param[in] color color to use when drawing
# @param[in] max_val scale factor for the histogram values (default is 1)
def draw_graph(img, data_array, color, max_val=1.0):
    image_w = img.shape[1]
    image_h = img.shape[0]
    data_size = data_array.shape[0]

    step = image_w / data_size
    x = step * 0.5
    cv.line(img,
            (0, image_h - 1 - int((image_h - 1) * data_array[0] / max_val)),
            (int(x), image_h - 1 - int((image_h - 1) * data_array[0] / max_val)),
            color, thickness=1)

    for i in range(1, data_size):
        cv.line(img,
                (int(x), image_h - 1 - int((image_h - 1) * data_array[i - 1] / max_val)),
                (int(x + step), image_h - 1 - int((image_h - 1) * data_array[i] / max_val)),
                color, thickness=1)
        x += step

    cv.line(img,
            (int(x), image_h - 1 - int((image_h - 1) * data_array[data_size - 1] / max_val)),
            (image_w - 1, image_h - 1 - int((image_h - 1) * data_array[data_size - 1] / max_val)),
            color, thickness=1)


# Draw a histogram in a given image drawing context
# @param[in, out] image image drawing context
# @param[in] data_array data to draw
# @param[in] color color to use when drawing
# @param[in] max_val scale factor for the histogram values (default is 1)
def draw_hist(img, data_array, color, max_val=1.0):
    image_w = img.shape[1]
    image_h = img.shape[0]
    data_size = data_array.shape[0]

    step = image_w / data_size
    x = 0
    for i in range(0, data_size):
        cv.rectangle(img,
                     (int(x), image_h - 1 - int((image_h - 1) * data_array[i] / max_val)),
                     (int(x + step) - 1, image_h - 1),
                     color, thickness=-1)
        x += step


# Check whether an input file is empty
# @param[in] img file to check
def check_none(img):
    if img is None:
        print("Error reading file \"{}\"", img)
        exit()


filename = "/Users/stone/Downloads/ISProjectData/3.jpg"
lena_color = "/Users/stone/PycharmProjects/image_processing/lab1_python/xh.jpg"
low_dynamic = "/Users/stone/PycharmProjects/image_processing/lab1_python/low_dynamic.png"

"""
# RGB histogram
# Read image from file

image = cv.imread(lena_color, cv.IMREAD_COLOR)
check_none(image)

# Show image

cv.imshow("original", image)

# Split image into layers

bgr_planes = cv.split(image)

# Calculate histogram

hist_b = cv.calcHist(bgr_planes, [0], None, [256], [0, 256])
hist_g = cv.calcHist(bgr_planes, [1], None, [256], [0, 256])
hist_r = cv.calcHist(bgr_planes, [2], None, [256], [0, 256])


# Show histogram

hist_img = np.full((256, 512, 3), 255, dtype=np.uint8)
hist_scale = np.max([hist_b.max(), hist_g.max(), hist_r.max()])
draw_graph(hist_img, hist_b, (255, 0, 0), hist_scale)
draw_graph(hist_img, hist_g, (0, 255, 0), hist_scale)
draw_graph(hist_img, hist_r, (0, 0, 255), hist_scale)
cv.imshow("Histogram RGB", hist_img)

"""

# """
# Gray histogram
# Read file

image = cv.imread(lena_color, cv.IMREAD_GRAYSCALE)
check_none(image)

# Show original image
cv.imshow("Gray", image)

# Calculate histogram
hist = cv.calcHist([image], [0], None, [256], [0, 256])

# Calculate Cumulative histogram
cum_hist = np.cumsum(hist) / image.shape[0] / image.shape[1]

# Show histogram
hist_img = np.full((256, 512, 3), 255, dtype=np.uint8)
draw_hist(hist_img, hist, (127, 127, 127), hist.max())
draw_graph(hist_img, cum_hist, (0, 0, 255), 1)
cv.imshow("Gray histogram", hist_img)
# """

# """
# Calculate min and max values of an image
i_min = image.min()
i_max = image.max()
# """

"""
# Linear equalization
lut = (255 * cum_hist).clip(0, 255).astype(np.uint8)
"""

# arithmetic
# lut = np.cumsum(hist + 50)

# Nonlinear dynamic range stretching
# alpha = 0.5
# lut = np.arange(256, dtype=np.float32)
# lut = np.power(((lut - i_min) / (i_max - i_min)).clip(0, 1), alpha)
# lut = (255 * lut).astype(np.uint8)

# Uniform transformation
# lut = (i_max - i_min) * cum_hist + i_min
# lut = lut.clip(0, 255).astype(np.uint8)

# Exponential transformation
# alpha = 8
# lut = i_min / 255.0 - 1 / alpha * np.log(1 - cum_hist)
# lut = (255 * lut).clip(0, 255).astype(np.uint8)

# Rayleigh transformation
# Low dynamic area -> not bad
# alpha = 0.3
# lut = i_min / 255 + 1 - np.sqrt(-2 * alpha * alpha / np.log(1 - cum_hist))
# lut = (lut * 255).clip(0, 255).astype(np.uint8)

# 2/3 degree transformation
# Low dynamic area -> good
# lut = np.power(cum_hist, 2.0 / 3)
# lut = (lut * 255).clip(0, 255).astype(np.uint8)

# Hyperbolicun transformation
# alpha = 0.04
# alpha = i_min / 255.0
# lut = np.power(alpha, cum_hist)
# lut = (lut * 255).clip(0, 255).astype(np.uint8)
#
# # Apply LUT -> get a high contrast image
# image_lut = cv.LUT(image, lut)
# cv.imshow("Gray image lut", image_lut)
#
# # Calculate histogram after LUT
# hist_lut = cv.calcHist([image_lut], [0], None, [256], [0, 256])
#
# # Calculate cumulative histogram after LUT
# cum_hist_lut = np.cumsum(hist_lut) / image.shape[0] / image.shape[1]
#
# # Show histogram and cumulative histogram after LUT
# hist_img_lut = np.full((256, 512, 3), 255, dtype=np.uint8)
# draw_hist(hist_img_lut, hist_lut, (127, 127, 127), hist_lut.max())
# draw_graph(hist_img_lut, cum_hist_lut, (0, 0, 255), 1)
# cv.imshow("Histogram after lut", hist_img_lut)
# # """


# Do OpenCV equalizeHist
"""
image_eq = cv.equalizeHist(image)
cv.imshow("Image after equalizeHist", image_eq)

# Calculate histogram after equalizeHist
hist_eq = cv.calcHist([image_eq], [0], None, [256], [0, 256])

# Calculate cumulative histogram after equalizeHist
cum_hist_eq = np.cumsum(hist_eq) / image.shape[0] / image.shape[1]

# Show histogram after equalizeHist
hist_eq_img = np.full((256, 512, 3), 255, dtype=np.uint8)
draw_hist(hist_eq_img, hist_eq, (127, 127, 127), hist_eq.max())
draw_graph(hist_eq_img, cum_hist_eq, (0, 0, 0), 1)
cv.imshow("Histogram after equalizeHist", hist_eq_img)

# """

# Do OpenCV createCLAHE
"""
clahe = cv.createCLAHE()
clahe.setClipLimit(4)
image_clahe = clahe.apply(image)

cv.imshow("Image after CLAHE", image_clahe)

# Calculate histogram after createCLAHE
hist_clahe = cv.calcHist([image_clahe], [0], None, [256], [0, 256])

# Calculate cumulative histogram after createCLAHE
cum_hist_clahe = np.cumsum(hist_clahe) / image.shape[0] / image.shape[1]

# Show histogram after createCLAHE
hist_clahe_img = np.full((256, 512, 3), 255, dtype=np.uint8)
draw_hist(hist_clahe_img, hist_clahe, (127, 127, 127), hist_clahe.max())
draw_graph(hist_clahe_img, cum_hist_clahe, (0, 0, 0), 1)
cv.imshow("Histogram after CLAHE", hist_clahe_img)
"""

"""
#
# Profile
#
cv.destroyAllWindows()

# Read an image from file
filename = "/Users/stone/PycharmProjects/image_processing/lab1_python/barcode_zhubi.jpg"
image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
check_none(image)

# Show original image
cv.imshow("Image", image)

# Get profile
if image.ndim == 2:
    profile = image[round(image.shape[0] / 2), :]
else:
    profile = image[round(image.shape[0] / 2), :, :]

# Create profile graph image
profile_img = np.full((256, image.shape[1], 3), 255, dtype=np.uint8)
if image.ndim == 2:
    draw_graph(profile_img, profile, (0, 0, 0), profile.max())
else:
    draw_graph(profile_img, profile[:, 0], (255, 0, 0), profile.max())
    draw_graph(profile_img, profile[:, 1], (0, 255, 0), profile.max())
    draw_graph(profile_img, profile[:, 1], (0, 0, 255), profile.max())
cv.imshow("Profile", profile_img)
"""

# """
#
# Projection
# 
cv.destroyAllWindows()

# Read an image from file
filename = "/Users/stone/PycharmProjects/image_processing/lab1_python/text2.png"
img = cv.imread(filename)
check_none(img)

# Show image
cv.imshow("Text", img)

if img.ndim == 2:
    proj_x = np.sum(img, 0) / 255
    proj_y = np.sum(img, 1) / 255
else:
    proj_x = np.sum(img, (0, 2)) / 255 / img.shape[2]
    proj_y = np.sum(img, (1, 2)) / 255 / img.shape[2]

# Create X projection graph image
proj_x_img = np.full((256, img.shape[1], 3), 255, dtype=np.uint8)
draw_graph(proj_x_img, proj_x, (0, 0, 0), proj_x.max())

# Create Y projection graph image
proj_y_img = np.full((256, img.shape[0], 3), 255, dtype=np.uint8)
draw_graph(proj_y_img, proj_y, (0, 0, 0), proj_y.max())
proj_y_img = cv.transpose(proj_y_img)
proj_y_img = cv.flip(proj_y_img, 1)

# And show them
cv.imshow("Projection to Ox", proj_x_img)
cv.imshow("Projection to Oy", proj_y_img)

# """

# Wait for a key press
cv.waitKey(0)
cv.destroyAllWindows()
