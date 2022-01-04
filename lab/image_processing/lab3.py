import cv2 as cv
import numpy
import numpy as np
import skimage
from scipy.signal import wiener
import matplotlib as mat

import utils
import scipy

filename = "/Users/stone/PycharmProjects/image_processing/lab1_python/top_4.jpg"

# Read an image from file
# img = cv.imread(filename, cv.IMREAD_COLOR)
img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
utils.check_none(img)

# Show source image
cv.imshow("Source", img)

# Four noises models
# """
img_noisy_sp = skimage.util.random_noise(img, "s&p", amount=0.01, salt_vs_pepper=0.5)
salt_pepper = (255 * img_noisy_sp).astype(np.uint8)
# cv.imshow("Salt & pepper", img_noisy)
img_noisy_gaussian = skimage.util.random_noise(img, "gaussian", clip=True, mean=0, var=0.01)
gaussian = (255 * img_noisy_gaussian).astype(np.uint8)
# cv.imshow("Gaussian", img_noisy)
img_noisy_poisson = skimage.util.random_noise(img, "poisson", clip=True)
poisson = (255 * img_noisy_poisson).astype(np.uint8)
# cv.imshow("Poisson", img_noisy)
img_noisy_speckle = skimage.util.random_noise(img, "speckle", clip=True, mean=0, var=0.01)
speckle = (255 * img_noisy_speckle).astype(np.uint8)
# cv.imshow("Speckle", img_noisy)
# cv.imshow("Salt_pepper      Gaussian      Poisson      Speckle", np.hstack([salt_pepper, gaussian, poisson, speckle]))

noises = np.vstack([img_noisy_sp, img_noisy_gaussian, img_noisy_poisson, img_noisy_speckle])
noise_names = ["Salt_pepper", "Gaussian", "Poisson", "Speckle"]
print("OK")
# """

# img_noisy = skimage.util.random_noise(img, "s&p", amount=0.01, salt_vs_pepper=1)
#
# img_noisy = (255 * img_noisy).astype(np.uint8)
#
# cv.imshow("Noisy", img_noisy)

# Kernel for low-pass filter
# fspecial('average', 3)
#       / 1 1 1 \
# 1/9   | 1 1 1 |
#       \ 1 1 1 /
#
"""
kernel_size = (3, 3)
kernel = np.ones((kernel_size[0], kernel_size[1]))
kernel = kernel / kernel_size[0] / kernel_size[1]

#       / 1 1 1 \
# 1/10   | 1 1 1 |
#       \ 1 1 1 /
#
# kernel_size = (3, 3)
# kernel = np.ones((kernel_size[0], kernel_size[1]))
# kernel[int(kernel_size[0] / 2), int(kernel_size[1] / 2)] = 2
# kernel = kernel / (kernel_size[0] * kernel_size[1] + 1)

# Run filter
# ddepth -1 means to use same image depth after filtering
img_filter = cv.filter2D(img_noisy, -1, kernel)
cv.imshow("Filter2d", img_filter)
"""

# Blur
# img_filter = cv.blur(img_noisy, (3, 3))
# cv.imshow("Filter2d", img_filter)


"""
img_noisy = skimage.util.random_noise(img, "s&p", amount=0.01, salt_vs_pepper=1)

# img_noisy = (255 * img_noisy).astype(np.uint8)
# 
# cv.imshow("Noisy", img_noisy)
"""

# Gaussian Blur
"""
rows, cols = img_noisy_sp.shape[0:2]
for index in range(int(noises.shape[0] / cols)):
    img_noisy = noises[rows * index: rows * (index + 1) - 1, :]
    img_filter = cv.GaussianBlur(img_noisy, ksize=(0, 0), sigmaX=1, borderType=cv.BORDER_REPLICATE)
    cv.imshow("Gaussian filter on " + noise_names[index], img_filter)

# img_filter = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=1, borderType=cv.BORDER_REPLICATE)
# cv.imshow("Gaussian filter", img_filter)
"""

#
# Counter-harmonic mean filter teacher
#
"""
rows, cols = img_noisy_sp.shape[0:2]
for index in range(int(noises.shape[0] / cols)):
    print(index)
    img_noisy = noises[rows * index: rows * (index + 1) - 1, :]

    # Filter parameters
    Q = -5
    kernel_size = (5, 5)
    kernel = np.ones((kernel_size[0], kernel_size[1]))

    # Convert to float and make image with border
    if img_noisy.dtype == np.uint8:
        img_copy = img_noisy.astype(np.float32) / 255
    else:
        img_copy = img_noisy
    img_copy = cv.copyMakeBorder(img_copy, int((kernel_size[0] - 1) / 2), int(kernel_size[0] / 2),
                                 int((kernel_size[1] - 1) / 2), int(kernel_size[1] / 2), cv.BORDER_REPLICATE)

    # Calculate temporary matrices for I ** Q and I ** (Q + 1)
    img_copy_power = np.power(img_copy, Q)
    img_copy_power_1 = np.power(img_copy, Q + 1)

    img_filter_top = np.zeros(img_noisy.shape, np.float32)
    img_filter_bottom = np.zeros(img_noisy.shape, np.float32)

    # Do filter
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            img_filter_top = img_filter_top + img_copy_power_1[i:i + rows - 1, j:j + cols]
            img_filter_bottom = img_filter_bottom + img_copy_power[i:i + rows - 1, j:j + cols]
    img_filter = img_filter_top / img_filter_bottom

    # Convert back to uint if needed
    if img_noisy.dtype == np.uint8:
        img_filter = (255 * img_filter).clip(0, 255).astype(np.uint8)

    cv.imshow("Counterharmonic mean filter with " + noise_names[index], img_filter)
"""

#
# Counter-harmonic mean filter
#
"""
rows, cols = img_noisy.shape[0:2]

# Filter parameters
Q = -2
kernel_size = (5, 5)
kernel = np.ones((kernel_size[0], kernel_size[1]), dtype=np.float32)
# kernel = kernel / kernel_size[0] / kernel_size[1]

# Convert to float and make image with border
if img_noisy.dtype is np.uint8:
    img_copy = img_noisy.astype(np.float32) / 255
else:
    img_copy = img_noisy
img_copy = cv.copyMakeBorder(img_copy, int((kernel_size[0] - 1) / 2), int(kernel_size[0] / 2),
                             int((kernel_size[1] - 1) / 2), int(kernel_size[1] / 2), cv.BORDER_REPLICATE)

# Calculate temporary matrices for I ** Q and I ** (Q + 1)
img_copy_power = np.power(img_copy, Q)
img_copy_power_1 = np.power(img_copy, Q + 1)

img_filter_top = np.zeros(img_noisy.shape, np.float32)
img_filter_bottom = np.zeros(img_noisy.shape, np.float32)

# Do filter
for i in range(kernel_size[0]):
    for j in range(kernel_size[1]):
        img_filter_top = img_filter_top + img_copy_power_1[i:i + rows, j:j + cols]
        img_filter_bottom = img_filter_bottom + img_copy_power[i:i + rows, j:j + cols]
        # img_filter_top = img_filter_top + kernel[i, j] * img_copy_power_1[i:i + rows, j:j + cols]
        # img_filter_bottom = img_filter_bottom + kernel[i, j] * img_copy_power[i:i + rows, j:j + cols]
img_filter = img_filter_top / img_filter_bottom

# Convert back to uint if needed
if img_noisy.dtype is np.uint8:
    img_filter = (255 * img_filter).clip(0, 255).astype(np.uint8)

cv.imshow("My Counterharmonic mean filter", img_filter)
# """

#
# Median filter
#
"""
rows, cols = img_noisy_sp.shape[0: 2]
for index in range(int(noises.shape[0] / rows)):
    print(index)
    img_noisy = noises[rows * index: rows * (index + 1), :]
    img_median_1 = cv.medianBlur(img, 3)
    # if img_noisy.dtype is np.float64:
    img_noisy = (img_noisy * 255).astype(np.uint8)
    img_median = cv.medianBlur(img_noisy, 3)
    cv.imshow("Median filter on " + noise_names[index], img_median)
# img_median = cv.medianBlur(img, 3)
# cv.imshow("Median filter", img_median)
"""

#
# Adaptive median filter
#
"""
rows, cols = img_noisy_sp.shape[0:2]
for index in range(int(noises.shape[0] / rows)):
    print(index)
    img_noisy = noises[rows * index: rows * (index + 1), :]

    weights = img_noisy.ndim is 2

    # Set initial and maximum kernel sizes
    kernel_size = (1, 1)
    max_kernel_size = 7

    # Convert to float and make image with border
    if img_noisy.dtype is np.uint8:
        img_copy_nb = img_noisy.astype(np.float32) / 255
    else:
        img_copy_nb = img_noisy
    img_copy = cv.copyMakeBorder(img_copy_nb, int((max_kernel_size - 1) / 2), int(max_kernel_size / 2),
                                 int((max_kernel_size - 1) / 2), int(max_kernel_size / 2), cv.BORDER_REPLICATE)
    img_filter = np.zeros_like(img_copy_nb)

    # Mask of pixels to filter
    mask = np.ones(img_noisy.shape, dtype=int)

    # While there are not filtered pixels
    while mask.any() and kernel_size[0] < max_kernel_size:
        # Increase kernel size and create kernel:
        kernel_size = (kernel_size[0] + 2, kernel_size[1] + 2)
        kernel_shift = (int((max_kernel_size - kernel_size[0]) / 2), int((max_kernel_size - kernel_size[1]) / 2))
        kernel = np.ones((kernel_size[0], kernel_size[1]), dtype=np.float32)

        # Fill kernels for all pixels
        img_layers = np.zeros(img_noisy.shape + (kernel_size[0] * kernel_size[1], ), dtype=np.float32)
        if weights:
            for i in range(kernel_size[0]):
                for j in range(kernel_size[1]):
                    img_layers[:, :, i * kernel_size[1] + j] = kernel[i, j] * img_copy[i + kernel_shift[0]:i + kernel_shift[0] + rows, j + kernel_shift[1]:j + kernel_shift[1] + cols]
        else:
            for i in range(kernel_size[0]):
                for j in range(kernel_size[1]):
                    img_layers[:, :, :, i * kernel_size[1] + j] = kernel[i, j] * img_copy[i + kernel_shift[0]:i + kernel_shift[0] + rows, j + kernel_shift[1]:j + kernel_shift[1] + cols, :]

        # Sort kernels
        img_layers.sort()

        # Calculate z_min, z_max, z_mean values
        if weights:
            z_min = img_layers[:, :, 0]
            z_mean = img_layers[:, :, int((kernel_size[0] * kernel_size[1] - 1) / 2)]
            z_max = img_layers[:, :, kernel_size[0] * kernel_size[1] - 1]
        else:
            z_min = img_layers[:, :, :, 0]
            z_mean = img_layers[:, :, :, int((kernel_size[0] * kernel_size[1] - 1) / 2)]
            z_max = img_layers[:, :, :, kernel_size[0] * kernel_size[1] - 1]

        # Calculate A and B conditions
        A_mask = np.logical_and(z_min < z_mean, z_mean < z_max)
        B_mask = np.logical_and(z_min < img_copy_nb, img_copy_nb < z_max)

        # Copy data that fulfills the condition
        img_filter[np.logical_and(mask, np.logical_and(A_mask, B_mask))] = \
            img_copy_nb[np.logical_and(mask, np.logical_and(A_mask, B_mask))]
        img_filter[np.logical_and(mask, np.logical_and(A_mask, np.logical_not(B_mask)))] = \
            z_mean[np.logical_and(mask, np.logical_and(A_mask, np.logical_not(B_mask)))]

        # Update masks with pixels that we copied
        mask = np.logical_and(mask, np.logical_not(A_mask))

    # Fill what left empty
    img_filter[mask] = img_copy_nb[mask]

    # Convert back to unit if needed
    if img_noisy.dtype is np.uint8:
        img_filter = (255 * img_filter).clip(0, 255).astype(np.uint8)

    cv.imshow("Adaptive mean filter on " + noise_names[index], img_filter)
"""

#
# Rank filter
#
"""
rows, cols = img_noisy_sp.shape[0:2]
for index in range(int(noises.shape[0] / rows)):
    print(index)
    img_noisy = noises[rows * index: rows * (index + 1), :]

    # Filter parameters
    kernel_size = (3, 3)
    rank = -2
    kernel = np.ones(kernel_size, dtype=np.float32)

    # Convert to float and make image with border
    if img_noisy.dtype is np.uint8:
        img_copy = img_noisy.astype(np.float32) / 255
    else:
        img_copy = img_noisy
    img_copy = cv.copyMakeBorder(img_copy, int((kernel_size[0] - 1) / 2), int(kernel_size[1] / 2), int((kernel_size[0] - 1) / 2), int(kernel_size[1] / 2), cv.BORDER_REPLICATE)

    # Fill arrays for each kernel item
    img_layers = np.zeros(img_noisy.shape + (kernel_size[0] * kernel_size[1], ), dtype=np.float32)
    if img_noisy.ndim is 2:
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                img_layers[:, :, i * kernel_size[1] + j] = kernel[i, j] * img_copy[i:i + rows, j:j + cols]
    else:
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                img_layers[:, :, :, i * kernel_size[1] + j] = kernel[i, j] * img_copy[i:i + rows, j:j + cols, :]

    # Sort
    img_layers.sort()

    # And select one with rank
    if img_noisy.ndim is 2:
        img_filter = img_layers[:, :, rank]
    else:
        img_filter = img_layers[:, :, :, rank]

    # Convert back to uint if needed
    if img_noisy.dtype is np.uint8:
        img_filter = (255 * img_filter).clip(0, 255).astype(np.uint8)

    cv.imshow("Rank filter on " + noise_names[index], img_filter)
"""

#
# Wiener filter
#
"""
rows, cols = img_noisy_sp.shape[0:2]
for index in range(int(noises.shape[0] / rows)):
    print(index)
    img_noisy = noises[rows * index: rows * (index + 1), :]

kernel_size = (7, 7)
kernel = np.ones(kernel_size[0], kernel_size[1])

# Convert to float and make image with border
if img_noisy.dtype is np.uint8:
    img_copy = img_noisy.astype(np.float32) / 255
else:
    img_copy = img_noisy
img_copy = cv.copyMakeBorder(img_copy, int((kernel_size[0] - 1) / 2), int(kernel_size[1] / 2), int((kernel_size[0] - 1) / 2), int(kernel_size[1] / 2), cv.BORDER_REPLICATE)

# Split into layers
bgr_planes = cv.split(img_copy)
bgr_planes_2 = []

kernel_power = np.power(kernel, 2)

# For all layers
for plane in bgr_planes:
    # Calculate temporary matrices for img ** 2
    plane_power = np.power(plane, 2)

    m = np.zeros(img_noisy.shape[0:2], np.float32)
    q = np.zeros(img_noisy.shape[0:2], np.float32)

    # Calculate variance values
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            m = m + kernel[i, j] * plane[i:i + rows, j: j + cols]
            q = q + kernel_power[i, j] * plane_power[i:i + rows, j:j + cols]

    m = m / np.sum(kernel)
    q = q / np.sum(kernel)
    q = q - m * m

    # Calculate noise as an average variance
    v = np.sum(q) / img_noisy.size

"""

#
# Wiener2 filter
#

# Scipy

"""
rows, cols = img_noisy_sp.shape[0: 2]
for index in range(int(noises.shape[0] / rows)):
    print(index)
    img_noisy = noises[rows * index: rows * (index + 1), :]
    # Convert to float32
    if img_noisy.dtype is np.uint8:
        img_filter = img_noisy.astype(np.float32) / 255
    else:
        img_filter = np.copy(img_noisy)

    img_filter = wiener(img_filter, (3, 3))

    # Convert back to uint if needed
    if img_noisy.dtype is np.uint8:
        img_filter = (255 * img_filter).clip(0, 255).astype(np.uint8)

    cv.imshow("Wiener2 filter Scipy on " + noise_names[index], img_filter)
"""

#
# Geometric mean filter
#
# fspecial('average', 5)
#       / 1 1 1 1 1 \
#       | 1 1 1 1 1 |
# 1/25  | 1 1 1 1 1 |
#       | 1 1 1 1 1 |
#       \ 1 1 1 1 1 /
"""
kernel_size = (5, 5)
kernel = utils.get_ones(kernel_size, "avg")

# Geometric mean is equal to exp(arithmetic mean(ln(img)))
if img_noisy.dtype == np.uint8:
    img_filter = np.power(np.exp(cv.filter2D(np.log(img_noisy).astype(np.float32), -1, kernel)), np.sum(kernel)). \
        clip(0, 255).astype(np.uint8)
else:
    img_filter = np.power(np.exp(cv.filter2D(np.log(img_noisy).astype(np.float32), -1, kernel)), np.sum(kernel))
cv.imshow("Geometric mean filter", img_filter)
"""

#
# Harmonic mean filter
#
"""
rows, cols = img_noisy.shape[0: 2]
# Filter params
kernel_size = (5, 5)
kernel = utils.get_ones(kernel_size, "avg")

# Convert to float and make image with boarder
if img_noisy.dtype is np.uint8:
    img_copy = img_noisy.astype(np.float32) / 255
else:
    img_copy = img_noisy
img_copy = cv.copyMakeBorder(img_copy, int((kernel_size[0] - 1) / 2), int(kernel_size[1] / 2), \
                             int((kernel_size[0] - 1) / 2), int(kernel_size[1] / 2), cv.BORDER_REPLICATE)

# Do filter
img_filter = np.zeros(img_noisy.shape, np.float32)
for i in range(kernel_size[0]):
    for j in range(kernel_size[1]):
        img_filter = img_filter + kernel[i, j] / img_copy[i:i + rows, j:j + cols]
img_filter = np.sum(kernel) / img_filter

# Convert back to uint if needed
if img_noisy.dtype is np.uint8:
    img_filter = (255 * img_filter).clip(0, 255).astype(np.uint8)

img_filter = img_filter.astype(np.float32)

print(img_filter.dtype)

cv.imshow("Harmonic mean filter", img_filter)
"""

#
# Laplace edge detection
#
# """
laplace = cv.Laplacian(img, cv.CV_64F)
laplace = numpy.uint8(numpy.absolute(laplace))

cv.imshow("Edge detection by Laplacaian", laplace)
# """

#
# Sobel edge detection
#
# """
sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0)
sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1)

sobel_x = numpy.uint8(numpy.absolute(sobel_x))
sobel_y = numpy.uint8(numpy.absolute(sobel_y))
sobel_combine = cv.bitwise_or(sobel_x,sobel_y)

# Display two images in a figure
cv.imshow("Edge detection by Sobel", sobel_combine)
# """

#
# Canny edge detection
#
# """
canny = cv.Canny(img, 30, 150)

canny = numpy.uint8(numpy.absolute(canny))

# Display two images in a figure
cv.imshow("Edge detection by Canny", canny)
# """

#
# Roberts edge detection
#
# """
kernel_x = np.array([[-1, 0], [0, 1]], dtype=int)
kernel_y = np.array([[0, -1], [1, 0]], dtype=int)

x = cv.filter2D(img, cv.CV_16S, kernel_x)
y = cv.filter2D(img, cv.CV_16S, kernel_y)

absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

cv.imshow("Roberts", Roberts)
# """

#
# Prewitt edge detection
#
# """
kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

x = cv.filter2D(img, cv.CV_16S, kernel_x)
y = cv.filter2D(img, cv.CV_16S, kernel_y)

absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
prewitt = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

cv.imshow("Prewitt", prewitt)
# """

utils.end()
