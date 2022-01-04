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


# Check whether file is error
# @param[in] img file to check
# @param[in] filename filepath to chekc
def check_error(img, filename):
    if not isinstance(img, np.ndarray) or img.data is None:
        print("Error reading file \"{}\"", format(filename))
        exit()


# Wait for a key press
def end():
    cv.waitKey(0)
    cv.destroyAllWindows()


# Get matrix filled with ones
def get_ones(kernel_size, option):
    if option is "avg":
        print("get avg")
        return np.ones((kernel_size[0], kernel_size[1])) / kernel_size[0] / kernel_size[1]
    else:
        return np.ones(kernel_size[0], kernel_size[1])

"""
def imnoise(img, noise_type, param1=None, param2=None):
    if img.dtype is not np.uint8 and img.dtype is not np.float32:
        print("Unsupported image values type.")
        return None

    # Salt and pepper (param1 - probability)
    if noise_type is "salt & pepper":
        if param1 is not None:
            d = param1
        else:
            d = 0.5
        if param2 is not None:
            s_vs_p = param2
        else:
            s_vs_p = 0.5
        out = np.copy(img)
        rng = np.random.default_rng()

        vals = rng.random(out.shape)

        # Salt
        if out.dtype is np.uint8:
            out[vals < d * s_vs_p] = 255
        else:
            out[vals < d * s_vs_p] = 1.0

        # Pepper
        out[np.logical_and(vals >= d * s_vs_p, vals < d)] = 0

        return out

    # Multiplicative Noise (param1 - variance)
    if noise_type is "speckle": # Variance of multiplicative noise, specified as a numeric scalar
        if param1 is not None:
            var = param1
        else:
            var = 0.05

        rng = np.random.default_rng()
        gauss = rng.normal(0, var ** 0.5, img.shape)

        if img.dtype is np.uint8:
            f_img = img.astype(np.float32)
            out = (f_img + )
"""


# Implement bwareaopen function
# @param[in] img: single-channel binary image, dtype is uint8
# @param[in] size: deleting area size
def bwareaopen(img, size):
    output = img.copy()
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(img)
    for i in range(1, nlabels - 1):
        regions_size = stats[i, 4]
        if regions_size < size:
            x0 = stats[i, 0]
            y0 = stats[i, 1]
            x1 = stats[i, 0] + stats[i, 2]
            y1 = stats[i, 1] + stats[i, 3]
            for row in range(y0, y1):
                for col in range(x0, x1):
                    if labels[row, col] == i:
                        output[row, col] = 0
    return output
