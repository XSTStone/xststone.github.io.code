import math

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


def abs_func(a, b):
    if a <= b:
        return b - a
    elif a > b:
        return a - b


# End with showing pic
def end(filename, img):
    cv.imshow(filename, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Read file to img
def read(img, filename, color):
    if color is "gray":
        img = cv.imshow(filename, cv.IMREAD_GRAYSCALE)
    if color is "color":
        img = cv.imshow(filename, cv.IMREAD_COLOR)
    return img


# Implement imfillholes
def imfillholes(img):
    if img.ndim != 2 or img.dtype != np.uint8:
        return None
    rows, cols = img.shape[0:2]
    mask = img.copy()
    for i in range(cols):
        if mask[0, i] == 0:
            cv.floodFill(mask, None, (i, 0), 255, 10, 10)
        if mask[rows - 1, i] == 0:
            cv.floodFill(mask, None, (i, rows - 1), 255, 10 ,10)

    for i in range(rows):
        if mask[i, 0] == 0:
            cv.floodFill(mask, None, (0, i), 255, 10, 10)
        if mask[i, cols - 1] == 0:
            cv.floodFill(mask, None, (cols - 1, i), 255, 10, 10)

    result = img.copy()
    result[mask == 0] = 255
    return result


# Formula 1.12 judge function
def skin_color_judge_1(blue, green, red):
    flag_red = red > 95
    flag_green = green > 40
    flag_blue = blue > 20
    flag_diff = (max(blue, green, red) - min(blue, green, red)) > 15
    flag_diff_red_green = abs_func(red, green) > 15
    flag_red_bigger_green = red > green
    flag_red_bigger_blue = red > blue
    if flag_red & flag_green & flag_blue & flag_diff & flag_red_bigger_blue & flag_diff_red_green & flag_red_bigger_green:
        return 1
    return 0


# Formula 1.13 judge function
def skin_color_judge_2(blue, green, red):
    flag_red = red > 220
    flag_green = green > 210
    flag_blue = blue > 170
    flag_diff = abs_func(red, green) <= 15
    flag_green_bigger_blue = green > blue
    flag_red_bigger_blue = red > blue
    if flag_red & flag_green & flag_blue & flag_diff & flag_red_bigger_blue & flag_green_bigger_blue:
        return True
    return False


# Formula 1.14 judge function
def skin_color_judge_3(blue, green, red):
    total = blue + green + red
    r = red / total
    g = green / total
    b = blue / total
    total_rgb = r + g + b
    flag_division = (r / g) > 1.185
    flag_rb = (r * b) / (total_rgb ** 2) > 0.107
    flag_rg = (r * g) / (total_rgb ** 2) > 0.112
    if flag_rg & flag_rb & flag_division:
        return 1
    return 0


def normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image-mean))

    image = (image - mean)/np.sqrt(var)

    return image


# Do weber function
def weber_func(brightness):
    if brightness <= 88:
        return 20 - 12 * brightness / 88
    elif 88 < brightness <= 138:
        return 0.002 * np.power(brightness - 88, 2)
    elif 138 < brightness <= 255:
        return 7 * (brightness - 138) / 117 + 13


# Mouse event handler function
def MouseHandler(event, x, y, flags, param):
    print("Enter handler function")
    if event is not cv.EVENT_LBUTTONDOWN:
        return

    if param is None:
        return
    img, img_lab, sampleAreas, radius = param

    sampleAreas.append((x, y))
    img_cp = img.copy()
    for pix in sampleAreas:
        cv.circle(img_cp, pix, radius, (0, 0, 255), 1)
    cv.imshow("Source", img_cp)

    color_marks = []
    color_marks_BGR = []
    for pix in sampleAreas:
        mask = np.zeros_like(img_lab[0])
        cv.circle(mask, pix, radius, 255, -1)
        a = img_lab[1].mean(where=mask > 0)
        b = img_lab[2].mean(where=mask > 0)
        color_marks.append((a, b))
        color_marks_BGR.append(img[mask > 0, :].mean(axis=(0)))

    distance = []
    for color in color_marks:
        distance.append(np.sqrt(np.power(img_lab[1] - color[0], 2) + np.power(img_lab[2] - color[1], 2)))
    distance_min = np.minimum.reduce(distance)

    labels = np.zeros_like(img_lab[0], dtype=np.uint8)
    img_plot = np.full((256, 256, 3), 255, dtype=np.uint8)

    for i in range(len(color_marks)):
        img_tmp = np.zeros_like(img)
        mask = distance_min == distance[i]
        img_tmp[mask] = img[mask]
        labels[mask] = i
        cv.imshow("Segmented area" + str(i), img_tmp)
        img_plot[img_lab[1][mask], img_lab[2][mask], :] = color_marks_BGR[i]

    cv.imshow("Color distribution", img_plot)


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
