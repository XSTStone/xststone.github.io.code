"""
binarization
double binarization (range binarization)
Otsu
adaptive

segmentation
    -> Weber principle
    RGB images by skin color
    -> CIE Lab color space
    k-means clustering
    -> texture segmentation

Task:
    binarization -> four methods
    segmentation -> Weber principle
    segmentation -> CIE lab
    segmentation -> texture segmentation
"""
import cv2
import cv2 as cv
import numpy as np
import skimage

import utils

# """
# Binarization
# filename = "/Users/stone/PycharmProjects/image_processing/lab1_python/top_4.jpg"

# Single binarization
# img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
# t = 127
# ret, img_new = cv.threshold(img, t, 255, cv.THRESH_BINARY)

# Double binarization
# img = cv.imread(filename, cv.IMREAD_COLOR)
# t1 = 127
# t2 = 200
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, img_new = cv.threshold(img_gray, t2, 255, cv.THRESH_TOZERO_INV)
# ret, img_new = cv.threshold(img_new, t1, 255, cv.THRESH_BINARY)

# Otsu binarization
# img = cv.imread(filename, cv.IMREAD_COLOR)
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, img_new = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)

# Adaptive binarization
# img = cv.imread(filename, cv.IMREAD_COLOR)
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img_new = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

# cv.imshow("Source", img)
# utils.end("new image", img_new)
# """

def weber_function(filename):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    if not isinstance(img, np.ndarray) or img.data == None:
        print("Error reading file \"{}\"".format(filename))
        exit()

    cv.imshow("Source", img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Grayscale", img_gray)
    img_weber = np.zeros_like(img_gray)
    img_weber2 = np.zeros_like(img_gray)
    n = 1

    while (img_weber == 0).any():
        img_min = img_gray[img_weber == 0].min()
        img_weight = utils.weber_func(img_min)
        n += 1
        mask = np.logical_and(img_gray >= img_min, img_gray <= img_min + img_weight)
        img_weber[mask] = n
        img_weber2[mask] = img_min

    n = n - 1
    img_weber = img_weber - 1

    cv.imshow("Weber segmentation JET",
              cv.applyColorMap((img_weber.astype(np.float32) * 255 / (n + 1)).astype(np.uint8), cv.COLORMAP_JET))
    cv.imshow("Weber segmentation", img_weber2)
    cv.waitKey(0)
    cv.destroyAllWindows()


"""
# Segmentation
filename = "/Users/stone/PycharmProjects/image_processing/lab1_python/top_4.jpg"

# Weber principle
img = cv.imread(filename, cv.IMREAD_COLOR)
if not isinstance(img, np.ndarray) or img.data is None:
    print("Error reading file \"{}\"".format(filename))
    exit()

cv.imshow("Source", img)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale", img_gray)
img_weber = np.zeros_like(img_gray)
img_weber2 = np.zeros_like(img_gray)
n = 1

while (img_weber == 0).any():
    img_min = img_gray[img_weber is 0].min()
    img_weight = utils.weber(img_min)
    n += 1
    mask = np.logical_and(img_gray >= img_min, img_gray <= img_min + img_weight)
    img_weber[mask] = n
    img_weber2[mask] = img_min

n = n - 1
img_weber = img_weber - 1

cv.imshow("Weber segmentation JET", cv.applyColorMap(img_weber.astype(np.float32) * 255 / (n + 1).astype(np.uint8), cv.COLORMAP_JET))
cv.imshow("Weber segmentation", img_weber2)
utils.end()

"""


def skin_color_segmentation_1(filename):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    cv.imshow("Source", img)
    rows, cols = img.shape[:2]
    blue, green, red = cv.split(img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray", img_gray)
    is_skin = np.zeros_like(img_gray)
    for row in range(rows):
        for col in range(cols):
            pixel_blue = blue[row, col]
            pixel_green = green[row, col]
            pixel_red = red[row, col]
            is_skin[row, col] = utils.skin_color_judge_1(pixel_blue, pixel_green, pixel_red)
    result = img_gray.copy()
    result[is_skin == 1] = 255
    result[is_skin == 0] = 0
    cv.imshow("Formula 1", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def skin_color_segmentation_2(filename):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    img_normal = utils.normalize(img)
    cv.imshow("Source", img)
    rows, cols = img_normal.shape[:2]
    blue, green, red = cv.split(img_normal)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    is_skin = np.zeros_like(img_gray)
    for row in range(rows):
        for col in range(cols):
            pixel_blue = blue[row, col]
            pixel_green = green[row, col]
            pixel_red = red[row, col]
            is_skin[row, col] = utils.skin_color_judge_2(pixel_blue, pixel_green, pixel_red)
    result = img_gray.copy()
    result[is_skin == 1] = 255
    result[is_skin == 0] = 0
    cv.imshow("Formula 2", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def skin_color_segmentation_3(filename):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    cv.imshow("Source", img)
    rows, cols = img.shape[:2]
    blue, green, red = cv.split(img)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    is_skin = np.zeros_like(img_gray)
    for row in range(rows):
        for col in range(cols):
            pixel_blue = blue[row, col]
            pixel_green = green[row, col]
            pixel_red = red[row, col]
            is_skin[row, col] = utils.skin_color_judge_3(pixel_blue, pixel_green, pixel_red)
    result = img_gray.copy()
    result[is_skin == 1] = 255
    result[is_skin == 0] = 0
    cv.imshow("Formula 3", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


# CIE lab
def CIE_lab(filename):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    cv.imshow("Source", img)
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    img_lab = cv.split(img_lab)
    sampleAreas = []
    cv.setMouseCallback('Source', utils.MouseHandler, (img, img_lab, sampleAreas, 10))

    while True:
        key = cv.waitKey(20) & 0xFF
        if key == 27:
            break
        elif key == 114:
            cv.destroyAllWindows()
            sampleAreas = []
            cv.imshow("Source", img)
            cv.setMouseCallback("Source", utils.MouseHandler, (img, img_lab, sampleAreas, 10))

    cv.destroyAllWindows()

# img = cv.imread(filename, cv.IMREAD_COLOR)
# cv.imshow("Source", img)
# img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# img_lab = cv.split(img_lab)
# sampleAreas = []
# cv.setMouseCallback('Source', utils.MouseHandler, (img, img_lab, sampleAreas, 10))
#
# while True:
#     key = cv.waitKey(20) & 0xFF
#     if key == 27:
#         break
#     elif key == 114:
#         cv.destroyAllWindows()
#         sampleAreas = []
#         cv.imshow("Source", img)
#         cv.setMouseCallback("Source", utils.MouseHandler, (img, img_lab, sampleAreas, 10))
#
# cv.destroyAllWindows()
# """


def k_means_cluster(filename):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    img_lab = cv.split(img_lab)
    ab = cv.merge([img_lab[1], img_lab[2]])
    ab = ab.reshape(-1, 2).astype(np.float32)
    k = 3
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv.kmeans(ab, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(img_lab[0].shape)
    segmented_frames = []
    for i in range(k):
        img_temp = np.zeros_like(img)
        mask = labels = i
        img_temp[mask] = img[mask, :]
        segmented_frames.append(img_temp)
    cv.imshow("pic 1", segmented_frames[0])
    cv.imshow("pic 2", segmented_frames[1])
    cv.imshow("pic 3", segmented_frames[2])
    cv.waitKey(0)
    cv.destroyAllWindows()


def k_means_cluster_web(filename):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    cv.imshow("Source", img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    # bigger k, more classes, more clear, more distinct
    k = 20
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((img.shape))

    cv.imshow("Result", segmented_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Texture segmentation
def texture_segmentation(filename):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray", img_gray)

    E = skimage.filters.rank.entropy(img_gray, skimage.morphology.square(9)).astype(np.float32)
    Eim = (E - E.min()) / (E.max() - E.min())
    cv.imshow("E", E)
    cv.imshow("Eim", Eim)
    cv.waitKey(0)

    ret, BW1 = cv2.threshold(np.uint8(Eim * 255), 0, 255, cv.THRESH_OTSU)
    cv.imshow("BW1", BW1)
    cv.waitKey(0)

    BWao = utils.bwareaopen(BW1, 2000)
    nhood = cv2.getStructuringElement(cv.MORPH_RECT, (9, 9))
    closeBWao = cv.morphologyEx(BWao, cv.MORPH_CLOSE, nhood)
    Mask1 = utils.imfillholes(closeBWao)

    cv.imshow("After bwareaopen 1", BWao)
    cv.waitKey(0)
    cv.imshow("After close 1", closeBWao)
    cv.waitKey(0)
    cv.imshow("After fill hole 1", Mask1)
    cv.waitKey(0)

    contours, h = cv.findContours(Mask1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(Mask1)
    cv.drawContours(boundary, contours, -1, 255, 1)
    cv.imshow("Boundary 1", boundary)
    cv.waitKey(0)
    segment_results = img_gray.copy()
    segment_results[boundary != 0] = 255
    # cv.imshow("Segment result 1", segment_results)
    cv.imshow("Segment result", segment_results)
    cv.waitKey(0)
    img2 = img_gray.copy()
    img2[Mask1 != 0] = 0
    cv.imshow("img2", img2)
    cv.waitKey(0)

    E2 = skimage.filters.rank.entropy(img2, skimage.morphology.square(9)).astype(np.float32)
    Eim2 = (E2 - E2.min()) / (E2.max() - E2.min())
    cv.imshow("Entropy 2", Eim2)
    cv.waitKey(0)

    ret, BW2 = cv.threshold(np.uint8(Eim2 * 255), 0, 255, cv.THRESH_OTSU)
    cv.imshow("BW2", BW2)
    cv.waitKey(0)

    BW2ao = utils.bwareaopen(BW2, 2000)
    nhood = cv2.getStructuringElement(cv.MORPH_RECT, (9, 9))
    closeBW2ao = cv.morphologyEx(BW2ao, cv.MORPH_CLOSE, nhood)
    Mask2 = utils.imfillholes(closeBW2ao)

    contours2, h = cv.findContours(Mask2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    boundary2 = np.zeros_like(Mask2)
    cv.drawContours(boundary2, contours2, -1, 255, 1)
    segment_results2 = img2.copy()
    segment_results2[boundary2 is not 0] = 255

    texture1 = img.copy()
    texture1[Mask2 == 0] = 0
    texture2 = img.copy()
    texture2[Mask2 != 0] = 0
    cv.imshow("Texture 1", texture1)
    cv.waitKey(0)
    cv.imshow("Texture 2", texture2)
    cv.waitKey(0)

    cv.waitKey(0)
    cv.destroyAllWindows()


# skin_color_segmentation_2("/Users/stone/PycharmProjects/image_processing/lab1_python/people1.jpeg")
# skin_color_segmentation_2("/Users/stone/PycharmProjects/image_processing/lab1_python/people2.jpeg")
# k_means_cluster_web("/Users/stone/PycharmProjects/image_processing/lab1_python/PeopleInITMO.jpg")
# CIE_lab("/Users/stone/PycharmProjects/image_processing/lab1_python/PeopleInITMO.png")
weber_function("/Users/stone/PycharmProjects/image_processing/lab1_python/top_4.jpg")
# texture_segmentation("/Users/stone/PycharmProjects/image_processing/lab1_python/bot_4.jpg")