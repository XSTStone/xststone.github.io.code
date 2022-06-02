"""
Open
Close
Source
bwarea open
erode
dilate
open+close
"""

import cv2 as cv
import numpy as np

import utils

filename = "/Users/stone/PycharmProjects/image_processing/lab1_python/char1.jpg"
filename2 = "/Users/stone/PycharmProjects/image_processing/lab1_python/photo1.jpg"
filename3 = "/Users/stone/PycharmProjects/image_processing/lab1_python/my_coins.png"

chinese = cv.imread(filename, cv.IMREAD_GRAYSCALE)
photo = cv.imread(filename2, cv.IMREAD_GRAYSCALE)
coins_color = cv.imread(filename3, cv.IMREAD_COLOR)
# cv.imshow("photo", photo)
# cv.imshow("chinese", chinese)
cv.imshow("coins", coins_color)

#
# Do part1
#
"""
# Do erode and dilate
kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))

chinese_eroded = cv.erode(chinese, kernel)
cv.imshow("chinese eroded", chinese_eroded)

chinese_dilated = cv.dilate(chinese, kernel)
cv.imshow("chinese dilated", chinese_dilated)

# Do open and close
kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))

chinese_closed = cv.morphologyEx(chinese, cv.MORPH_CLOSE, kernel, iterations=1)
cv.imshow("chinese closed", chinese_closed)

chinese_opened = cv.morphologyEx(chinese, cv.MORPH_OPEN, kernel, iterations=1)
cv.imshow("chinese opened", chinese_opened)

chinese_opened_closed = cv.morphologyEx(chinese_opened, cv.MORPH_CLOSE, kernel, iterations=2)
cv.imshow("chinese opened closed", chinese_opened_closed)

# Do bwarea open
photo_bwareaopen = utils.bwareaopen(photo, 20000)
cv.imshow("photo bwareaopen", photo_bwareaopen)

#
chinese_inner_contour = cv.subtract(chinese, chinese_eroded)
cv.imshow("Inner contour (I - erosion)", chinese_inner_contour)
chinese_outer_contour = cv.subtract(chinese_dilated, chinese)
cv.imshow("Outer contour (dilation - I)", chinese_outer_contour)

# End of basic operations
# """

#
# Do part2

# """
coins = cv.cvtColor(coins_color, cv.COLOR_BGR2GRAY)
retval, coins_threshold = cv.threshold(coins, 160, 255, cv.THRESH_BINARY_INV) # Make it change into binary image
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# Erode
coins_threshold_eroded = cv.morphologyEx(coins_threshold, cv.MORPH_ERODE, kernel, iterations=14, borderType=cv.BORDER_CONSTANT, borderValue=(0))  # Erode the binary image then after several dilations, the contour can be detected
cv.imshow("coins_threshold_eroded", coins_threshold_eroded)

# Dilate
T = np.zeros_like(coins_threshold)
while cv.countNonZero(coins_threshold_eroded) < coins_threshold_eroded.size:
    dilated = cv.dilate(coins_threshold_eroded, kernel, borderType=cv.BORDER_CONSTANT, borderValue=(0))
    closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel, borderType=cv.BORDER_CONSTANT, borderValue=(0))
    split = cv.subtract(closed, dilated)
    T = cv.bitwise_or(split, T)
    coins_threshold_eroded = dilated

# Border close
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
T = cv.morphologyEx(T, cv.MORPH_CLOSE, kernel, iterations=14, borderType=cv.BORDER_CONSTANT, borderValue=(255))
T_eroded = cv.morphologyEx(~T, cv.MORPH_ERODE, kernel, iterations=1)
T_contour = cv.subtract(~T, T_eroded)
cv.imshow("T", T)

# Remove border
coins_threshold = cv.bitwise_and(~T, coins_threshold)
coins_threshold[T_contour > 0] = 255
result = cv.bitwise_and(coins_threshold, coins)

cv.imshow("result", result)
# """

#
# Do part3
#
"""
coins_gray = cv.cvtColor(coins_color, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(coins_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow("thresh", thresh)

# noise removal
kernel = np.ones((9, 9), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
cv.imshow("opening", opening)

# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)
cv.imshow("sure_bg", sure_bg)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
cv.imshow("distance", dist_transform)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(coins_color, markers)
coins_color[markers == -1] = [255, 0, 0]

cv.imshow("Result", coins_color)

"""


utils.end()
