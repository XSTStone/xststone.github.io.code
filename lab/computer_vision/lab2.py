import math

import cv2 as cv
import numpy as np
from fractions import Fraction
import utils as ut

def line_search(filename):
    image = cv.imread(filename, cv.IMREAD_COLOR)
    cv.imshow("original", image)
    image_edge_canny = cv.Canny(image, 50, 200, None, 3)
    cv.imshow("canny", image_edge_canny)

    linesP = cv.HoughLinesP(image_edge_canny, 1, np.pi / 180, 50, None, 50, 4)
    image_output = np.zeros_like(image)
    image_output[True] = [255, 255, 255]
    start_point_color = (0, 0, 0)
    end_point_color = (0, 255, 0)
    point_size = 1
    thickness = 4
    counter = 0
    largest_length = 0
    shortest_length = 0
    if linesP is not None:
        for i in range(0, len(linesP)):
            line = linesP[i][0]
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            pt1_x = line[0]
            pt1_y = line[1]
            pt2_x = line[2]
            pt2_y = line[3]
            distance = math.sqrt(pow(pt1_x - pt2_x, 2) + pow(pt1_y - pt2_y, 2))
            if shortest_length > distance:
                shortest_length = distance
            if largest_length < distance:
                largest_length = distance
            cv.line(image_output, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)
            cv.circle(image_output, pt1, point_size, start_point_color, thickness)
            cv.circle(image_output, pt2, point_size, end_point_color, thickness)
            counter += 1
    cv.imshow("Classic", image_output)
    print("counter = ", counter)
    print("large dis = ", largest_length)
    print("short dis = ", shortest_length)

    cv.waitKey(0)
    cv.destroyAllWindows()


def circle_search(filename):
    image = cv.imread(filename, cv.IMREAD_COLOR)
    image_gray = cv.cvtColor(image, cv.IMREAD_GRAYSCALE)
    image_edge_canny = cv.Canny(image_gray, 80, 190, None, 3)
    cv.imshow("canny", image_edge_canny)

    image_edge_canny = np.uint8(image_edge_canny)
    circles = cv.HoughCircles(image_edge_canny, cv.HOUGH_GRADIENT, 1, 70, param1=80, param2=100, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        cv.circle(image_gray, (i[0], i[1]), i[2], (0, 0, 0), 2)
        cv.circle(image_gray, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv.imshow("circles", image_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()


# line_search("/Users/stone/PycharmProjects/image_processing/lab1_python/barcode_zhubi.jpg")
# circle_search("/Users/stone/PycharmProjects/image_processing/lab1_python/OpenCV.png") # 80 200 40 50 44
# circle_search("/Users/stone/PycharmProjects/image_processing/lab1_python/coins1.jpg") # 80 200 90 160 100
circle_search("/Users/stone/PycharmProjects/image_processing/lab1_python/coins9.png") # 80 190 70 80 100