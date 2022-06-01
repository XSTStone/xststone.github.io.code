import cv2 as cv
import numpy as np
import os
import skimage
import matplotlib.pyplot as plt
import math
import time

# Descriptor type
FP_SIFT = 0
FP_ORB = 1

# Descriptor matching method
# Brute force
MM_BF = 1
# Fast Library for Approximate Nearest Neighbors
MM_FLANN = 2
# K nearest neighbors
MM_KNN = 4
# Do a cross-check
MM_CC = 8


def DoFeaturePoints(fn="tower.png", method=FP_SIFT, draw_rich=True, num_points=None, color=-1, fn_out=None):
    # Read an image from file
    I = cv.imread(fn, cv.IMREAD_COLOR)
    if not isinstance(I, np.ndarray) or I.data == None:
        print("Error reading file \"{}\"".format(fn))
        exit()
    # show image
    cv.imshow("Source", I)
    # convery to grayscale
    Igray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)

    # Do Feature points search
    if method == FP_SIFT:
        # SIFT
        method_str = "SIFT"
        if num_points != None:
            sift = cv.SIFT_create(num_points)
        else:
            sift = cv.SIFT_create()
        Ifp = sift.detect(Igray)
    elif method == FP_ORB:
        # ORB
        method_str = "ORB"
        if num_points != None:
            orb = cv.ORB_create(num_points)
        else:
            orb = cv.ORB_create()
        Ifp = orb.detect(Igray)
    else:
        print("DoFeaturePoints: Unkown method")

    # Create output image
    if draw_rich:
        Iout = cv.drawKeypoints(I, Ifp, None, color=color, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        Iout = cv.drawKeypoints(I, Ifp, None, color=color)

    # Display an image
    cv.imshow("{} feature points".format(method_str), Iout)

    # Save to file
    if fn_out != None:
        cv.imwrite(fn_out, Iout)

    cv.waitKey()
    cv.destroyAllWindows()


def DoMatch(fn1="tower.png", fn2="building.png", fp_method=0, match_method=1, num_matches=None, knn_ratio=0.75,
            fn_out_match=None, fn_out_box=None, fn_out_trans=None):
    # Read an image from file
    I1 = cv.imread(fn1, cv.IMREAD_COLOR)
    if not isinstance(I1, np.ndarray) or I1.data == None:
        print("Error reading file \"{}\"".format(fn1))
        exit()
    I2 = cv.imread(fn2, cv.IMREAD_COLOR)
    if not isinstance(I2, np.ndarray) or I2.data == None:
        print("Error reading file \"{}\"".format(fn2))
        exit()

    # Show images
    cv.imshow("Query", I1)
    cv.imshow("Source", I2)

    # convery to grayscale
    I1gray = cv.cvtColor(I1, cv.COLOR_BGR2GRAY)
    I2gray = cv.cvtColor(I2, cv.COLOR_BGR2GRAY)

    # Do Feature points search and description
    if fp_method == FP_SIFT:
        # SIFT
        fp_method_str = "SIFT"
        sift = cv.SIFT_create()
        I1fp, I1des = sift.detectAndCompute(I1gray, None)
        I2fp, I2des = sift.detectAndCompute(I2gray, None)
    elif fp_method == FP_ORB:
        # ORB
        fp_method_str = "ORB"
        orb = cv.ORB_create()
        I1fp, I1des = orb.detectAndCompute(I1gray, None)
        I2fp, I2des = orb.detectAndCompute(I2gray, None)
    else:
        print("DoMatch: Unknown feature points method")
        return

    # Create output image
    I1out = I1.copy()
    cv.drawKeypoints(I1, I1fp, I1out, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    I2out = I2.copy()
    cv.drawKeypoints(I2, I2fp, I2out, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Dispaly images
    cv.imshow("{} query points".format(fp_method_str), I1out)
    cv.imshow("{} source points".format(fp_method_str), I2out)

    # Do feature points matching
    if match_method & MM_BF:
        # Brute force single match
        match_method_str = "Brute force"

        # Enable cross-check only if no KNN is selected
        crossCheck = False
        if (match_method & MM_CC) and not (match_method & MM_KNN):
            crossCheck = True

        # Create matcher
        if fp_method == FP_SIFT:
            matcher = cv.BFMatcher(crossCheck=crossCheck)
        else:
            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=crossCheck)
    elif match_method & MM_FLANN:
        # FLANN
        match_method_str = "FLANN"

        # Define parameters
        FLANN_INDEX_KDTREE = 1
        FLANN_INDEX_LSH = 6
        if fp_method == FP_SIFT:
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=9,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
        search_params = dict(check=50)  # or pass an empty dictionary

        # Create matcher
        matcher = cv.FlannBasedMatcher(index_params, search_params)

    else:
        print("DoMatch: Unknown match method")
        return

    # Run matching
    if match_method & MM_KNN:
        # KNN matching
        match_method_str_2 = "KNN"
        matches = matcher.knnMatch(I1des, I2des, k=2)

        # Select good matches
        good = []
        for m in matches:
            if len(m) > 1:
                if m[0].distance < knn_ratio * m[1].distance:
                    good.append(m[0])
        matches = good
    else:
        # Single point matching
        if match_method & MM_CC:
            match_method_str_2 = "CC"
        else:
            match_method_str_2 = "single"
        matches = matcher.match(I1des, I2des)
    # Draw matches
    if num_matches != None:
        # Sort and draw first num_matches
        matches = sorted(matches, key=lambda x: x.distance)
        Imatch = cv.drawMatches(I1, I1fp, I2, I2fp, matches[:num_matches], None,
                                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                matchColor=(0, 255, 0))
    else:
        Imatch = cv.drawMatches(I1, I1fp, I2, I2fp, matches, None,
                                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                matchColor=(0, 255, 0))

    # Display an image
    cv.imshow("{} {} matches".format(match_method_str, match_method_str_2), Imatch)

    # Homography
    MIN_MATCH_COUNT = 10
    if len(matches) < MIN_MATCH_COUNT:
        print("DoMatch:Not enough matches to calculate homography")
    else:
        # Create arrays of coordinates in two images
        I1pts = np.float32([I1fp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        I2pts = np.float32([I2fp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Run RANSAC method to find the transformation
        M, mask = cv.findHomography(I1pts, I2pts, cv.RANSAC, 5)
        mask = mask.ravel().tolist()

        # Calculate a rectangle of the first image in second CS
        h, w = I1.shape[:2]
        I1box = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        I1to2box = cv.perspectiveTransform(I1box, M)

        # Draw a red box of first image on the second one
        I2res = cv.polylines(I2, [np.int32(I1to2box)], True, (0, 0, 255), 1, cv.LINE_AA)
        cv.imshow("Query on Source", I2res)

        # Draw found good matches with found transformation
        Itrans = cv.drawMatches(I1, I1fp, I2res, I2fp, matches, None, matchesMask=mask,
                                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                matchColor=(0, 255, 0))

        # Display an image
        cv.imshow("Transformation", Itrans)

    # Save to file
    if fn_out_match != None:
        cv.imwrite(fn_out_match, Imatch)
    if fn_out_box != None:
        cv.imwrite(fn_out_box, I2res)
    if fn_out_trans != None:
        cv.imwrite(fn_out_trans, Itrans)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # filename = "/Users/stone/PycharmProjects/image_processing/lab1_python/people1.jpeg"
    # DoFeaturePoints(filename, FP_SIFT, True, 400, -1)
    # DoFeaturePoints(filename, FP_SIFT, True)
    # DoFeaturePoints(filename, FP_ORB, True, 30, -1)
    # DoFeaturePoints(filename, FP_ORB, True)
    # #
    # filename = "/Users/stone/PycharmProjects/image_processing/lab1_python/XH.jpg"
    # DoFeaturePoints(filename, FP_SIFT, True, 400, -1)
    # DoFeaturePoints(filename, FP_SIFT, True)
    # DoFeaturePoints(filename, FP_ORB, True, 50, -1)
    # DoFeaturePoints(filename, FP_ORB, True)
    #
    # filename = "/Users/stone/PycharmProjects/image_processing/lab1_python/mountain.png"
    # DoFeaturePoints(filename, FP_SIFT, True, 100, -1)
    # DoFeaturePoints(filename, FP_SIFT, True)
    # DoFeaturePoints(filename, FP_ORB, True, 50, -1)
    # DoFeaturePoints(filename, FP_ORB, True)


    building = "/Users/stone/PycharmProjects/image_processing/lab1_python/building.png"
    tower = "/Users/stone/PycharmProjects/image_processing/lab1_python/tower.png"
    # DoMatch(tower, building, FP_SIFT, MM_FLANN + MM_KNN, 100, knn_ratio=0.75)
    head = "/Users/stone/PycharmProjects/image_processing/lab1_python/XH_head.png"
    total = "/Users/stone/PycharmProjects/image_processing/lab1_python/XH.jpg"
    DoMatch(head, total, FP_SIFT, MM_FLANN, 100, knn_ratio=0.75)
