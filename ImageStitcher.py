# implemented and experimented in stitching.ipynb notebook
from tqdm import tqdm
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations  # gives all permuations
import os
from scipy.ndimage import maximum_filter  # different than max pooling
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage import io, color, filters
from skimage.util import view_as_windows
import skimage
import cv2 as cv  # only to compare and validate
import numpy as np
np.random.seed(1)


class ImageStitcher:
    def __init__(self):
        self.print_help()
        self.sift = cv.SIFT_create()

    def print_help(self):
        help_text = """
        pass in the input in any order (left_image, right_image): we make 2 attempts
        Does not accommodate >2 images
        Does not accomodate vertical panorama
        """
        print(help_text)

    def detect_features(self, images):
        keypoints_list = []
        descriptors_list = []
        for image in images:
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
        return keypoints_list, descriptors_list

    def match_features(self, descriptors_list):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = []
        num_images = len(descriptors_list)
        for i in range(num_images):
            for j in range(i + 1, num_images):
                match = flann.knnMatch(descriptors_list[i].astype(
                    np.float32), descriptors_list[j].astype(np.float32), k=2)
                good_matches = [
                    m for m, n in match if m.distance < 0.75 * n.distance]
                if good_matches:
                    matches.append((i, j, good_matches))
        return matches

    def plot_matches(self, img1, img2, kp1, kp2, matches):
        # Create an image that shows matching keypoints between two images
        match_img = cv.drawMatches(
            img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        # Convert to RGB for plotting
        plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
        plt.title("Matched Features")
        plt.axis('off')
        plt.show()

    def ransac_homography(self, keypoints1, keypoints2, good_matches, max_iters=1000, threshold=5.0):
        points1 = np.array(
            [keypoints1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        points2 = np.array(
            [keypoints2[m.trainIdx].pt for m in good_matches], dtype=np.float32)
        max_inliers_count = 0
        best_h = None
        best_inliers = []

        # TODO :: optimize without max_iters (do this part with cpython?)
        for _ in tqdm(range(max_iters)):
            if len(points1) < 4:
                continue
            indices = np.random.choice(len(points1), 4, replace=False)
            src_pts = points1[indices]
            dst_pts = points2[indices]
            h, _ = cv.findHomography(src_pts, dst_pts, 0)
            projected_pts = cv.perspectiveTransform(
                points1.reshape(-1, 1, 2), h)
            errors = np.linalg.norm(projected_pts.squeeze() - points2, axis=1)
            inliers = errors < threshold
            inlier_count = np.sum(inliers)
            if inlier_count > max_inliers_count:
                max_inliers_count = inlier_count
                best_h = h
                best_inliers = inliers

        if best_h is not None and max_inliers_count > 4:
            inlier_src_pts = points1[best_inliers]
            inlier_dst_pts = points2[best_inliers]
            best_h, _ = cv.findHomography(
                inlier_src_pts, inlier_dst_pts, cv.RANSAC)
        return best_h, best_inliers

    def attach(self, images, plot=False):
        keypoints_list, descriptors_list = self.detect_features(images)
        matches = self.match_features(descriptors_list)
        # Start with the first image as the base (we cover both)
        stitched_image = images[0]
        for i, j, good_matches in matches:
            h, _ = self.ransac_homography(
                keypoints_list[i], keypoints_list[j], good_matches)
            if plot:
                self.plot_matches(
                    images[i], images[j], keypoints_list[i], keypoints_list[j], good_matches)
            if h is not None:
                height, width, _ = images[j].shape
                stitched_image = cv.warpPerspective(
                    stitched_image, h, (width * 2, height))
                stitched_image[0:height, 0:width] = images[j]
        return stitched_image

    def stitch(self, images):
        attempt1 = self.attach(images, plot=True)
        attempt2 = self.attach(images[::-1], plot=False)
        # incorrectly stitched input will have black borders.
        # remove this bit if the user is going to pass in the right order. (run time for finding the base (left) image is almost the same as making 2 attempts)
        if attempt1.sum() > attempt2.sum():  # this is pretty janky but solves for incorrectly passed inputs
            return attempt1
        else:
            return attempt2
