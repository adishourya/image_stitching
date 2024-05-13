import numpy as np
np.random.seed(1)

import cv2 # only to compare and validate
import skimage
from skimage.util import view_as_windows
from skimage import io, color, filters
from scipy.ndimage import gaussian_filter, maximum_filter

from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter # different than max pooling
import os

import matplotlib.pyplot as plt

class Transformations:
    @staticmethod
    def grey_scale(I):
        return 0.299 * I[:,:,0] + 0.587 * I[:,:,1] + 0.114 * I[:,:,2]

    @staticmethod
    def convolve(I, k):
        return convolve2d(I, k, mode='same', boundary='fill', fillvalue=0)

    @staticmethod
    def first_order_grad(I):
        return skimage.filters.sobel_h(I), skimage.filters.sobel_v(I)

    @staticmethod
    def normalize(arr,scale=254):
        return ((arr - arr.min())/ (arr.max() - arr.min()) * scale)

    @staticmethod
    def calculate_M(Ix, Iy, sigma=1.5):
        Ixx = gaussian_filter(Ix**2, sigma)
        Iyy = gaussian_filter(Iy**2, sigma)
        Ixy = gaussian_filter(Ix*Iy, sigma)
        # does not exactly return block Matrix of M but all its entries
        return Ixx, Iyy, Ixy

    @staticmethod
    def harris_response(Ixx, Iyy, Ixy, k=0.04):
        detM = Ixx * Iyy - Ixy**2
        traceM = Ixx + Iyy
        return detM - k * traceM**2

    @staticmethod # from wikipedia
    def non_maximum_suppression(response, size=3):
        data_max = maximum_filter(response, size=size)
        mask = (response == data_max)
        return response * mask

    @staticmethod
    def harris_corner_detector(img, sigma=1.5, k=0.04, threshold=0.01, plot_overlay = True):
        image = img
        if image.ndim == 3:
            image = Transformations.grey_scale(image)

        Ix, Iy = Transformations.first_order_grad(image)
        Ixx, Iyy, Ixy = Transformations.calculate_M(Ix, Iy, sigma)
        HR = Transformations.harris_response(Ixx, Iyy, Ixy, k)
        response = Transformations.non_maximum_suppression(HR)

        # Thresholding
        corners = np.zeros_like(response)
        corners[response > threshold * response.max()] = 1

        if plot_overlay:
            plt.figure(figsize=(15,8))
            plt.imshow(img)
            plt.scatter(np.where(corners)[1], np.where(corners)[0], color='r', s=1)
            plt.title('Hand Calculated Harris Corners')
            plt.axis("off")
            plt.show()
        return response , corners


class HarrisDetector(Transformations):
    @staticmethod
    def hand_harris_corner_detector(img, sigma=1.5, k=0.04, threshold=0.01, plot_overlay = True):
        image = img
        if image.ndim == 3:
            image = HarrisDetector.grey_scale(image)

        Ix, Iy = HarrisDetector.first_order_grad(image)
        Ixx, Iyy, Ixy = HarrisDetector.calculate_M(Ix, Iy, sigma)
        HR = HarrisDetector.harris_response(Ixx, Iyy, Ixy, k)
        response = HarrisDetector.non_maximum_suppression(HR)

        # Thresholding
        corners = np.zeros_like(response)
        corners[response > threshold * response.max()] = 1

        if plot_overlay:
            HarrisDetector.plot_overlay(img,corners,"Hand Calculated Harris Corner Detector")

        return response, corners


    @staticmethod
    def opencv_implementation(img,blockSize = 2 , ksize=3,k=0.04 , plot_overlay = True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        corners = dst > 0.01 * dst.max()

        if plot_overlay:
            HarrisDetector.plot_overlay(img,corners,"OpenCV Implementattion")
        return corners

    def compare_opencv(self,img):
        _,hand_corners = self.harris_corner_detector(img,plot_overlay=False)
        open_cv_corners = self.opencv_implementation(img,plot_overlay=False)
        fig, ax = plt.subplot_mosaic("AB",figsize=(15,8))
        ax["A"].imshow(img)
        ax["A"].scatter(np.where(hand_corners)[1], np.where(hand_corners)[0], color="r", s=1)
        ax["A"].set_title("Hand Calculated Harris Corners")
        ax["B"].imshow(img)
        ax["B"].scatter(np.where(open_cv_corners)[1], np.where(open_cv_corners)[0], color="b", s=1)
        ax["B"].set_title("Open CV2 Calculated Harris Corners")
        ax["A"].axis("off")
        ax["B"].axis("off")
        plt.show()



    @staticmethod
    def plot_overlay(img,corners,title):
        plt.figure(figsize=(15,8))
        plt.imshow(img)
        plt.scatter(np.where(corners)[1], np.where(corners)[0], color='r', s=1)
        plt.title(title)
        plt.axis("off")
        plt.show()

