# check our Image_augmentation for usage details

import numpy as np
np.random.seed(1)

import skimage
from skimage.util import view_as_windows
from skimage import io, color, filters
from scipy.ndimage import gaussian_filter, maximum_filter

from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter # different than max pooling
import os

import matplotlib.pyplot as plt

class SingleImages():
    def __init__(self,generate_crops = False,crop_ratio=0.15, random_rotate=True, num_horizontal_splits=2, num_vertical_splits=1):
        path = "single_wide_images"
        self.rand_rotate = random_rotate
        self.hsplits = num_horizontal_splits
        self.vsplits = num_vertical_splits
        self.single_images = list()
        self.crops = list()
        self.crop_ratio = crop_ratio

        for file in os.listdir(path):
            img = skimage.io.imread(path + "/" + file)
            self.single_images.append(img)

        if generate_crops:
            for img in self.single_images:
                cropped = self.crop_image_with_overlap(img)
                if self.rand_rotate:
                    cropped = [skimage.transform.rotate(c, np.random.randint(-5,5)) for c in cropped]
                cropped = [SingleImages.convert_image(c) for c in cropped]
                self.crops.append(cropped)

    def convert_image(image):
        # Convert the image from 0-1 to 0-255
        if image.dtype == np.float64 or image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        return image

    @staticmethod
    def plot_image(img, figsize = (15,8)):
        plt.figure(figsize=figsize)
        plt.imshow(img,cmap="Blues")
        plt.axis("off")

    def crop_image_with_overlap(self,image):
        height, width = image.shape[:2]
        crop_height = height // self.vsplits
        crop_width = width // self.hsplits
        overlap_h = int(crop_height * self.crop_ratio)
        overlap_w = int(crop_width * self.crop_ratio)

        crops = []

        for i in range(self.vsplits):
            for j in range(self.hsplits):
                start_row = max(0, i * crop_height - overlap_h * i)
                start_col = max(0, j * crop_width - overlap_w * j)
                end_row = min(height, start_row + crop_height + overlap_h)
                end_col = min(width, start_col + crop_width + overlap_w)
                crop = image[start_row:end_row, start_col:end_col]
                crops.append(crop)

        return crops

    def plot_crops(self, img, num_horizontal_splits=2, num_vertical_splits=2):
        plt.figure(figsize=(15, 8))
        for i, crop in enumerate(img):
            plt.subplot(num_vertical_splits, num_horizontal_splits, i + 1)
            plt.imshow(crop)
            plt.axis('off')
        plt.show()


class PairWise:
    def __init__(self):
        path = "difficult_examples"
        self.pair_images_names = " # 0 pond L, 1 Himalyan L , Mountain 2 , Himalayan R 3 , Mountain R , pond 5"
        self.pair_images = list()
        for file in os.listdir(path):
            img = skimage.io.imread(path + "/" + file)
            self.pair_images.append(img)

        self.pond = [self.pair_images[0], self.pair_images[5] ]
        self.mountain = [self.pair_images[2], self.pair_images[4] ]
        self.himalyan = [self.pair_images[1], self.pair_images[3] ]
        self.pair_images = [self.pond , self.himalyan , self.mountain]

    def plot_pair(self,ix):
        plt.figure(figsize=(15,8))
        plt.subplot(1,2,1)
        plt.imshow(self.pair_images[ix][0])
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(self.pair_images[ix][1])
        plt.axis("off")

