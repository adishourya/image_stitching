# Calculating Harris Corners
```py
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

```
## Steps to calculate Harris Corners
* Convert image to greyscale (color channels has no effect on the corners of the image)
* calculate the first order gradients with the help of a sobel filter
* Design the M block matrix by first passing the square of the first order gradients through a gaussian filter
* return the harris response (optionally apply non maximum suppression; different than max pooling) `return detM - k * traceM**2`
* apply thresholding for robust corners
