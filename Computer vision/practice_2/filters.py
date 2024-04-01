import cv2
import numpy


class VConvolutionFilter(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src):
        return cv2.filter2D(src=src, ddepth=-1, kernel=self._kernel)


class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


img_src = cv2.imread("../images/rgb.png")

blur = BlurFilter()
blur_img = blur.apply(img_src)

emboss = EmbossFilter()
emboss_img = emboss.apply(img_src)

sharp = SharpenFilter()
sharp_img = sharp.apply(img_src)

cv2.imshow("Old image", img_src)
cv2.imshow("Blur image", blur_img)
cv2.imshow("Emboss image", emboss_img)
cv2.imshow("Sharp image", sharp_img)
cv2.waitKey()
cv2.destroyAllWindows()
