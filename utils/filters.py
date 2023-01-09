import cv2
import numpy as np

def median_filtering(img, neighbor=3):
    print(neighbor)
    return cv2.medianBlur(img,neighbor)

def bilateral_filtering(img):
    return cv2.bilateralFilter(img, 9, 75, 75)


def to_threshold_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    #
    (T, thresh) = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    return thresh


def image_processing_clean_up(image):
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
    image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]
    image = cv2.GaussianBlur(image, (1, 1), 0)
    return image