import cv2


def median_filtering(img):
    return cv2.medianBlur(img,3)

def bilateral_filtering(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def gaussian_blurring(img):
    return cv2.bilateralFilter(img,9,75,75)


def to_threshold_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    #
    (T, thresh) = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    return thresh
