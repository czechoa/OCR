from PIL import Image
from matplotlib import pyplot as plt


def get_image_text_from_image(image, bbox):
    return image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]


def plot_image_text(image, bbox):
    image_text = get_image_text_from_image(image, bbox)

    plt.imshow(image_text)
    plt.show()
    return image_text


def plot_image(image):
    plt.imshow(image)
    plt.show()