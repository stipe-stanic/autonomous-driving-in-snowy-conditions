import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(image: np.ndarray) -> None:
    plt.imshow(image, cmap='gray')
    plt.show()


def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def select_white_yellow_hls(image):
    converted = convert_hls(image)
    # show_image(converted)

    # white color mask
    lower = np.uint8([0, 200, 0])  # hue, lightness, saturation
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # show_image(white_mask)

    # yellow color mask
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # show_image(yellow_mask)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    # show_image(mask)
    return cv2.bitwise_and(image, image, mask=mask)


def apply_smoothing(image, kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# Ratio of 1:3
def detect_edges(image, low_treshold=50, high_treshold=150):
    return cv2.Canny(image, low_treshold, high_treshold)


if __name__ == '__main__':
    image = plt.imread("data/YellowWhite.jpg")
    print(f'Image shape: {image.shape}')  # h, w
    show_image(image)

    white_yellow_image = select_white_yellow_hls(image)
    show_image(white_yellow_image)

    gray_image = convert_gray_scale(white_yellow_image)
    show_image(gray_image)

    blurred_image = apply_smoothing(gray_image)
    show_image(blurred_image)

    edge_image = detect_edges(image)
    show_image(edge_image)