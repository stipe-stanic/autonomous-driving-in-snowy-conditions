from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(image: np.ndarray) -> None:
    plt.imshow(image, cmap='gray')
    plt.show()


image = cv2.imread("data/road_with_lanes.jpg")
print(f'Image shape: {image.shape}')  # h, w
# show_image(image)


lane_image = np.copy(image)
gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
show_image(gray_image)

# Noise reduction and smoothening
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
# show_image(blurred_image)

# Edge detection
canny_image = cv2.Canny(blurred_image, 50, 150)  # Low to high threshold 1:3
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(blurred_image, cmap='gray')
ax[1].imshow(canny_image, cmap='gray')
plt.show()


# ROI
def region_of_interest(image: np.ndarray) -> np.ndarray[int]:
    height = image.shape[0]
    triangle = np.array([(200, height), (1100, height), (550, 250)], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [triangle], 255)

    return mask


mask = region_of_interest(blurred_image)
cropped_image = canny_image & mask
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(mask, cmap='gray')
ax[1].imshow(cropped_image, cmap='gray')
plt.show()

# Hough transform
lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=100,
                        lines=np.array([]), minLineLength=40, maxLineGap=5)
# rho - number of pixels for polar grid
# theta - number of pixels for an angle (radian) grid
# threshold - min no. of intersections
# minLineLength - min. width to be considered as a line
# maxLineGap - No. of pixels between 2 lines


def display_line(image, lines):
    lane_image = np.zeros_like(image)

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)  # shape of 4 elements (4, 1)
        cv2.line(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return lane_image


line_image = display_line(lane_image, lines)
plt.imshow(line_image)
plt.show()

combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)  # super impose line on the image
plt.figure(figsize=(8, 10))
plt.imshow(combo_image)
plt.show()


def make_coordinates(image, line_params):
    """Optimizes lines by making multiple lines as one. Generates coordinates.

    :param image:
    :param line_params:
    :return:
    """

    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * 0.5)  # a point at 0.6 of the image height
    x1 = int((y1 - intercept) / slope)  # y = mx + c
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_line(image, lines):
    """Determines best fit lines"""
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # 1st degree polynomial - straight line
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    print(f'Left: {left_fit_average}')
    print(f'Right: {right_fit_average}')

    return np.array([left_line, right_line])


avg_lines = average_line(lane_image, lines)
line_image = display_line(lane_image, avg_lines)
plt.imshow(line_image)
plt.show()

combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
plt.figure(figsize=(8, 10))
plt.imshow(combo_image)
plt.show()

cap = cv2.VideoCapture("data/straight_road.mp4")

output_video_filename = 'lane_detection_test_video/lane_detection_test.mp4'
fps = 30.0
frame_width, frame_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# unpacking string 'mp4v' so it's equivalent to calling cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_filename, fourcc, fps, (int(frame_width), int(frame_height)))
i = 0
while cap.isOpened():
    res, frame = cap.read()
    if res is False:
        break

    lane_image = np.copy(frame)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(blur, 50, 150)
    plt.imsave(f'lane_edge_test/image_{i + 1}.jpg', canny_image, cmap='gray')
    i = i + 1
    mask = region_of_interest(canny_image)
    cropped_image = canny_image & mask

    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=100, lines=np.array([]),
                           minLineLength=40, maxLineGap=5)
    line_image = display_line(lane_image, lines)
    avg_lines = average_line(lane_image, lines)
    line_image = display_line(lane_image, avg_lines)

    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)  # Supper impose line on the image

    out.write(combo_image)

# Releases the VideoWriter and VideoCapture
out.release()
cap.release()



















