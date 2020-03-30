import os
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np


original_subset_dir = "dataset/Original_Subset"
detection_subset_dir = "dataset/Detection_Subset"

original_subset_image_names = os.listdir(original_subset_dir)
detection_subset_image_names = os.listdir(detection_subset_dir)


def read_original_subset():
    for image_name in original_subset_image_names:
        print(image_name)

        original_img = cv2.imread(original_subset_dir + '/' + image_name)
        detected_img = cv2.imread(detection_subset_dir + '/' + image_name)

        #masked_img = cv2.bitwise_and(original_img, detected_img)
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)

        edges, plot_input = find_edges(original_img)

        # hough_transform(image_name, gray_img, edges)

        masked_img = cv2.bitwise_and(detected_img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        accumulator, thetas, rhos = find_hough_lines(cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY))

        lined_image = hough_to_image_space(original_img, accumulator, thetas, rhos)

        plot_input = np.concatenate((plot_input, lined_image), axis=1)

        # PLOT
        plt.imshow(plot_input, cmap='gray')
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.show()


def find_edges(img):
    edges = cv2.Canny(img, 100, 200)
    # sigma = 0.33
    # v = np.median(img)
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    # edges = cv2.Canny(img, 150, 155, apertureSize=3)

    plot_input = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), axis=1)

    return edges, plot_input


def hough_transform(image_name, img, edges):
    lines = cv2.HoughLines(edges, 1, np.pi/90, 0)
    print(lines)

    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('houghlines' + image_name + '.jpg', img)


def find_hough_lines(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width = img.shape[0]
    height = img.shape[1]
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def hough_to_image_space(img, accumulator, thetas, rhos):

    # Easiest peak finding based on max votes
    idx = np.argmax(accumulator)
    rho = rhos[int(round(idx / accumulator.shape[1]))]
    theta = thetas[idx % accumulator.shape[1]]

    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

    print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))
    lineThickness = 2
    cv2.line(img, pt1, pt2, (255, 0, 0), lineThickness)

    return img


if __name__ == '__main__':
    read_original_subset()
