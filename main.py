import os
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np


original_subset_dir = "dataset/Original_Subset"
detection_subset_dir = "dataset/Detection_Subset"

original_subset_image_names = os.listdir(original_subset_dir)
detection_subset_image_names = os.listdir(detection_subset_dir)


def detect_barcode():
    for image_name in original_subset_image_names:
        print(image_name)

        # Read original and ground truth images
        original_img = cv2.imread(original_subset_dir + '/' + image_name)
        detected_img = cv2.imread(detection_subset_dir + '/' + image_name)

        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)

        # Obtain an edge map of the input image
        edges, plot_input = obtain_edge_map(original_img)

        # Mask edge map with ground truth so only barcode lines will be found
        masked_img = cv2.bitwise_and(detected_img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        # Utilize Hough transform on edge map
        accumulator, thetas, rhos = find_hough_lines(cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY))

        # Transform Hough space to image space
        plot_input = hough_to_image_space(original_img, detected_img, accumulator, thetas, rhos, plot_input)

        # PLOT
        plt.style.use("ggplot")
        plt.imshow(cv2.cvtColor(plot_input, cv2.COLOR_BGR2RGB))
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.show()


def obtain_edge_map(img):
    # sigma = 0.33
    # v = np.median(img)
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    # edges = cv2.Canny(img, 150, 155, apertureSize=3)
    edges = cv2.Canny(img, 100, 200)
    plot_input = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), axis=1)
    return edges, plot_input


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
    # 2 * diag_len rows, num_thetas columns
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)

    # Return the indices of the elements that are non-zero.
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


def hough_to_image_space(original_img, detected_img, accumulator, thetas, rhos, plot_input):

    indices = np.where(accumulator > 100)

    for i in range(len(indices[0])):
        # Easiest peak finding based on max votes
        idx = dene(accumulator.shape[1], indices[0][i], indices[1][i])
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
        cv2.line(original_img, pt1, pt2, (0, 0, 255), lineThickness)
        cv2.line(detected_img, pt1, pt2, (0, 0, 255), lineThickness)

    plot_input = np.concatenate((plot_input, original_img), axis=1)
    plot_input = np.concatenate((plot_input, detected_img), axis=1)
    return plot_input


def dene(m, r, c):
    return (r * m) + c


if __name__ == '__main__':
    detect_barcode()
