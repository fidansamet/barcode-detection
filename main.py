import os
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np


original_subset_dir = "dataset/Original_Subset"
detection_subset_dir = "dataset/Detection_Subset"
line_thickness = 2
plt.style.use("ggplot")
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
        plt.imshow(cv2.cvtColor(plot_input, cv2.COLOR_BGR2RGB))
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.show()


def obtain_edge_map(img):
    sigma = 0.2
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # edges = cv2.Canny(img, 150, 155, apertureSize=3)
    edges = cv2.Canny(img, lower, upper)
    plot_input = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), axis=1)
    print("EDGE")
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

        curr_rhos = np.add(np.array(cos_t) * x, np.array(sin_t) * y) + diag_len

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            accumulator[int(curr_rhos[t_idx]), t_idx] += 1

    print("HOUGH")
    return accumulator, thetas, rhos


def hough_to_image_space(original_img, detected_img, accumulator, thetas, rhos, plot_input):
    threshold = get_avg_threshold(accumulator)
    print(threshold)
    y_idxs, x_idxs = np.where(accumulator >= threshold)

    for i in range(len(y_idxs)):
        # Easiest peak finding based on max votes
        idx = get_flatten_idx(accumulator.shape[1], y_idxs[i], x_idxs[i])
        rho = rhos[int(round(idx / accumulator.shape[1]))]
        theta = thetas[idx % accumulator.shape[1]]

        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

        cv2.line(original_img, pt1, pt2, (0, 0, 255), line_thickness)
        cv2.line(detected_img, pt1, pt2, (0, 0, 255), line_thickness)

    plot_input = np.concatenate((plot_input, original_img), axis=1)
    plot_input = np.concatenate((plot_input, detected_img), axis=1)
    print("TRANSFORM")
    return plot_input


def get_n_max_idx(arr, n):
    idx = np.argpartition(arr, arr.size - n, axis=None)[-n:]
    a = np.unravel_index(idx, arr.shape)
    return np.column_stack(a)


def get_avg_threshold(accumulator):
    out_tpl = np.nonzero(accumulator)
    top_n = int(len(out_tpl[0]) / 1400)
    print("top n")
    print(top_n)
    res = get_n_max_idx(accumulator, top_n)
    sum = 0
    for i in range(len(res)):
        sum += accumulator[res[i][0]][res[i][1]]

    return sum/len(res)


def get_flatten_idx(m, r, c):
    return (r * m) + c


if __name__ == '__main__':
    detect_barcode()
