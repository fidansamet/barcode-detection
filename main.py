import os
import cv2
from matplotlib import pyplot as plt


original_subset_dir = "dataset/Original_Subset"
detection_subset_dir = "dataset/Detection_Subset"

original_subset_image_names = os.listdir(original_subset_dir)
detection_subset_image_names = os.listdir(detection_subset_dir)


def read_original_subset():
    for image_name in original_subset_image_names:
        print(image_name)
        img = cv2.imread(original_subset_dir + '/' + image_name)
        find_edges(img)
        #cv2.imshow('image', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


def find_edges(img):
    edges = cv2.Canny(img, 100, 200)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    read_original_subset()
