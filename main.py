import os
import cv2

original_subset_dir = "dataset/Original_Subset"
detection_subset_dir = "dataset/Detection_Subset"

original_subset_image_names = os.listdir(original_subset_dir)
detection_subset_image_names = os.listdir(detection_subset_dir)


def read_original_subset():
    for image_name in original_subset_image_names:
        print(image_name)
        image = cv2.imread(original_subset_dir + '/' + image_name)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    read_original_subset()
