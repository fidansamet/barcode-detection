This program detects barcode lines by using Canny edge detection and Hough Transform methods. Program needs ground-truth binary segmentation (detected) map to detect barcode lines.

There are two variables that specify input images' directory names: "original_subset_dir" for original images and "detection_subset_dir" for ground-truth binary segmentation maps. Original image's and ground-truth binary segmentation map's names must be same for masking them.

Outputs will be shown as plots. 

To run the program run the following command: python3 main.py


Functions:

detect_barcode_lines(): For all input images, obtains edge map, masks edge map and original image for Hough Transformation, finds lines, converts Hough space to image space and plots the output.
obtain_edge_map(img):	Firstly, converts image from BGR to gray. Then applies median filtering to it to reduce salt-and-pepper noise and obtains blurred image of the input image. After that calculates lower and upper thresholds for edge detection by using median of the pixel intensities and sigma to vary the percentage thresholds. Thresholds are constructed based on the +/- percentages controlled by the sigma. Then applies OpenCV's Canny edge detection method to obtain an edge map of the input image. Finally, creates plot input for requested visual form, concatenates original image and edge map. Returns edges for masking and plot input for output.
find_lines(img): Firstly, calculates range of the rho with diagonal length of the image. It is from negative diagonal length to positive diagonal length. Theta ranges from −90∘ to 90∘ and it is global. Then creates 2-dimensional array as an accumulator, so its row number is equal to total rhos number and column number is equal to total thetas number. After creation, gets non-zero indices in image as rows and columns, for each value and for each theta, finds the closest rho and increments it by one in the accumulator. Each value in accumulator shows how many edge points voted for possible lines. Returns accumulator and rhos to find and draw the correct lines.
hough_to_image_space(original_img, detected_img, accumulator, rhos, plot_input): Gets threshold for input image by using its accumulator. Then gets indices as row and column of most clear lines' in input image according to threshold. Since parameter of that line is rho and theta, converts Hough space to image space and draws lines to original and detected images. Returns concatenated plot input for requested visual form.
get_n_max_idx(arr): Finds and returns 100 maximum values' indices from given array.
get_avg_threshold(arr): Gets max 100 values' indices in given array. Then calculates average of 100 maximum numbers in given array and returns it as threshold.
get_flatten_idx(total_cols, row, col): Calculates and returns flatten index of given row and column position in 2-dimensional array according to total column number.


Globals:

original_subset_dir: Original images directory name
detection_subset_dir: Ground-truth binary segmentation maps directory name
original_subset_image_names: Names of original images in given directory
detection_subset_image_names: Names of ground-truth binary segmentation maps in given directory
line_thickness: Thickness of drawn lines
thetas: Theta values from -90 degrees to +90 degrees
cos_theta: Cos values of thetas
sin_theta: Sin values of thetas
num_thetas: Number of thetas
max_n: Number of maximum numbers to calculate threshold for voted lines elemination
sigma: Percentage thresholds for constructing lower and upper thresholds in edge detection

