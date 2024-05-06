import cv2
import numpy as np

def connected_component_labeling(image):
    # Perform connected component labeling using 8-connectivity rule
    
    # Essentially scans the image and divides the image into components by scanning the 8 adjacent pixels
    # and determining whether they are also foreground pixels (I think)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
    return num_labels, labels, stats, centroids

def max_pooling_2d(image, kernel_size):
    # Determine the size of the pooled image after pooling
    # This is done by dividing the dimensions of the original image by the specified kernel dimensions, which calculates how many kernel "windows" can fit over the image
    pooled_height = image.shape[0] // kernel_size[0]
    pooled_width = image.shape[1] // kernel_size[1]
    pooled_image = np.zeros((pooled_height, pooled_width), dtype=image.dtype)

    # Apply max pooling, which applies the kernel window over the image and picks out the highest intensity pixel to represent that region.
    # This results in a smaller map that keeps the features of the original image (such as edges that correspond to high-intensity pixels)
    for i in range(pooled_height):
        for j in range(pooled_width):
            h_start = i * kernel_size[0]
            h_end = h_start + kernel_size[0]
            w_start = j * kernel_size[1]
            w_end = w_start + kernel_size[1]

            # Use the maximum value within the current window
            pooled_image[i, j] = np.max(image[h_start:h_end, w_start:w_end])

    return pooled_image

def find_boundary_pixels(binary_image):
    # Get the height and width of the image
    height, width = binary_image.shape

    # List to store the coordinates of boundary pixels
    boundary_pixels = []

    # Iterate over each pixel in the image
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Check if the pixel is black
            if binary_image[y, x] == 0:
                # Get the values of the 8 neighboring pixels
                neighborhood = binary_image[y - 1:y + 2, x - 1:x + 2]

                # Check if there is a white pixel (255) in the neighborhood
                if 255 in neighborhood:
                    boundary_pixels.append((x, y))

    return boundary_pixels

def multilevel_threshold(hist):
    """
    Calculate the histogram and find the first segmentation threshold,
    dividing grayscale levels from 0 to 254 into four portions.

    :param hist: Histogram containing grayscale levels from 0 to 254
    :return: The first segmentation threshold
    """
    # Calculate the total number of pixels and determine the number of pixels per portion
    total_pixels = np.sum(hist[:-1])  # Exclude grayscale value 255
    pixels_per_portion = total_pixels // 4

    # Find the segmentation thresholds
    thresholds = []
    current_sum = 0
    for i, pixel_count in enumerate(hist[:-1]):
        current_sum += pixel_count
        if current_sum >= pixels_per_portion:
            thresholds.append(i)
            current_sum -= pixels_per_portion  # Reset current sum for the next threshold
            if len(thresholds) == 3:
                break

    # If three thresholds are not found, raise an error
    if len(thresholds) < 3:
        raise ValueError("Not enough segmentation points found in grayscale levels 0 to 254.")
    else:
        # Return the first threshold
        return thresholds[0]

def process_image(image_path, scale_x=10, scale_y=10):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Smooth with Gaussian blur to denoise for more accurate processing, which is done by averaging pixels in a 9 by 9 kernel
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)
    # Apply max pooling, which uses a 10 by 10 kernel "window" and finds the maximum values within these windows (further denoising)
    pooled_image = max_pooling_2d(blurred, (scale_x, scale_y))
    # Calculate the histogram
    hist = np.bincount(pooled_image.ravel(), minlength=256)
    hist[-1] = 0  # Ignore grayscale value 255
    # Set parameters for binarization
    threshold_value = multilevel_threshold(hist)  # Threshold segmentation based on the method proposed in the Fast automatic multilevel thresholding method article.
    max_value = 255
    # Perform binarization
    _, binary_image = cv2.threshold(pooled_image, threshold_value, max_value, cv2.THRESH_BINARY_INV)
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_16U)
    # Store transformed centroids
    transformed_centroids = []
    # Create an output image, copying from the original binary image
    output_image = np.copy(gray_image)
    # Convert output image to color if it's not already
    if len(output_image.shape) == 2:  # Only width and height
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    # Mark the center of each connected component
    for i in range(1, num_labels):  # Skip background
        # Scale statistics to match the size of the original image
        x = int(stats[i][0] * scale_x)
        y = int(stats[i][1] * scale_y)
        w = int(stats[i][2] * scale_x)
        h = int(stats[i][3] * scale_y)
        centroid = (int(centroids[i][0] * scale_x), int(centroids[i][1] * scale_y))
        transformed_centroids.append(centroid)
        cv2.circle(output_image, centroid, 2, (0, 0, 255), -1)  # Mark with a red circle
        # Draw bounding box
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return transformed_centroids, output_image

def segment_image(image_index, scale_x, scale_y):

    # Loads image in grayscale using image indexing from for loop
    gray_image = cv2.imread(f'TVHeads/test/{image_index}.png', cv2.IMREAD_GRAYSCALE)

    
    centers, output_image = process_image(f'TVHeads/test/{image_index}.png', scale_x, scale_y)

    output_size = (100, 100)  # Size of the output blocks
    for i, centroid in enumerate(centers):
        cx, cy = centroid
        # Calculate the coordinates of the region to be cropped
        x_start = cx - output_size[0] // 2
        y_start = cy - output_size[1] // 2
        x_end = x_start + output_size[0]
        y_end = y_start + output_size[1]

        # Create a fully black background
        image = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)

        # Calculate the coordinates to copy from the original image
        copy_x_start = max(x_start, 0)
        copy_y_start = max(y_start, 0)
        copy_x_end = min(x_end, gray_image.shape[1])
        copy_y_end = min(y_end, gray_image.shape[0])

        # Calculate the position to copy into the output image
        output_x_start = copy_x_start - x_start
        output_y_start = copy_y_start - y_start

        # Copy the required region from the original image
        image[output_y_start:output_y_start + copy_y_end - copy_y_start,
        output_x_start:output_x_start + copy_x_end - copy_x_start] = gray_image[copy_y_start:copy_y_end,
                                                                     copy_x_start:copy_x_end]
        # Set a unique filename for each image
        filename = f"TVHeads/segments/{image_index}_{i + 1}.jpg"  # Start numbering from 1
        # Save the image
        cv2.imwrite(filename, image)

scale_x = 10
scale_y = 10

for image_index in range(1, 226):  # range(1, 226) generates numbers from 1 to 225
    segment_image(image_index, scale_x, scale_y)