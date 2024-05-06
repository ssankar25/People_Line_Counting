import cv2
import numpy as np

def connected_component_labeling(image):
    """
    Perform connected component labeling using 8-connectivity rule
    
    Scans the image and divides the image into components by scanning the 8 adjacent pixels
    and determining whether they are also foreground pixels

    Args:
        image (numpy array): A binary image where the objects are expected to be in white (255) and the background in black (0).

    Returns:
        A tuple containing:
            - num_labels (int): The number of unique labels (connected components found in the image plus one for the background).
            - labels (numpy array): An array the same size as the input image where each element has a value that corresponds 
              to the label of the connected component.
            - stats (numpy.ndarray): A matrix with stats for each label, including the bounding box, area, etc.
            - centroids (numpy.ndarray): The centroid (center of mass) for each labeled component.
    """

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
    return num_labels, labels, stats, centroids

def max_pooling_2d(image, kernel_size):
    """
    Apply max pooling operation to reduce the size of an image by downscaling using the maximum value in each window.

    Args:
        image (numpy array): The input image to be pooled.
        kernel_size: The dimensions (height, width) of the pooling window.

    Returns:
        pooled_image: The pooled image (numpy array) with reduced dimensions.
    """
        
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

def multilevel_threshold(hist):
    """
    Calculate the histogram and find a segmentation threshold that targets the brighter
    portions of the image by capturing a certain percentage of the brightest pixels.

    Args:
        hist: Histogram containing grayscale levels from 0 to 254

    Returns: 
        The first segmentation threshold that captures a specific percentage of pixels
    """
    total_pixels = np.sum(hist[:-1])  # Consider all pixels except the last bin (255)
    target_percentage = 0.975  # Target the brightest 95% of pixels, adjust as needed
    target_pixels = total_pixels * target_percentage

    cumulative_sum = 0
    for i in range(254, -1, -1):  # Start from the brightest and move to darker
        cumulative_sum += hist[i]
        if cumulative_sum >= total_pixels - target_pixels:
            return i  # Return the threshold that captures the brightest pixels

    return 0  # Default to the lowest threshold if nothing suitable is found


def process_frame(frame, scale_x=10, scale_y=10):
    """
    Process the frame to prepare it for object detection.
    Converts image to grayscale, applying Gaussian blur, pooling, thresholding, and performing morphological operations
    to reduce noise.

    Args:
        frame (numpy array): The original image frame from a video.
        scale_x (int): The downscale factor for width used during pooling.
        scale_y (int): The downscale factor for height used during pooling.

    Returns:
        dict: A dictionary with centroids of detected components keyed by their labels.
    """

    # Convert frame to grayscale if it is in color
    if len(frame.shape) == 3:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = frame

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)

    # Perform max pooling to reduce resolution while preserving features
    pooled_image = max_pooling_2d(blurred, (scale_x, scale_y))

    # Calculate histogram for threshold determination
    hist = np.bincount(pooled_image.ravel(), minlength=256)
    hist[-1] = 0  # Ignore the last bin
    threshold_value = multilevel_threshold(hist)

    # Apply binary thresholding to segment the image
    _, binary_image = cv2.threshold(pooled_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)  # Adjust the size of the kernel as needed
    binary_image = cv2.erode(binary_image, kernel, iterations=2)  # Erode to remove small artifacts
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)  # Dilate to restore object size and fill holes

    # Use connected components to identify and label distinct blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)

    # Adjust centroids to match the original image scale and return them
    return {
        i: (int(centroids[i][0] * scale_x), int(centroids[i][1] * scale_y))
        for i in range(1, num_labels)  # Skip the background component
    }

def update_tracking(centroids, tracks, line_entry_y, line_exit_y, head_count):
    """
    Update the tracking information for each detected centroid and adjust the head count based on crossing predefined lines.

    A head center must pass the bottom line and then the top line to consider entering the room (head count incremented).
    Passing the top line and then the bottom line is consiered exiting (head count decremented).

    Args:
        centroids: Dictionary of current frame centroids.
        tracks: Dictionary maintaining track history for each object.
        line_entry_y: The y-coordinate of the entry line.
        line_exit_y: The y-coordinate of the exit line.
        head_count: The current count of heads or objects that have crossed the lines.

    Returns:
        Updated head count after processing the current frame's movements.
    """

    current_ids = set(tracks.keys())
    detected_ids = set(centroids.keys())

    for cid in current_ids - detected_ids:
        del tracks[cid]

    for cid, pos in centroids.items():
        x, y = pos
        if cid in tracks:
            track = tracks[cid]
            prev_y = track['pos'][-1][1]


            track['pos'].append((x, y))

            if prev_y >= line_entry_y and y < line_entry_y:
                track['crossed_entry_up'] = True
            
            if track['crossed_entry_up'] and prev_y >= line_exit_y and y < line_exit_y:
                head_count -= 1
                track['crossed_entry_up'] = False

            if prev_y <= line_exit_y and y > line_exit_y:
                track['crossed_exit_down'] = True

            if track['crossed_exit_down'] and prev_y <= line_entry_y and y > line_entry_y:
                head_count += 1
                track['crossed_exit_down'] = False
                
            # Reset crossed_entry_up if crossing from above entry line to below
            if track['crossed_entry_up'] and prev_y <= line_entry_y and y > line_entry_y:
                track['crossed_entry_up'] = False

        else:     

                tracks[cid] = {
                    'pos': [(x, y)],
                    'crossed_entry_up': False,
                    'crossed_exit_down': False
                 }

    return head_count


def draw_scene(frame, centroids, line_entry_y, line_exit_y, head_count):
    """
    Draw the visualization of the current frame including lines, centroids, and head count.

    Args:
        frame (numpy array): The current frame to draw on.
        centroids: Dictionary containing centroids of detected objects to be drawn.
        line_entry_y: The y-coordinate of the entry line.
        line_exit_y: The y-coordinate of the exit line.
        head_count: The current head count to display.

    Returns:
        frame: The frame (numpy array) with drawings overlayed.
    """

    # Draw entry and exit lines
    cv2.line(frame, (0, line_entry_y), (frame.shape[1], line_entry_y), (0, 255, 0), 2)
    cv2.line(frame, (0, line_exit_y), (frame.shape[1], line_exit_y), (0, 0, 255), 2)

    # Draw centroids and bounding boxes
    for idx, (x, y) in centroids.items():
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        cv2.rectangle(frame, (x-10, y-10), (x+10, y+10), (255, 255, 0), 2)

    # Display head count
    cv2.putText(frame, f"Head count: {head_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# Initialize variables
head_count = 0
tracks = {}
line_entry_y = 150
line_exit_y = 70


# Open the video file
cap = cv2.VideoCapture('output_video.avi')

# Frame counter
frame_count = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    # Process every fifth frame
    if frame_count % 5 != 0:
        continue

    # Process the frame
    centroids = process_frame(frame)

    # Update head count based on centroids
    head_count = update_tracking(centroids, tracks, line_entry_y, line_exit_y, head_count)

    # Draw the current scene
    output_frame = draw_scene(frame, centroids, line_entry_y, line_exit_y, head_count)

    cv2.imshow('Head Counting', output_frame)
    if cv2.waitKey(250)  & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
