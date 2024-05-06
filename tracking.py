import cv2
import numpy as np

# Define the kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Open the video file
cap = cv2.VideoCapture('test_video.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay_time = int(1000 / fps)

# Define the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (240, 180))

ret, frame1 = cap.read()

# Convert frame to grayscale and perform preprocessing
gray1 = (cv2.subtract(frame1[:, :, 2], frame1[:, :, 0]).astype(np.uint8))
gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)

_, binary_image1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY_INV)

while True:
    ret, frame2 = cap.read()

    # Convert frame to grayscale and perform preprocessing
    gray2 = (cv2.subtract(frame2[:, :, 2], frame2[:, :, 0]).astype(np.uint8))
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    _, binary_image2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY_INV)

    # Compute the difference between two consecutive frames
    diff1 = cv2.subtract(binary_image1, binary_image2)
    erosion1 = cv2.erode(diff1, kernel, iterations=1)
    dilation1 = cv2.dilate(erosion1, kernel, iterations=1)

    _, binary_image = cv2.threshold(dilation1, 10, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []

    # Filter out small contours and extract bounding rectangles
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:
            x, y, w, h = cv2.boundingRect(contour)
            rectangles.append((x, y, w, h, area))

    filtered_rectangles = []

    # Filter out contained rectangles
    for rect1 in rectangles:
        x1, y1, w1, h1, area1 = rect1
        is_contained = False
        for rect2 in rectangles:
            x2, y2, w2, h2, area2 = rect2

            if (x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2 and area1 < area2):
                is_contained = True
                break

        if not is_contained:
            filtered_rectangles.append(rect1)

    # Draw rectangles on the frame
    for rect in filtered_rectangles:
        x, y, w, h, _ = rect
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame2)

    # Display the frame
    cv2.imshow('Frame', frame2)

    # Update the previous grayscale frame
    gray1 = gray2

    if cv2.waitKey(delay_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()