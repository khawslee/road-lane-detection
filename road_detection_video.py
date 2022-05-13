import cv2
import numpy as np
import threading, time
import queue

# Define region of interest
def region_of_interest(img, vertices):
    # Create a blank image
    mask = np.zeros_like(img)
    # Define black as mask color
    match_mask_color = 255
    # Build the poly mask using mask color
    cv2.fillPoly(mask, vertices, match_mask_color)
    # Bitwise image with mask to remove unwanted area
    masked_image = cv2.bitwise_and(img, mask)
    # Return the masked image
    return masked_image

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

# Draw lines on image
def draw_the_lines(img, lines):
    # Create RGB blank image
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Draw each line on blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Blend line image over original image
    img_weight = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img_weight


def processing(image):
    # Set the vertices region
    height, width = image.shape[:2]
    bottom_left = [width * 0.1, height]
    top_left = [width * 0.45, height * 0.45]
    bottom_right = [width * 0.8, height]
    top_right = [width * 0.48, height * 0.45]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    # Convert road image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # GaussianBlur the image
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    # Canny edge detect on the grayscale road image
    canny_image = auto_canny(blurred)
    # Crop image base on region of interest
    cropped_image = region_of_interest(canny_image, vertices)
    # Find each line using HoughLinesP function
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    # Draw the lines on the road
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

# Multithread Read File
def readFile():
    global finished

    while not finished:
        # Read 1 frame from video
        ret, frame = cap.read()
        if not ret:
            finished = True

        while not finished:
            try:
                # Store frame into buffer
                input_buffer.put(frame, timeout=1)
                break
            except queue.Full:
                pass

# Multithread processing video frame
def processingFile():
    global finished
    global frame

    cv2.namedWindow(window_name)

    start_time = time.time()
    frame_number = 0
    total_wait_time = 0
    while True:
        try:
            # Get one frame from buffer
            frame = input_buffer.get(timeout=1)
           # Perform road lane detection
            frame = processing(frame)
            # Show the video frame
            cv2.imshow(window_name, frame)
        except queue.Empty:
            if finished:
                break
        # Add delay to sync playback with source video FPS
        wait_time = (start_time + frame_number * time_frame) - time.time()
        if wait_time > 0:
            total_wait_time += wait_time
            time.sleep(wait_time)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            finished = True
            print("Playback terminated.")
            break
        frame_number += 1
    end_time = time.time()

    print("Video FPS = %0.3f" % fps)
    print("Frames rendered = %d (includes repeats during pause)" % frame_number)
    print("Time taken = %0.3f seconds" % (end_time - start_time))
    print("Actual FPS = %0.3f" % (frame_number / (end_time - start_time)))
    print("Total wait time %0.3f" % total_wait_time)
    print("Average wait time %0.3f" % (total_wait_time / frame_number))

# Open video file using ConcurrentVideoCapture helper
cap = cv2.VideoCapture('./data/lane1.mp4')
# Define frame buffer for video
input_buffer = queue.Queue(20)

# Store the actual FPS
fps = cap.get(cv2.CAP_PROP_FPS)
# fps in time - seconds
time_frame = 1.0 / fps

# Set video playback finish variable
finished = False
#Set the window name
window_name = 'Video File'

# Thread read the video file
tReadFile = threading.Thread(target=readFile)
# Thread process the video frame
tProcessingFile = threading.Thread(target=processingFile)

# Start both thread
tReadFile.start()
tProcessingFile.start()
# Wait until thread termination
tProcessingFile.join()
tReadFile.join()

# Release the video capture
cap.release()
# Close all opencv windows
cv2.destroyAllWindows()
