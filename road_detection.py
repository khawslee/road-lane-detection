import matplotlib.pylab as plt
import cv2
import numpy as np

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

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

# Read road image
image = cv2.imread('./data/road.jpg')
# Convert image to RGB for Windows viewing
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the height of image
height = image.shape[0]
# Get the width of image
width = image.shape[1]

# Define the region of interest
region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height)
]

# Convert road image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# GaussianBlur the image
blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
# Canny edge detect on the blurred road image
canny_image = auto_canny(blurred)
# Crop image base on region of interest
cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
# Find each line using HoughLinesP function
lines = cv2.HoughLinesP(cropped_image,
                        rho=2,
                        theta=np.pi / 180,
                        threshold=160,
                        lines=np.array([]),
                        minLineLength=40,
                        maxLineGap=100)
# Draw the lines on the road
image_with_lines = draw_the_lines(image, lines)

# Show the image with lines
plt.imshow(image_with_lines)
plt.show()
