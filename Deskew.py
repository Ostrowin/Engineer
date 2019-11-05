# import the necessary packages
import numpy as np
import cv2
 
def Deskew(image): 
# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
    gray = cv2.bitwise_not(image)
     
# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
     
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
    if angle < -45:
            angle = -(90 + angle)
     
# otherwise, just take the inverse of the angle to make
# it positive
    else:
            angle = -angle

            # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    kernel = np.ones((2,2), np.uint8)
    erosion = cv2.erode(rotated, kernel, 1)
    dilated = cv2.dilate(rotated, kernel, 1)

# show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    return rotated
