import cv2
import numpy as np

character = []
character_cnt = []
black_pixels = [0,0,0]
pad = 20

def validate_char(cnt):
    # to validate a contour of appropriate aspect ratio and area
    rect2 = cv2.minAreaRect(cnt)
    box2 = cv2.cv.BoxPoints(rect2)
    box2 = np.int32(box2)
    output = False

    width = rect2[1][0]
    height = rect2[1][1]
    area = int(height*width)

    if (width != 0) and (height != 0):
        if (((height / width > 1.3) & (height > width)) | (width/height > 1.3) & (width > height)):
            if ((area < 1700) & (area > 200)):
                if ((max(height, width) < 72) & (min(width,height) > 18)):
                    output = True
    return output

path = 'testing/valid_license_plate.tif'
img = cv2.imread(str(path))

imgblur = cv2.blur(img, (3,3))

_ , thresh_valid_plate = cv2.threshold(imgblur, 80, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow('Inverse Threshold Number plate', thresh_valid_plate)
cv2.imwrite('testing/inverse_threshold.tif', thresh_valid_plate)
thresh_valid_plate = cv2.cvtColor(thresh_valid_plate, cv2.COLOR_BGR2GRAY)

median_blur = cv2.medianBlur(thresh_valid_plate, 5)
#cv2.imshow('Median blur', median_blur)
#cv2.waitKey(0)

character_contours, hierarchy = cv2.findContours(thresh_valid_plate.copy() , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bounding_boxes = [cv2.boundingRect(c) for c in character_contours]
(character_contours, bounding_boxes) = zip(*sorted(zip(character_contours, bounding_boxes), key= lambda b:b[1][0], reverse = False))

for cnt in character_contours:
    if validate_char(cnt):
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255,255,255), 2)

        rect_char = cv2.minAreaRect(cnt)
        width_char = int(rect_char[1][0])
        height_char = int(rect_char[1][1])
        centre_char = (int(rect_char[0][0]), int(rect_char[0][1]))
        box_char = cv2.cv.BoxPoints(rect_char)
        box_char = np.int32(box_char)
        x,y,w,h = cv2.boundingRect(cnt)
        character_image = thresh_valid_plate.copy()[y:y+h, x:x+w]
        character.append(character_image)

cv2.imshow('Plate with bounding boxes', img)
cv2.imwrite('testing/plate_boundingBox.tif', img)

#segment characters
for i,char in enumerate(character):
    #cv2.imshow('Character ' + str(i), char)
    padImage = cv2.copyMakeBorder(char, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=black_pixels)
    cv2.imwrite('testing/Numbers/number'+ str(i) +'.png', padImage)

cv2.waitKey(0)
cv2.destroyAllWindows()

