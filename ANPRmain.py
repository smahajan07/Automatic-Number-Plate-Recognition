import cv2
import numpy as np

mask_list =[]
validated_masklist = []
final_maskist = []
index = []
cropped_images = []
valid_license_plate = []

def validate(cnt):
    # to validate a contour of appropriate aspect ratio and area
    rect = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    output = False
    width = rect[1][0]
    height = rect[1][1]

    if ((width != 0) and (height != 0)):
        if (((height / width > 3) and (height > width)) | ((width / height > 3) and (width > height))):
            if ((height * width < 18000) and (height * width > 2000)):
                output = True
    return output

def generate_seeds(centre, width, height):
    # fill seeds
    minsize = int(min(width, height))
    seed = [None]*10
    for i in range(10):
        random_int1 = np.random.randint(1000)
        random_int2 = np.random.randint(1000)
        seed[i] = (centre[0] + random_int1%int(minsize/2) - int(minsize/2),centre[1] + random_int2%int(minsize/2) - int(minsize/2) )
    return seed

def generate_masks(img, seed_point):
    h = image.shape[0]
    w = image.shape[1]

    mask = np.zeros((h+2 , w+2), np.uint8)

    lodiff = 50
    updiff = 50
    connectivity = 4
    newmaskval = 255

    flags =  connectivity + (newmaskval<<8) + cv2.cv.CV_FLOODFILL_FIXED_RANGE + cv2.cv.CV_FLOODFILL_MASK_ONLY
    cv2.floodFill(img, mask, seed_point, (255,0,0), (lodiff, lodiff, lodiff), (updiff,updiff,updiff), flags)

    return mask

def rmsdiff(im1, im2):
    #calculate difference between images
    diff = im1-im2
    output = False
    if np.sum(abs(diff))/float(min(np.sum(im1), np.sum(im2)))<0.01:
        output = True
    return output

file_path = 'data/plate3.jpg'
image = cv2.imread(str(file_path))
cv2.imshow('Original', image)
cv2.waitKey(0)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_blur = cv2.GaussianBlur(image_gray, (5,5), 0)

image_sobel = cv2.Sobel(image_blur, cv2.CV_8U, 1,0, ksize=3)
# cv2.imshow('Edges', image_sobel)
# cv2.waitKey(0)

rect , image_thresh = cv2.threshold(image_sobel, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#cv2.imshow('Threshold', image_thresh)
#cv2.waitKey(0)

element = cv2.getStructuringElement(cv2.MORPH_RECT, (23,5))
image_morph = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, element)
#cv2.imshow('Morphological operations', image_morph)
#cv2.waitKey(0)

contour, hierarchy = cv2.findContours(image_morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#find all contours
for cnt in contour:
    rect = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0,255,0), 2)

#cv2.imshow('Original with contours', image)
#cv2.waitKey(0)

#find valid contours of particular aspect ratio and area
for cnt in contour:
    if validate(cnt):
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0,0,255), 2)

cv2.imshow('Original with right contours', image)
cv2.imwrite('testing/Contours.tif', image)
cv2.waitKey(0)

image_mask = cv2.imread(str(file_path))
#cv2.imshow('Original with mask - Copy', image_mask)
#cv2.waitKey(0)

for cnt in contour:
    if validate(cnt):
        rect = cv2.minAreaRect(cnt)
        centre = (int(rect[0][0]), int(rect[0][1]))
        width = rect[1][0]
        height = rect [1][1]
        seeds = generate_seeds(centre, width, height)

        for seed in seeds:
            cv2.circle(image, seed, 1, (0,0,255), -1)
            mask = generate_masks(image_mask, seed)
            mask_list.append(mask)

for mask in mask_list:
    contour = np.argwhere(mask.transpose() == 255)
    if validate(contour):
        validated_masklist.append(mask)

try:
    assert (len(validated_masklist) != 0)
except AssertionError:
    print('No valid masks could be generated')

#check redundancy in images
for i in range(len(validated_masklist)-1):
    for j in range(i+1, len(validated_masklist)):
        if rmsdiff(validated_masklist[i], validated_masklist[j]):
            index.append(j)

for mask_no in list(set(range(len(validated_masklist)))- set(index)):
    final_maskist.append(validated_masklist[mask_no])

for mask in final_maskist:
    contour = np.argwhere(mask.transpose() == 255)
    rect = cv2.minAreaRect(contour)
    width = int(rect[1][0])
    height = int(rect[1][1])
    centre = (int(rect[0][0]), int(rect[0][1]))
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)

    if ((width/float(height)>1)):
        cropped_image = cv2.getRectSubPix(image_mask, (width,height), centre)
    else:
        cropped_image = cv2.getRectSubPix(image_mask, (height, width), centre)

    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    cropped_image = cv2.equalizeHist(cropped_image)
    cropped_image = cv2.resize(cropped_image, (260,63))
    cropped_images.append(cropped_image)

#self trained Haar classifier with approx 130 positive images and 50 negative images
number_plate = cv2.CascadeClassifier('output.xml')

index = 0
max_area = 2000

for i in range(len(cropped_images)):
    plate = number_plate.detectMultiScale(cropped_images[i], 1.3, 5)
    if len(plate)>0:
        x,y,w,h = plate[0]
    area = w*h
    if area<max_area:
        max_area=area
        index = i

valid_license_plate.append(cropped_images[index])
cv2.imwrite('testing/valid_license_plate.tif', valid_license_plate[0])

cv2.waitKey(0)
cv2.destroyAllWindows()