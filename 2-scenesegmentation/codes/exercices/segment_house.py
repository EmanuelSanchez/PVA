import numpy as np
import cv2
import matplotlib.pyplot as plt


imBGR = cv2.imread("../ressources/house.jpg")

# Convert to BGR to HSV
imHSV = cv2.cvtColor(imBGR, cv2.COLOR_BGR2HSV)

# plt.figure()
# plt.imshow(imHSV[..., ::-1])

# Select a sub part of the image:

# brick
# x1 = 642
# y1 = 291
# x2 = 691
# y2 = 368
x1 = 635
y1 = 300
x2 = 698
y2 = 371

patchBrick = imHSV[y1:y2, x1:x2]
imBGRSegments = cv2.rectangle(imBGR.copy(), (x1,y1), (x2,y2), [0,255,0], 2)

# grass
# x1 = 900
# y1 = 750
# x2 = 950
# y2 = 800
x1 = 780
y1 = 750
x2 = 1065
y2 = 805

patchGrass = imHSV[y1:y2:,x1:x2]
imBGRSegments = cv2.rectangle(imBGRSegments, (x1,y1), (x2,y2), [0,255,0], 2)

# roof
# x1 = 1000
# y1 = 210
# x2 = 1050
# y2 = 250
x1 = 1005
y1 = 210
x2 = 1065
y2 = 255

patchRoof = imHSV[y1:y2:,x1:x2]
imBGRSegments = cv2.rectangle(imBGRSegments, (x1,y1), (x2,y2), [0,255,0], 2)

# path
# x1 = 750
# y1 = 820
# x2 = 800
# y2 = 860
x1 = 500
y1 = 820
x2 = 850
y2 = 865

# sky
x1 = 10
y1 = 10
x2 = 500
y2 = 120

patchSky = imBGR[y1:y2:,x1:x2]

patchPath = imHSV[y1:y2:,x1:x2]
imBGRSegments = cv2.rectangle(imBGRSegments, (x1,y1), (x2,y2), [0,255,0], 2)

fig = plt.figure()
plt.title('Sub parts')
imgBrick = fig.add_subplot(2,2,1)
plt.imshow(patchBrick[..., ::-1])
plt.axis('off')
imgGrass = fig.add_subplot(2,2,2)
plt.imshow(patchGrass[..., ::-1])
plt.axis('off')
imgRoof = fig.add_subplot(2,2,3)
plt.imshow(patchRoof[..., ::-1])
plt.axis('off')
imgPath = fig.add_subplot(2,2,4)
plt.imshow(patchPath[..., ::-1])
plt.axis('off')

plt.figure()
plt.title('Sub parts in original image')
plt.imshow(imBGRSegments[..., ::-1])
plt.axis('off')

# Build a histogram:
histBrick = cv2.calcHist([patchBrick], channels=[0, 1], mask=None, histSize=[179, 255], ranges=[0, 179, 0, 255])
histGrass = cv2.calcHist([patchGrass], channels=[0, 1], mask=None, histSize=[179, 255], ranges=[0, 179, 0, 255])
histRoof = cv2.calcHist([patchRoof], channels=[0, 1], mask=None, histSize=[179, 255], ranges=[0, 179, 0, 255])
histPath = cv2.calcHist([patchPath], channels=[0, 1], mask=None, histSize=[179, 255], ranges=[0, 179, 0, 255])



# plt.figure()
# plt.plot(hist)

# Compute the histogram back projection:
lhMapBrick = cv2.calcBackProject([imHSV], [0, 1], histBrick, [0, 179, 0, 255], 255)
lhMapGrass = cv2.calcBackProject([imHSV], [0, 1], histGrass, [0, 179, 0, 255], 255)
lhMapRoof = cv2.calcBackProject([imHSV], [0, 1], histRoof, [0, 179, 0, 255], 255)
lhMapPath = cv2.calcBackProject([imHSV], [0, 1], histPath, [0, 179, 0, 255], 255)

# We can use a filter, the we convolute with rectanguar kernel (filter)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# cv2.filter2D(lhMapBrick,-1,kernel,lhMapBrick)
# cv2.filter2D(lhMapGrass,-1,kernel,lhMapGrass)
# cv2.filter2D(lhMapRoof,-1,kernel,lhMapRoof)
# cv2.filter2D(lhMapPath,-1,kernel,lhMapPath)

fig = plt.figure()
plt.title('Segmentations')
fig.add_subplot(2,2,1)
plt.axis('off')
plt.imshow(lhMapBrick[..., ::-1])
fig.add_subplot(2,2,2)
plt.axis('off')
plt.imshow(lhMapGrass[..., ::-1])
fig.add_subplot(2,2,3)
plt.axis('off')
plt.imshow(lhMapRoof[..., ::-1])
fig.add_subplot(2,2,4)
plt.axis('off')
plt.imshow(lhMapPath[..., ::-1])
plt.axis('off')

# Don't forget to plot a BGR image
    # plt.figure()
    # plt.imshow(lhMapBrick[..., ::-1])
    # plt.axis('off')

lhmap_void = np.zeros_like(lhMapBrick)
maps = np.dstack([lhmap_void+0.1*255, lhMapBrick, lhMapGrass, lhMapRoof, lhMapPath])
labelling = np.argmax(maps, axis=2)

fig = plt.figure()
plt.axis('off')
plt.title('Results')
fig.add_subplot(1,2,1)
plt.imshow(imBGR[..., ::-1])
plt.axis('off')
fig.add_subplot(1,2,2)
plt.imshow(labelling, cmap="Accent")
plt.axis('off')
plt.show()
