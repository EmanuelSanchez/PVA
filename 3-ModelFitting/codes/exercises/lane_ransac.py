import numpy as np
import cv2
import matplotlib.pyplot as plt
from modelfitting import toolbox

plt.ion()


# datax0, datay0 are the dataset x, y
# The line is defined with two points (x1, y1) (x2, y2)
def displayLineModel(datax0, datay0, x1, y1, x2, y2, c):
    plt.figure(1)
    plt.clf()
    # plt.scatter(x0, y0, s=20*np.minimum(1./dist, 1))
    plt.scatter(datax0, datay0, s=1)
    # plt.scatter(datax0[inliersMask], datay0[inliersMask], c=c)
    plt.plot([x1, x2], [y1, y2], 'r', marker='o')
    plt.xlim(0, imBGR.shape[1])
    plt.ylim(imBGR.shape[0], 0)

dataset = '../ressources/kitti/odometry/01/image_2/'
frameIdx = 40
while True:
    print("Frame #%d" % frameIdx)

    imgName = "%s/%06d.png" % (dataset, frameIdx)
    imBGR = cv2.imread(imgName)
    if imBGR is None:
        print("File '%s' does not exist. The end ?" % imgName)
        break

    imGRAY = cv2.cvtColor(imBGR, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(imGRAY, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edges = tophat>50

    (ey, ex) = np.where(edges > 0)  # Get all non-zero coordinates
    y0 = ey
    x0 = ex

    # Do the ransac,
    Titer = 100
    lines = np.zeros((0, 4))  # Fill that with the lines (x1, y1) (x2, y2) detected
    for i in range(Titer):
        # FILL HERE
        # FILL HERE
        # FILL HERE

        y1, x1, y2, x2 = 0, 0, 0, 0
        displayLineModel(x0, y0, x1, y1, x2, y2, 'r')
        # plt.waitforbuttonpress()

    # Display the lines
    plt.figure(1)
    plt.clf()
    plt.imshow(imBGR[..., ::-1]//2)
    plt.scatter(x0, y0, color='r', s=1)
    # plt.scatter(x0[bestModelInliersMask], y0[bestModelInliersMask], c='r')
    for i in range(len(lines)):
        y1, x1, y2, x2 = lines[i]
        plt.plot([x1, x2], [y1, y2], 'g', marker='o', linewidth=2)
    plt.xlim(0, imBGR.shape[1])
    plt.ylim(imBGR.shape[0], 0)

    # Use either of the following functions:
    plt.waitforbuttonpress()  # Wait for a button (click, key) to be pressed
    frameIdx += 1
    # plt.waitforbuttonpress()  # Pause for 0.02 second. Useful to get an animation