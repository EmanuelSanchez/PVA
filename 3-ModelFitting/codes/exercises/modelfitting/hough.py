import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.ion()

from modelfitting import toolbox

# dataset = '/Data/Databases/kitti/odometry/01/image_2'
dataset = '../ressources/kitti/odometry/01/image_2/'

frameIdx = 0
while True:
    print("Frame #%d" % frameIdx)

    imgName = "%s/%06d.png" % (dataset, frameIdx)
    imBGR = cv2.imread(imgName)
    if imBGR is None:
        print("File '%s' do not exist. The end ?" % imgName)
        break

    imGRAY = cv2.cvtColor(imBGR, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imGRAY, 100, 200, apertureSize=3)

    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(edges / 255.)

    # lines = cv2.HoughLines(edges, 1, 2*np.pi / 180, 130, min_theta=-0.9*np.pi/2, max_theta=0.9*np.pi/2)
    # lines = lines[:, 0]

    lines = np.zeros((0, 2))
    linesLeft = cv2.HoughLines(edges, 1, 2*np.pi / 180, 100, min_theta=-0.9*np.pi/2, max_theta=0)
    linesRight = cv2.HoughLines(edges, 1, 2*np.pi / 180, 100, min_theta=0, max_theta=0.9*np.pi/2)
    if linesLeft is not None:
        lines = np.concatenate((lines, linesLeft[:, 0][:10]))
    if linesRight is not None:
        lines = np.concatenate((lines, linesRight[:, 0][:10]))
    np.random.shuffle(lines)

    # lines = cv2.HoughLinesP(edges, 1, 2*np.pi / 180, 130)

    plt.subplot(2, 1, 2)
    plt.imshow(imBGR[..., ::-1] / 255.)

    if lines is not None:
        print("%d lines detected" % len(lines))
        for rho, theta in lines:
            x1, y1, x2, y2 = toolbox.line_pts(rho, theta)
            plt.plot((x1, x2), (y1, y2), linewidth=2)
            # plt.text(max(min(x1, imBGR.shape[1]), 0), max(min(y1, imBGR.shape[0]), 0), "theta %.1f" % theta)

            # if True:
            # idx = np.random.random_integers(0, len(lines[:, 0])-1)
            # rho2, theta2 = lines[:, 0][randIdx]
            for rho2, theta2 in lines:
                _x1, _y1, _x2, _y2 = toolbox.line_pts(rho2, theta2)
                inter = toolbox.seg_intersect([x1, y1], [x2, y2], [_x1, _y1], [_x2, _y2])
                if inter is not None:
                    plt.plot(inter[0], inter[1], '+', markerSize=25)
    else:
        print("No lines detected")

    plt.xlim(0, imBGR.shape[1])
    plt.ylim(imBGR.shape[0], 0)
    plt.draw()

    plt.waitforbuttonpress(0.02)
    frameIdx += 1

plt.show()