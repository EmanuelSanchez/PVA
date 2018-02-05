import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.ion()


dataset = '../ressources/kitti/odometry/01/image_2/'
frameIdx = 0
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

    (y0, x0) = np.where(edges > 0)  # Get all non-zero coordinates

    # Detect lines and display them
    # How could you adjust to push the algorithm to detect road lanes ?
    lines = cv2.HoughLines(tophat, 1, np.pi/180, 300)

    # continue if not lines was detected
    try:
        if (not lines.any()):
            continue
    except AttributeError:
        continue

    # gets rho and theta values (Hogh Transform)
    rho = lines[:,0,0]
    theta = lines[:,0,1]

    # calculates two points to draw the line
    x1 = (rho * np.cos(theta) - 1000 * np.sin(theta)).astype(int)
    y1 = (rho * np.sin(theta) + 1000 * np.cos(theta)).astype(int)
    x2 = (rho * np.cos(theta) + 1000 * np.sin(theta)).astype(int)
    y2 = (rho * np.sin(theta) - 1000 * np.cos(theta)).astype(int)

    # Display the lines
    plt.figure(1)
    plt.clf()
    plt.imshow(imBGR[..., ::-1]//2)
    # plt.imshow(tophat[..., ::-1]//2)
    plt.scatter(x0, y0, color='r', s=1)
    for (x1, y1, x2, y2) in zip(x1, y1, x2, y2):
        plt.plot([x1, x2], [y1, y2], 'g', marker='o', linewidth=2)
    plt.xlim(0, imBGR.shape[1])
    plt.ylim(imBGR.shape[0], 0)

    # Use either of the following functions:
    # plt.waitforbuttonpress()  # Wait for a button (click, key) to be pressed
    frameIdx += 1
    # Use either of the following functions:
    plt.waitforbuttonpress()  # Wait for a button (click, key) to be pressed
    # plt.waitforbuttonpress(0.01)  # Pause for 0.02 second. Useful to get an animation
