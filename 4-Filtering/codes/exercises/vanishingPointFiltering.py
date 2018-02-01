import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
import toolbox as toolbox
import particlefilter as pf


plt.ion()

dataset = '../ressources/kitti/odometry/01/image_2/'
hough_thresh = 50

# Read first frame to extract width, height
imgName = "%s/%06d.png" % (dataset, 0)
imBGR = cv2.imread(imgName)
w, h = imBGR.shape[1], imBGR.shape[0]
def newParticles(N):
    # particles = None
    # Generate new particles
    return np.random.random((N, 2))*[h, w]  # Build an Nx2 array with row being particles, cols the (y,x) coordinate
    # return particles

def motionUpdate(particles):
    # Do the motion update
    return particles


def resamplingFun(particles, weights, resampleN):
    # Modify this function to resample N particles using the weights
    # INFO: particles[i] has weights weights[i]
    # resampleN = []
    # total = sum(fitness(p) for p in population)
    # top = 0
    # for p in particles:
    #     f = weights(p)/total
    #     wheel.append((top, top+f, p))
    #     top += f
    # return wheel
    # print(np.random.ranfloat)
    newParticles = particles
    return newParticles


def scoringParticles(z, particles):
    # Compute the scores array that correspond to the particles and the observation (set of candidates vanishing points) z
    # scores = np.zeros_like(len(particles) if particles is not None else 0.) + 1.
    scores= np.zeros(len(particles))
    for i in range(len(particles)):
        p = particles[i] # [x, y] of particle
        for j in range(len(z)):
            _i = z[j] # [x, y] of interserction

            dst = np.sqrt(np.power(_i[0] - p[0], 2) + np.sqrt(np.power(_i[1] - p[1], 2)))

            scores[i] = np.exp(-np.power(dst,2) / (2*np.power(1, 2.)))
    # scores = np.sum(cdist(z, particles), axis=0)
    # print(scores)
    # print(_scores)
    return scores

# Initialize the particle filter. (100 particles is enough)
pf.init(newParticles, 100)

frameIdx = 500
while True:
    print("Frame #%d" % frameIdx)

    imgName = "%s/%06d.png" % (dataset, frameIdx)
    imBGR = cv2.imread(imgName)
    if imBGR is None:
        print("File '%s' do not exist. The end ?" % imgName)
        break

    imGRAY = cv2.cvtColor(imBGR, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imGRAY, 100, 200, apertureSize=3)  # Compute some edges

    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(edges / 255., cmap="gray")

    # Detect lines using left and right criteria
    # Lines are stored as Mx2 matrix. With each row: (rho, theta)
    lines = np.zeros((0, 2))
    linesLeft = cv2.HoughLines(edges, 1, 1*np.pi / 180, hough_thresh, min_theta=-0.9*np.pi/2, max_theta=0)
    linesRight = cv2.HoughLines(edges, 1, 1*np.pi / 180, hough_thresh, min_theta=0, max_theta=0.9*np.pi/2)
    if linesLeft is not None:
        lines = np.concatenate((lines, linesLeft[:, 0][:10]))
    if linesRight is not None:
        lines = np.concatenate((lines, linesRight[:, 0][:10]))
    np.random.shuffle(lines)  # Shuffle the lines to avoid ordering


    # Array that contains the vanishing candidates (intersection between lines)
    vanishingCands = np.zeros((0, 2)) # (y,x)
    print(vanishingCands.shape)

    # Compute all the vanishing candidates (intersection between two lines)
    if lines is not None:
        print("%d lines detected" % len(lines))

        # Compute intersection between lines (use toolbox.seg_intersect)
        # and fill the "vanishingCands" array (using a np.vstack())
        for rho, theta in lines:
            x1a, y1a, x2a, y2a = toolbox.line_pts(rho, theta)
            plt.plot((x1a, x2a), (y1a, y2a), linewidth=1)

            for rho2, theta2 in lines:
                # Compute interaction between (rho, theta) and (rho2, theta2)
                x1b, y1b, x2b, y2b = toolbox.line_pts(rho2, theta2)
                # intersection = toolbox.seg_intersect([x1a, y1a], [x2a, y2a], [x1b, x2b], [x2b, y2b])
                intersection = toolbox.seg_intersect([x1a, y1a], [x2a, y2a], [x1b, y1b], [x2b, y2b])

                # Add the intersection to the vanishing candidates:
                #   np.vstack([vanishingCands, intersection])
                if intersection is not None: vanishingCands = np.vstack([vanishingCands, intersection])
    else:
        print("No lines detected")

    # Display the vanishing candidates
    plt.scatter(vanishingCands[:, 0], vanishingCands[:, 1], marker='o', s=100)

    plt.xlim(0, imBGR.shape[1])
    plt.ylim(imBGR.shape[0], 0)

    plt.subplot(2, 1, 2)
    plt.imshow(imBGR[..., ::-1] / 255.)

    # Weight the particles using a scoring function you want.
    pf.weighting(vanishingCands, scoringParticles)  # You have to update the scoring function

    # Display stuff
    if pf.particles is not None:
        plt.scatter(pf.particles[:, 1], pf.particles[:, 0], marker='+', s=2 + 200*pf.weights/pf.weights.max(), c='r', edgecolor='none')  # Disply
    plt.xlim(0, imBGR.shape[1])
    plt.ylim(imBGR.shape[0], 0)
    plt.draw()

    # Resample the particles
    pf.resampling(resamplingFun)

    # Apply the motion update
    pf.motionUpdate(motionUpdate)

    # plt.waitforbuttonpress(0.02)
    plt.waitforbuttonpress()
    frameIdx += 1

plt.show()
