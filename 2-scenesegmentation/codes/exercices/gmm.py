import cv2
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


imBGR = cv2.imread("../ressources/landscape_hdr.jpg")

# Segment that image in N different cluster using a Gaussian mixture model (e.g. N=17)
# 1. Create the gaussian mixture:
# g = mixture.GaussianMixture(n_components=N, max_iter=100)

N = 17
g = mixture.GaussianMixture(n_components=N, max_iter=100)

# 2. Fit the image observed to the GMM:
# g.fit(obs)

print(imBGR.shape)
print(imBGR[:,:,0].shape)
print(imBGR[:,:,0].shape)

newIm = imBGR[:,:,0].reshape(imBGR.size, 1)
g.fit(newIm)

# means and cov of the gmm are stored in:
# g.means_
# g.covariances_
print('aca')
print(imBGR.shape[2])
cluster = g.predict(newIm)
cluster = cluster.reshape(imBGR.shape[0])

means = g.means_
covariances = g.covariances_

# print(means)

# Use the Probability Distribution Function (PDF) to compute which pixel belong to which Gaussian
#  and assign the mean of the gaussian to the pixels



# plot a BGR imge
plt.figure()
plt.imshow(imBGR[..., ::-1])
plt.axis('off')
