# Luis Roldao - Universidad Simon Bolivar
# 30-Nov-2017
# In order to create the environment for running this code, remember to run the
# following command in your Anaconda command line:
# --> conda create --name 3dclass --channel ccordoba12 python=2.7 pcl python-pcl numpy matplotlib mayavi

import pcl
from mayavi import mlab
import numpy as np
import time
import datetime


# Read a .pcd file, just give the path to the file. The function will return the pointcloud as a numpy array.
def read_pcd_file(input_filename):
    return pcl.load(input_filename).to_array()


# Save your pointcloud as a .pcd file in order to use it in other # programs (cloudcompare for example).
def write_pcd_file(pointcloud, output_path):
    output_pointcloud = pcl.PointCloud()
    output_pointcloud.from_array(np.float32(pointcloud))
    output_pointcloud.to_file(output_path)
    return


# To visualize the passed pointcloud.
def viewer_pointcloud(pointcloud):
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.points3d(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], color=(0, 0, 0), mode='point')
    mlab.show()
    return


# To visualize two pointclouds (The original one and the one obtained after the Ransac normally) and the
# plane obtained by the Ransac all together.
def viewer_original_vs_ransac_pointcloud_vs_plane(ransac_pcl, original_pcl, plane_model):
    sensor_range = 120.0
    mlab.figure(bgcolor=(1, 1, 1))
    x, y = np.ogrid[-sensor_range+50:sensor_range+50:1, -sensor_range:sensor_range:1]
    mlab.points3d(original_pcl[:, 0], original_pcl[:, 1], original_pcl[:, 2], color=(0, 0, 0), mode='point')
    mlab.points3d(ransac_pcl[:, 0], ransac_pcl[:, 1], ransac_pcl[:, 2], color=(1, 0, 0), mode='point')
    mlab.surf(x, y, (-plane_model[3] - (plane_model[0]*x) - (plane_model[1]*y)) / plane_model[2],
              color=(0.8, 0.8, 1), opacity=0.3)
    mlab.show()
    return


# To visualize two pointclouds in the viewer.
def viewer_pointcloud1_vs_pointcloud2(pointcloud1, pointcloud2):
    sensor_range = 120.0
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.points3d(pointcloud1[:, 0], pointcloud1[:, 1], pointcloud1[:, 2], color=(0, 0, 0), mode='point')
    mlab.points3d(pointcloud2[:, 0], pointcloud2[:, 1], pointcloud2[:, 2], color=(1, 0, 0), mode='point')
    mlab.show()
    return


# Transform (rotate, translate) a PointCloud using the given transformation matrix.
def transform_pointcloud(transf_matrix, pointcloud):
    return np.delete(np.transpose(np.dot(transf_matrix,
                                         np.transpose(np.c_[pointcloud, np.ones(pointcloud.shape[0])]))), 3, axis=1)


def normalize_vector(vector):
    return vector/np.linalg.norm(vector)


# --------------------------------------------------------------------------------------------------------------------
# This is the function to complete, it should receive a pointcloud (numpy array [x, y, z],[x, y, z]...),
# the number of iterations of the Ransac and the threshold to be used. It should return a new pointcloud
# numpy array with the points extracted by the Ransac and a numpy array with the variables of the plane
# (A, B, C, D) - Remember that the equation of the plane Ax+By+Cz+D=0 defines the plane itself.
def random_sampling_consensus(pointcloud, numb_iterations, threshold):

    # FILL THE FUNCTION --------------------------------
    # auxiliary variable to save the best score
    best_score = 0

    for i in range(numb_iterations):
        # sample 3 random points
        sample = pointcloud[np.random.randint(0, pointcloud.shape[0], 3)]
        p1 = sample[0]
        p2 = sample[1]
        p3 = sample[2]

        # calculate a normal vector to the plane
        v = np.cross(p2-p1, p2-p3)

        try:
            # checks that 3 points are not collinear
            if (not v.any()):
                continue
            else:
                # calculate plane model
                x = 0
                y = 1
                z = 2

                A = (p3[y] - p2[y]) * (p1[z] - p2[z]) - (p3[z] - p2[z]) * (p1[y] - p2[y])
                B = (p3[z] - p2[z]) * (p1[x] - p2[x]) - (p3[x] - p2[x]) * (p1[z] - p2[z])
                C = (p3[x] - p2[x]) * (p1[y] - p2[y]) - (p3[y] - p2[y]) * (p1[x] - p2[x])
                D = (-1) * A * p1[x] - B * p1[y] - C * p1[z]

                xi = pointcloud[:, 0]
                yi = pointcloud[:, 1]
                zi = pointcloud[:, 2]

                # calculate distance from points to model
                d = np.abs( A*xi + B*yi + C*zi + D ) / np.sqrt( A*A + B*B + C*C )

                # calculate the score of the model (RANSAC)
                # score = d[d < threshold].shape[0]
                # calculate the score of the model (M-SAC)
                aux = threshold - d
                score = aux[aux > 0].shape[0]

                # save the best model
                if (score > best_score):
                    best_score = score
                    plane_model = (A, B, C, D)
                    # indexes of the cloud points that satisfy the threshold
                    indexes = d < threshold

        except AttributeError:
            continue

    ransac_pointcloud = pointcloud[indexes]
    # Return the requested variables
    return ransac_pointcloud, plane_model
# --------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------
# Given 2 plane models, calculate the rotation matrix that fits both planes
def calculate_rotation_matrix(plane_model1, plane_model2):

    # FILL THE FUNCTION --------------------------------
    # the values (A, B, C) of the model of the plane (Ax + By + Cz = 0)
    # are equivalent to a vector normal to the plane, but it is not necessary
    # unitary. So it must be normalized
    b = plane_model1[0:3] / np.linalg.norm(plane_model1[0:3])
    a = plane_model2[0:3] / np.linalg.norm(plane_model2[0:3])

    v = np.cross(a, b)
    c = np.dot(a, b)

    # calculate the skew-symmetric cross-product matrix of v
    v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # identity matrix with v_x size
    I = np.identity(v_x.shape[0])

    # calculate the Rotation Matrix
    R = I + v_x + v_x * v_x * (1 / (1 + c))

    # translation matrix
    t = np.zeros((R.shape[0],1))

    # calculate the transformation matrix
    zeros = np.zeros((1, R.shape[0]))
    transformation_matrix = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

    # Return the requested variables
    return transformation_matrix
# --------------------------------------------------------------------------------------------------------------------


def main():
    # Exercise 1 - Ransac to detect the Main Plane
    # pointcloud = read_pcd_file("../resources/pclexercise1.pcd")
    # viewer_pointcloud(pointcloud)
    # ransac_pointcloud, plane_model = random_sampling_consensus(pointcloud, 100, 0.2) # Fill the function
    # viewer_original_vs_ransac_pointcloud_vs_plane(ransac_pointcloud, pointcloud, plane_model)

    # Exercise 2 - Detect the Rotation Matrix from 2 frames
    numb_iter, threshold = 100, 0.2
    pointcloud1 = read_pcd_file("../resources/pcl1exercise2.pcd")
    pointcloud2 = read_pcd_file("../resources/pcl2exercise2.pcd")
    #viewer_pointcloud1_vs_pointcloud2(pointcloud1, pointcloud2)
    ransac_pointcloud1, plane_model1 = random_sampling_consensus(pointcloud1, numb_iter, threshold)
    ransac_pointcloud2, plane_model2 = random_sampling_consensus(pointcloud2, numb_iter, threshold)
    Transf_matrix = calculate_rotation_matrix(plane_model1, plane_model2) # Fill the function
    pointcloud2_fixed = transform_pointcloud(Transf_matrix, pointcloud2)
    print("ahi va")
    viewer_pointcloud1_vs_pointcloud2(pointcloud1, pointcloud2_fixed)

if __name__ == '__main__':
    main()
