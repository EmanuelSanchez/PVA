# Luis Roldao - Universidad Simon Bolivar
# 30-Nov-2017
# In order to create the environment for running this code, remember to run the
# following command in your Anaconda command line:
# --> conda create --name 3dclass --channel ccordoba12 python=2.7 pcl python-pcl numpy matplotlib mayavi

import pcl
from mayavi import mlab
import numpy as np
import time


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

    # Return the requested variables
    return ransac_pointcloud, plane_model
# --------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------
# Given 2 plane models, calculate the rotation matrix that fits both planes
def calculate_rotation_matrix(plane_model1, plane_model2):

    # FILL THE FUNCTION --------------------------------

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
    # numb_iter, threshold = 100, 0.2
    # pointcloud1 = read_pcd_file("../resources/pcl1exercise2.pcd")
    # pointcloud2 = read_pcd_file("../resources/pcl2exercise2.pcd")
    # viewer_pointcloud1_vs_pointcloud2(pointcloud1, pointcloud2)
    # ransac_pointcloud1, plane_model1 = random_sampling_consensus(pointcloud1, numb_iter, threshold)
    # ransac_pointcloud2, plane_model2 = random_sampling_consensus(pointcloud2, numb_iter, threshold)
    # Transf_matrix = calculate_rotation_matrix(plane_model1, plane_model2) # Fill the function
    # pointcloud2_fixed = transform_pointcloud(Transf_matrix, pointcloud2)
    # viewer_pointcloud1_vs_pointcloud2(pointcloud1, pointcloud2_fixed)


if __name__ == '__main__':
    main()