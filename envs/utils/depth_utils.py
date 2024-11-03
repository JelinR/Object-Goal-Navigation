# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for processing depth images.
"""
from argparse import Namespace

import itertools
import numpy as np
import torch

import envs.utils.rotation_utils as ru


def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix


def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    x, z = np.meshgrid(np.arange(Y.shape[-1]),
                       np.arange(Y.shape[-2] - 1, -1, -1))
    for _ in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x[::scale, ::scale] - camera_matrix.xc) * \
        Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * \
        Y[::scale, ::scale] / camera_matrix.f
    XYZ = np.concatenate((X[..., np.newaxis],
                          Y[::scale, ::scale][..., np.newaxis],
                          Z[..., np.newaxis]), axis=X.ndim)
    return XYZ


def transform_camera_view(XYZ, sensor_height, camera_elevation_degree):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix(
        [1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[:, :, 0] = XYZ[:, :, 0] + current_pose[0]
    XYZ[:, :, 1] = XYZ[:, :, 1] + current_pose[1]
    return XYZ

#Bins 3D points into a 3D bin count grid of shape (map_size, map_size, n_z_bins)
#Each 3D bin contains the number of 3D points binned into it (counts)
def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZ_cms.shape

    #Flattens the scenes and channels dimension
    #So, for each combination of scene and channel, we get a point set of shape (H, W, 3)
    XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    counts = []

    #Looping through each point set (for each combination of scene and channel)
    for XYZ_cm in XYZ_cms:
        isnotnan = np.logical_not(np.isnan(XYZ_cm[:, :, 0]))
        X_bin = np.round(XYZ_cm[:, :, 0] / xy_resolution).astype(np.int32)
        Y_bin = np.round(XYZ_cm[:, :, 1] / xy_resolution).astype(np.int32)

        #Assigns z values to the corresponding bins in z_bins
        Z_bin = np.digitize(XYZ_cm[:, :, 2], bins=z_bins).astype(np.int32)

        #Checks validity (if each point falls within the range of bins for x, y and z)
        #Also checks that the points are not NaN
        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0,
                            Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
        isvalid = np.all(isvalid, axis=0)

        #Get a flattened index for the count grid
        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin

        #Invalid points have their index set to zero
        ind[np.logical_not(isvalid)] = 0

        #Counts occurrences in each bin by first flattening the ind
        #Reshapes it later into the desired shape
        count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                            minlength=map_size * map_size * n_z_bins)
        counts = np.reshape(count, [map_size, map_size, n_z_bins])

    counts = counts.reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])

    return counts


def get_point_cloud_from_z_t(Y_t, camera_matrix, device, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    grid_x, grid_z = torch.meshgrid(torch.arange(Y_t.shape[-1]),
                                    torch.arange(Y_t.shape[-2] - 1, -1, -1))
    grid_x = grid_x.transpose(1, 0).to(device)
    grid_z = grid_z.transpose(1, 0).to(device)
    grid_x = grid_x.unsqueeze(0).expand(Y_t.size())
    grid_z = grid_z.unsqueeze(0).expand(Y_t.size())

    X_t = (grid_x[:, ::scale, ::scale] - camera_matrix.xc) * \
        Y_t[:, ::scale, ::scale] / camera_matrix.f
    Z_t = (grid_z[:, ::scale, ::scale] - camera_matrix.zc) * \
        Y_t[:, ::scale, ::scale] / camera_matrix.f

    XYZ = torch.stack(
        (X_t, Y_t[:, ::scale, ::scale], Z_t), dim=len(Y_t.size()))

    return XYZ


def transform_camera_view_t(
        XYZ, sensor_height, camera_elevation_degree, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix(
        [1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = torch.matmul(XYZ.reshape(-1, 3),
                       torch.from_numpy(R).float().transpose(1, 0).to(device)
                       ).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose_t(XYZ, current_pose, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = torch.matmul(XYZ.reshape(-1, 3),
                       torch.from_numpy(R).float().transpose(1, 0).to(device)
                       ).reshape(XYZ.shape)
    XYZ[..., 0] += current_pose[0]
    XYZ[..., 1] += current_pose[1]
    return XYZ


def splat_feat_nd(init_grid, feat, coords):
    """
    Args:
        init_grid: B X nF X W X H X D X ..
        feat: B X nF X nPt
        coords: B X nDims X nPt in [-1, 1]

        (nPt is Number of points)
    Returns:
        grid: B X nF X W X H X D X ..
    """
    #To save adjacent grid cells along each dimension
    #Of shape: (n_dims, 2, nPt)
    wts_dim = []
    pos_dim = []

    #Grid Dimensions: (Width, Height, Depth)
    grid_dims = init_grid.shape[2:] 

    B = init_grid.shape[0]  #Number of Batches / Scenes
    F = init_grid.shape[1]  #Number of Feature Maps / channels

    n_dims = len(grid_dims)

    #This flattens the dimension (W, H, D) into a single dimension
    #The last single dimension can be likened to a collection of all point cloud elements
    #Collapses all 3D grid cells into one dimension, helping refer to each cell in 1D
    grid_flat = init_grid.view(B, F, -1)

    for d in range(n_dims):

        #Reversing normalization that was done earlier (normalization and centering)
        #Returns 3D position (with only resolution scaling)
        pos = coords[:, [d], :] * grid_dims[d] / 2 + grid_dims[d] / 2
        pos_d = []
        wts_d = []

        #Gets the grid cell below and above the pos
        #Saves the pos_ix and weights for bilinear interpolation
        for ix in [0, 1]:
            pos_ix = torch.floor(pos) + ix

            #Check if within bounds
            safe_ix = (pos_ix > 0) & (pos_ix < grid_dims[d])
            safe_ix = safe_ix.type(pos.dtype)   #If True, then 1, if False, then 0

            #If pox_ix is more than 0.5 wrt pos, then it gets a small weight, else it gets a higher weight
            #So, the closer pox_ix is to pox, the higher the weight
            wts_ix = 1 - torch.abs(pos - pos_ix)

            wts_ix = wts_ix * safe_ix
            pos_ix = pos_ix * safe_ix

            pos_d.append(pos_ix)
            wts_d.append(wts_ix)

        #For all points along a specific dimension,
        #Saves adjacent grid cells and their weights
        pos_dim.append(pos_d)
        wts_dim.append(wts_d)

    #Along each dim
    l_ix = [[0, 1] for d in range(n_dims)]

    #Loops over the 8 points of a cube
    #From (0, 0, 0) to (1, 1, 1)
    for ix_d in itertools.product(*l_ix):

        #Of shape: (nPts,)
        #index: Keeps track of final grid cell index to add feature
        #wts: For a grid cell, accumulates weights from each dims
        wts = torch.ones_like(wts_dim[0][0])
        index = torch.zeros_like(wts_dim[0][0])

        for d in range(n_dims):

            #grid_dims[d]: Size of grid in current dimension (d)
            #pos_dim[d][ix_d[d]]: Exact cell position in current dimension (d)
            #index: Points to the correct flattened cell in grid_flat
            index = index * grid_dims[d] + pos_dim[d][ix_d[d]]

            #Starts at 1 (100% of the feature)
            #Multiplies the weights across the three dimensions
            wts = wts * wts_dim[d][ix_d[d]]

        index = index.long()    #Converts to type(int64)

        #The underscore (_) at the end means this is applied in place
        #The dimension 2 contains the flattened spatial points
        grid_flat.scatter_add_(2, index.expand(-1, F, -1), feat * wts)
        grid_flat = torch.round(grid_flat)

    return grid_flat.view(init_grid.shape)
