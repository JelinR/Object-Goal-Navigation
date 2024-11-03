import numpy as np
import envs.utils.depth_utils as du


class MapBuilder(object):
    def __init__(self, params):
        self.params = params
        frame_width = params['frame_width']
        frame_height = params['frame_height']
        fov = params['fov']
        self.camera_matrix = du.get_camera_matrix(
            frame_width,
            frame_height,
            fov)
        self.vision_range = params['vision_range']

        self.map_size_cm = params['map_size_cm']
        self.resolution = params['resolution']
        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']
        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        self.agent_height = params['agent_height']
        self.agent_view_angle = params['agent_view_angle']
        return

    def update_map(self, depth, current_pose):
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN

        #Get organized point cloud from depth image
        #Here, we assume that the depth shape is HxW, no scenes or channels dimensions (as seen in agent_view_cropped)
        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix,
                                                scale=self.du_scale)

        #Gets the 3D point cloud account for the camera elevation and angle
        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)

        #Gets the 3D point cloud accounting also for the camera location
        #Transforms to agent centered frame
        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        agent_view_centered = du.transform_pose(agent_view, shift_loc)

        #Bins 3D points onto a 3D count grid with each 3D bin containing counts of the 3D points binned into it
        #Output shape: (..., vision_range, vision_range, len(z_bins) + 1)
        agent_view_flat = du.bin_points(
            agent_view_centered,
            self.vision_range,
            self.z_bins,
            self.resolution)

        #Gets the obstacle map by only looking at a portion of the 3D grid
        #This takes out the local map (vision_range, vision_range) at z bin / position = 1
        #Then it applies the obstacle threshold, which expresses the value in terms of the threshold
        #We define the values above 0.5 are to be seen as obstacles
        agent_view_cropped = agent_view_flat[:, :, 1]
        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        #Gets the explored region map by flattening along the z dimension (dim = 2)
        #Explored area is shown as 1 in the map
        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0


        #Tranforms 3D point cloud (before binning) to Geocentric frame
        #Bins to get 3D count grid
        geocentric_pc = du.transform_pose(agent_view, current_pose)

        geocentric_flat = du.bin_points(
            geocentric_pc,
            self.map.shape[0],
            self.z_bins,
            self.resolution)

        #Update map with newly observed geocentric map
        self.map = self.map + geocentric_flat

        #Gets obstacle map (wrt geocentric frame) same as above
        map_gt = self.map[:, :, 1] / self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        #Gets explored map (wrt geocentric frame) same as above
        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) /
               (self.map_size_cm // (self.resolution * 2)),

               - (current_loc[1] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) /
               (self.map_size_cm // (self.resolution * 2)),
               
               90 - np.rad2deg(current_loc[2])]
        return loc

    def reset_map(self, map_size):
        self.map_size_cm = map_size

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

    def get_map(self):
        return self.map
