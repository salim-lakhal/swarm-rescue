import numpy as np
import cv2
from spg_overlay.utils.grid import Grid
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR

class OccupancyGrid(Grid):
    """Simple occupancy grid"""

    def __init__(self,
                 size_area_world,
                 resolution: float,
                ):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))
        self.binary_grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.obstacles = []
        self.iteration = 0

    def find_frontier(self):
        """
        Find frontier cells in the occupancy grid.
        A frontier is a free cell adjacent to an unknown cell.
        """
        frontier_points = []

        # Loop through each cell in the grid
        for x in range(1, self.grid.shape[0] - 1):  # Exclude borders to avoid out-of-bounds
            for y in range(1, self.grid.shape[1] - 1):
                if self.grid[x, y] == -4.0:  # Free cell
                    # Check if any adjacent cell is unknown (0)
                    neighbors = self.grid[x-1:x+2, y-1:y+2]
                    if np.any(neighbors == 0):  # There is an unknown cell around
                        frontier_points.append((x, y))

        # Convert grid indices to world coordinates
        return [
            (pt[0] * self.resolution, pt[1] * self.resolution)
            for pt in frontier_points
        ]
    
    def update_grid(self, pose: Pose,lidar_values,lidar_rays_angles):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40

        lidar_dist = lidar_values[::EVERY_N].copy()
        lidar_angles = lidar_rays_angles[::EVERY_N].copy()

        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # For empty zones
        # points_x and point_y contains the border of detected empty zone
        # We use a value a little bit less than LIDAR_DIST_CLIP because of the
        # noise in lidar
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        # All values of lidar_dist_empty_clip are now <= max_range
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip,
                                                  cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip,
                                                  sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y,
                                      EMPTY_ZONE_VALUE)

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)
        
        self.iteration += 1
        if self.iteration % 5 == 0:
            self.display(self.grid, pose, title="Occupancy Grid")
            self.display(self.zoomed_grid, pose, title="Zoomed Occupancy Grid")
        
    # Renvoie une liste de tuples (x,y) d'obstacle carré de taille 8 pixel
    def get_obstacles(self):
        obstacles = []
        for x in range (1, self.grid.shape[0] - 1):
            for y in range(1, self.grid.shape[1] - 1):
                if self.grid[x, y] > 0:
                   x_world,y_world= self._conv_grid_to_world(x,y)
                   obstacles.append((x_world, y_world))
        return obstacles

    def save_grid(self):
        if self.iteration == 100:
            with open('resultat_occupancy_grid.txt', 'w') as file:
                for row in self.grid.T:
                    file.write(' '.join(map(str, row)) + '\n')

