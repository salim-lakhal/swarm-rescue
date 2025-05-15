import numpy as np
import cv2
from spg_overlay.utils.grid import Grid
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


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
        self.binary_image = np.where(self.grid != 0, 255, 0).astype(np.uint8).T
        self.visited_cells = 0
        self.clusters = []

    
    def update_grid(self, pose: Pose,lidar_values,lidar_rays_angles):
        if(lidar_values is None or lidar_rays_angles is None):
            pass
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

        # Créer une image binaire où les zones explorées sont 255 et les non explorées sont 0
        self.binary_image = np.where(self.grid != 0, 255, 0).astype(np.uint8).T

        #self.display_grid(pose)

    def display_grid(self,pose):
        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)
        
        
        #config_grid = np.where(self.grid > 2, 255, 0).astype(np.uint8)
        config_image = cv2.resize(np.where(self.grid > 2, 255, 0).astype(np.uint8),new_zoomed_size,interpolation=cv2.INTER_NEAREST)
        explo_image = cv2.resize(np.where(self.grid != 0, 255, 0).astype(np.uint8),new_zoomed_size,interpolation=cv2.INTER_NEAREST)

        self.iteration += 1
        if self.iteration % 5 == 0:
            #self.display(cv2.cvtColor(config_image,cv2.COLOR_BGR2GRAY),pose,title="Config Grid")
            self.display(explo_image,pose,title="Exploration Grid")
            self.display(self.grid, pose, title="Occupancy Grid")
            self.display(self.zoomed_grid, pose, title="Zoomed Occupancy Grid")

    def find_frontier(self):
        """
        Find frontier cells in the occupancy grid.
        A frontier is a free cell adjacent to an unknown cell.
        """
        frontier_points = []

        # Loop through each cell in the grid
        for x in range(1, self.grid.shape[0] - 1):  # Exclude borders to avoid out-of-bounds
            for y in range(1, self.grid.shape[1] - 1):
                if self.grid[x, y] <= -4.0 :#-4.0:  # Free cell
                    # Check if any adjacent cell is unknown (0)
                    neighbors = self.grid[x-1:x+2, y-1:y+2]
                    if np.any(neighbors == 0):  # There is an unknown cell around
                        frontier_points.append((x, y))

        # Convert grid indices to world coordinates
        return np.array([
            self._conv_grid_to_world(pt[0],pt[1])
            for pt in frontier_points
        ])
    
    def extract_exploration_boundaries(self, robot_position):
        """
        Extrait les frontières entre les zones explorées (différentes de -4) et non explorées (0),
        en excluant les points derrière les obstacles.
        :param robot_position: Position du robot (x, y) dans le monde.
        :return: Tableau des points de frontière visibles.
        """
        # Trouver les contours entre les zones explorées et non explorées
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Liste pour stocker les points de frontière visibles
        boundary_points = []

        # Position du robot
        robot_x, robot_y = robot_position

        for contour in contours:
            for point in contour:
                x, y = point[0]
                x_world, y_world = self._conv_grid_to_world(x, y)

                # Vérifier si le point est visible depuis la position du robot
                if self.is_visible(robot_x, robot_y, x_world, y_world):
                    boundary_points.append((x_world, y_world))

        return np.array(boundary_points)
    
    def cluster_boundary_points(self, robot_position, eps=70, min_samples=2):
        """
        Clusterise les boundary points en utilisant DBSCAN.
        :param robot_position: Position du robot pour extraire les points de frontière.
        :param eps: Distance maximale entre deux points pour les considérer comme voisins.
        :param min_samples: Nombre minimal de points pour former un cluster.
        :return: Liste des centroïdes des clusters et le nombre de clusters retenus.
        """
        # Extraire les points de frontière
        #data = self.extract_exploration_boundaries(robot_position)
        data = self.find_frontier()

        if data is None :
            return ([],0)
        elif np.size(data) == 1:
            return ([],0)
        elif np.size(data) == 0:
            return ([],0)


        # Appliquer DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = db.labels_

        # Identifier les points aberrants
        outliers = data[labels == -1]

        # Filtrer les données pour supprimer les points aberrants
        filtered_data = data[labels != -1]
        filtered_labels = labels[labels != -1]


        # Initialiser une liste pour stocker les centroïdes
        centroids = []

        # Parcourir chaque cluster unique (en ignorant le bruit, label = -1)
        unique_labels = set(filtered_labels)
        for label in unique_labels:
            # Filtrer les points du cluster actuel
            cluster_points = filtered_data[filtered_labels == label]

            self.clusters.append(cluster_points)

            # Vérifier si le cluster a au moins 10 points
            if len(cluster_points) >= 1:
                # Calculer la moyenne des points dans le cluster (centroïde)
                mean = np.mean(cluster_points, axis=0)
                centroids.append(mean)
            else:
                print(f"Cluster {label} ignoré car il contient moins de 1 points.")

        # Convertir la liste des centroïdes en un tableau NumPy
        centroids = np.array(centroids)

        # Nombre de clusters retenus
        n_cluster = len(centroids)

        return centroids, n_cluster
    
    def is_visible(self, x1, y1, x2, y2):
        """
        Vérifie si le point (x2, y2) est visible depuis le point (x1, y1) en vérifiant les obstacles.
        :param x1, y1: Coordonnées du point de départ (robot).
        :param x2, y2: Coordonnées du point à vérifier.
        :return: True si le point est visible, False sinon.
        """
        # Convertir les coordonnées du monde en coordonnées de la grille
        x1_grid, y1_grid = self._conv_world_to_grid(x1, y1)
        x2_grid, y2_grid = self._conv_world_to_grid(x2, y2)

        # Générer les points le long de la ligne entre (x1_grid, y1_grid) et (x2_grid, y2_grid)
        line_points = np.array(list(zip(
            np.linspace(x1_grid, x2_grid, num=20),  # 100 points intermédiaires
            np.linspace(y1_grid, y2_grid, num=20)
        ))).astype(int)

        # Vérifier si un obstacle se trouve sur la ligne
        for (x, y) in line_points:
            if self.grid[x, y] == 2:  # 2 représente un obstacle
                return False
        return True
        
    # Renvoie une liste de tuples (x,y) d'obstacle carré de taille 8 pixel
    def get_obstacles(self):
        obstacles = []
        for x in range (1, self.grid.shape[0] - 1):
            for y in range(1, self.grid.shape[1] - 1):
                if self.grid[x, y] > 0:
                   x_world,y_world= self._conv_grid_to_world(x,y)
                   obstacles.append((x_world,y_world))
        return obstacles
    
    def get_obstacles_grid(self):
        obstacles = []
        for x in range (0, self.grid.shape[0] - 1):
            for y in range(0, self.grid.shape[1] - 1):
                if self.grid[x, y] > 0:
                   x_world,y_world= self._conv_grid_to_world(x,y)
                   obstacles.append((((x_world+self.size_area_world[0]/2) / 20), ((y_world+self.size_area_world[1]/2) / 20), 0.1) )
        return obstacles
    
    def get_visited_ratio(self):
        self.visited_cells = np.count_nonzero(self.binary_image == 255)
        return self.visited_cells/(self.x_max_grid*self.y_max_grid)



