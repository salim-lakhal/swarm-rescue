"""
First assessment controller
"""
import math
import arcade
from math import sqrt
from copy import deepcopy
from typing import Optional
from heapq import heappush, heappop
import queue
from random import randrange
from enum import Enum
import random

import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle, circular_mean
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign
from spg_overlay.utils.utils import clamp
from spg_overlay.utils.timer import Timer
from spg_overlay.utils.fps_display import FpsDisplay
from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.keyboard_controller import KeyboardController
from statemachine import StateMachine, State
from drone_modules.slam_module import SLAMModule
from drone_modules.exploration_grid import ExplorationGrid
from drone_modules.path_planner import RRT
from drone_modules.path_tracker import PathTracker
from drone_modules.occupancy_grid import OccupancyGrid
from drone_modules.path_tracker_modules.stanley_controller_piecewise import StanleyController
from drone_modules.path_tracker_modules.pure_pursuit import PurePursuitController

class MyDroneFirst(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 0
        GRASPING_WOUNDED = 1
        SEARCHING_RESCUE_CENTER = 2
        DROPPING_AT_RESCUE_CENTER = 3
        FOLLOW_PATH = 5
        INITIAL = 4
        PATH_PLANNING = 6

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        

        # Modules
        self.slam = SLAMModule() #Ignorer pas encore implémenter
        self.grid = OccupancyGrid(size_area_world=self.size_area, resolution=10)
        self.explorationGrid = ExplorationGrid(drone=self,grid=self.grid)
        #self.path_planner = PathPlanner(grid=self.grid,resolution=8) #Ignorer pas encore implémenter
        self.pathTracker = PathTracker()
        self.purePursuit = PurePursuitController()
        self.stanleyController = StanleyController(wheelbase=0,yaw_rate_gain=0.3,steering_damp_gain=0.1,control_gain=2.5,softening_gain=1.0)
        self.iter = 0
        self.grasp = 0
        self.pose_initial = Pose()
        self.pose = Pose()
        self.path = Path()
        self.fps_display = FpsDisplay(period_display=1)
        self.timer = Timer(start_now=True)
        self.keyController = KeyboardController()
        self.state = self.Activity.SEARCHING_WOUNDED    #SEARCHING_WOUNDED
        self.states = {self.Activity.SEARCHING_WOUNDED: self.searching_wounded,
                    self.Activity.GRASPING_WOUNDED: self.grasping_wounded,
                    self.Activity.SEARCHING_RESCUE_CENTER: self.searching_rescue_center,
                    self.Activity.DROPPING_AT_RESCUE_CENTER: self.dropping_at_rescue_center,
                    self.Activity.FOLLOW_PATH: self.follow_path,
                    self.Activity.PATH_PLANNING: self.path_planning
                    }
        self.state_machine = [self.Activity.SEARCHING_WOUNDED,
                            self.Activity.GRASPING_WOUNDED,
                            self.Activity.SEARCHING_RESCUE_CENTER,
                            self.Activity.DROPPING_AT_RESCUE_CENTER,
                            self.Activity.FOLLOW_PATH,
                            self.Activity.PATH_PLANNING]
        self.points = queue.Queue()
        self.planned = False
        self.rescue = self.gps_values()
        self.historic = []
        self.point_index = 0
        self.follow = False

    def control(self): # BOUCLE PRINCIPALE
        """
        Le drone utlise une Occupancy Grid pour explorer les zones non visitée 
        qui est inclus dans la classe ExplorationGrid qui s'occupe de l'exploration 
        de la grille. Lorsque la personne est trouvée nous utilisons process_semantic_sensor pour retourner à la base.
        """
        delta_time = self.timer.get_elapsed_time()
        self.timer.restart()  # Redémarrage pour le prochain cycle
        self.pose.position = self.true_position()
        self.pose.orientation = self.true_angle()
        lidar_values = self.lidar_values()
        lidar_rays_angles = self.lidar_rays_angles()   
        vx,vy = self.measured_velocity()
        v_angle = self.measured_angular_velocity()
        # Mise à jour & Affichage de l'Occupancy Grid
        self.grid.update_grid(self.pose,lidar_values,lidar_rays_angles)
        command = self.states[self.state]()
        print(self.state)
        
        return command


################
#  STATE MACHINE                
################



    def update_state(self):    
        """
        Transitions of the State Machine 
        using the self.state_machine
        It increments the self.index to 
        move from a state to another
        """

        if self.state != self.Activity.FOLLOW_PATH:
            self.index += 1
            self.index = self.index % 6
            self.state = self.state_machine[self.index]
        

        return None
    
    
    def searching_wounded(self):
        """
        Searching for a wounded person
        """
        #command = self.explorationGrid.control()
        
        found_wounded,found_rescue_center,command = self.process_semantic_sensor()
        if found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED
        if min(self.lidar_values())<20 or self.follow:
            self.path.reset()
            command = self.wall_following()
            self.follow = True
        elif not self.follow:
            if not self.planned:
                self.path_planning()
                self.planned = True
                self.follow = False
            command = self.follow_path()
            self.iter = 0
        command["grasper"] = 0
        return command
    
    def grasping_wounded(self):
        """
        Grasping a wounded person
        """
        found_wounded,found_rescue_center,command = self.process_semantic_sensor()
        command["grasper"] = 1
        if self.grasped_entities():
            command["grasper"] = 1
            self.state = self.Activity.SEARCHING_RESCUE_CENTER
        self.iter = 1
        print("Grasping Wounded - grasped:", self.grasped_entities())

        return command
    
    def searching_rescue_center(self):
        """
        Searching for the rescue center
        """
        self.path.reset()
        test = self.path_planning(start=self.rescue)
        print("Searching Rescue Center - path planned:", test)
        command = self.follow_path()
        #command = self.explorationGrid.control() 
        #command["grasper"] = 1
        
        return command
    
    def follow_path(self):
        """
        Follow a path
        """
        found_wounded,found_rescue_center,command = self.process_semantic_sensor()
        delta_time = self.timer.get_elapsed_time()

        # Si on a finis le chemin
        if self.pathTracker.isFinish(self.path):
            self.point_index +=1
            self.path.reset()
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER
            command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
            self.planned = False
        elif self.is_inside_return_area and self.iter ==1:
            self.path.reset()
            self.state = self.Activity.SEARCHING_WOUNDED
        command = self.pathTracker.control(self.pose,self.path,delta_time)
        print("Follow Path - path finished:", self.pathTracker.isFinish(self.path))
        return command
    
    def dropping_at_rescue_center(self):
        """
        Dropping the wounded person at the rescue center
        """
        found_wounded,found_rescue_center,command = self.process_semantic_sensor()
        command["grasper"] = 1
        if not self.grasped_entities():
            self.state = self.Activity.SEARCHING_WOUNDED
            command["grasper"] = 1
        self.iter = 0
        print("Dropping - grasped:", self.grasped_entities())
        return command
    def randomPoint(self):
        point = (random.uniform(-self.grid.size_area_world[0]/2+50,self.grid.size_area_world[0]/2-50),random.uniform(-self.grid.size_area_world[1]/2+50,self.grid.size_area_world[1]/2-50))
        resp = (int((point[0] + self.grid.size_area_world[0]/2)/10) , int((point[1] + self.grid.size_area_world[1]/2)/10))
        return resp
    
    def path_planning(self,start=[]):
        goal = [(self.true_position()[0] + self.grid.size_area_world[0]/2)/10,(self.true_position()[1] + self.grid.size_area_world[1]/2)/10]
        if len(start) == 0:
            point = self.randomPoint()
            self.historic.append(point)
        else:
            point = start
            self.historic.append(point)
        obstacles = self.grid.get_obstacles()
        obstacle_list = self.conv_obstacle(obstacles)
        play_area = [5,self.grid.size_area_world[0]/10-5,5,self.grid.size_area_world[1]/10-5]
        path_planning = RRT(start=self.historic[self.point_index],
                    goal=goal,
                    obstacle_list=obstacle_list,
                    rand_area = [0, self.grid.size_area_world[0]/10],  
                    play_area=play_area
                    )
        self.path.reset()
        path = path_planning.planning()
        for node in path:
            current_node = np.zeros(2, )
            current_node[0] = node[0]*10 - self.grid.size_area_world[0]/2
            current_node[1] = node[1]*10 - self.grid.size_area_world[1]/2
            self.path.append(Pose(current_node))
        return None
          
    @staticmethod
    def conv_obstacle(obstacles):
        converted_obstacles = []
        for obstacle in obstacles:
            converted_obstacle = (int(obstacle[0] / 10), int(obstacle[1] / 10), 1)
            converted_obstacles.append(converted_obstacle)
        return set(converted_obstacles)

    def wall_following(self):
            """
            État de suivi de mur.
            """
                                # Récupérer les valeurs du LiDAR
            lidar_values = self.lidar_values()
            lidar_angles = self.lidar_rays_angles()

                # Appeler la fonction de suivi de mur
            command = self.wall_following_control(lidar_values, lidar_angles, K=50, forward_speed=1.0, angular_speed=0.5)

            return command

    def wall_following_control(self, lidar_values, lidar_angles, K=50, forward_speed=1.0, angular_speed=0.5):
            """
            Fonction de contrôle pour le suivi de mur.
            
            :param lidar_values: Liste des distances mesurées par le LiDAR.
            :param lidar_angles: Liste des angles correspondants aux mesures du LiDAR.
            :param K: Distance constante à maintenir par rapport au mur.
            :param forward_speed: Vitesse de déplacement vers l'avant.
            :param angular_speed: Vitesse de rotation.
            :return: Commande de mouvement pour le drone.
            """
            # Initialisation de la commande
            command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

            # Détection des distances devant et à droite
            front_dist = min([dist for dist, angle in zip(lidar_values, lidar_angles) if -np.pi/4 < angle < np.pi/4])
            right_dist = min([dist for dist, angle in zip(lidar_values, lidar_angles) if np.pi/4 < angle < 3*np.pi/4])

            # Erreur de distance par rapport à K
            error_distance = right_dist - K

            # Contrôleur PID pour la rotation
            rotation_command = self.pathTracker.pid_steering.compute(error_distance, delta_time=1/30)
            rotation_command = clamp(rotation_command, -1.0, 1.0)

            # Contrôleur PID pour la vitesse
            forward_command = self.pathTracker.pid_forward.compute(front_dist - K, delta_time=1/30)
            forward_command = clamp(forward_command, 0.0, forward_speed)

            # Logique de suivi de mur
            if front_dist < K:  # Obstacle détecté devant
                # Ralentir et ajuster la rotation
                command["forward"] = forward_command * 0.5
                command["rotation"] = rotation_command
            elif right_dist > K * 1.2:  # Plus de mur à droite
                # Tourner à droite pour suivre un nouveau mur
                command["rotation"] = -angular_speed
                command["forward"] = forward_command * 0.5
            else:
                # Suivre le mur à distance K
                command["rotation"] = rotation_command
                command["forward"] = forward_command

            return command

    def process_lidar_sensor(self, the_lidar_sensor): 
        command = {"forward": 1.0,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller = 1.0

        values = the_lidar_sensor.get_sensor_values()

        if values is None:
            return command, False

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution

        far_angle_raw = 0
        near_angle_raw = 0
        min_dist = 1000
        if size != 0:
            # far_angle_raw : angle with the longer distance
            far_angle_raw = ray_angles[np.argmax(values)]
            min_dist = min(values)
            # near_angle_raw : angle with the nearest distance
            near_angle_raw = ray_angles[np.argmin(values)]

        far_angle = far_angle_raw
        # If far_angle_raw is small then far_angle = 0
        if abs(far_angle) < 1 / 180 * np.pi:
            far_angle = 0.0

        near_angle = near_angle_raw
        far_angle = normalize_angle(far_angle)

        # The drone will turn toward the zone with the more space ahead
        if size != 0:
            if far_angle > 0:
                command["rotation"] = angular_vel_controller
            elif far_angle == 0:
                command["rotation"] = 0
            else:
                command["rotation"] = -angular_vel_controller

        # If near a wall then 'collision' is True and the drone tries to turn its back to the wall
        collision = False
        if size != 0 and min_dist < 50:
            collision = True
            if near_angle > 0:
                command["rotation"] = -angular_vel_controller
            else:
                command["rotation"] = angular_vel_controller

        return command, collision

    def define_message_for_all(self):#Ignorer
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        return msg_data

    def process_communication_sensor(self): 
        found_drone = False

        command_comm = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0}

        if self.communicator:
            received_messages = self.communicator.received_messages
            nearest_drone_coordinate1 = (
                self.measured_gps_position(), self.measured_compass_angle())
            nearest_drone_coordinate2 = deepcopy(nearest_drone_coordinate1)
            (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
            (nearest_position2, nearest_angle2) = nearest_drone_coordinate2

            min_dist1 = 10000
            min_dist2 = 10000
            diff_angle = 0

            # Search the two nearest drones around
            for msg in received_messages:
                message = msg[1]
                coordinate = message[1]
                (other_position, other_angle) = coordinate

                dx = other_position[0] - self.measured_gps_position()[0]
                dy = other_position[1] - self.measured_gps_position()[1]
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # if another drone is near
                if distance < min_dist1:
                    min_dist2 = min_dist1
                    min_dist1 = distance
                    nearest_drone_coordinate2 = nearest_drone_coordinate1
                    nearest_drone_coordinate1 = coordinate
                    found_drone = True
                elif distance < min_dist2 and distance != min_dist1:
                    min_dist2 = distance
                    nearest_drone_coordinate2 = coordinate

            if not found_drone:
                return found_drone, command_comm

            # If we found at least 2 drones
            if found_drone and len(received_messages) >= 2:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                (nearest_position2, nearest_angle2) = nearest_drone_coordinate2
                diff_angle1 = normalize_angle(
                    nearest_angle1 - self.measured_compass_angle())
                diff_angle2 = normalize_angle(
                    nearest_angle2 - self.measured_compass_angle())
                # The mean of 2 angles can be seen as the angle of a vector, which
                # is the sum of the two unit vectors formed by the 2 angles.
                diff_angle = math.atan2(0.5 * math.sin(diff_angle1) + 0.5 * math.sin(diff_angle2),
                                        0.5 * math.cos(diff_angle1) + 0.5 * math.cos(diff_angle2))

            # If we found only 1 drone
            elif found_drone and len(received_messages) == 1:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                diff_angle1 = normalize_angle(
                    nearest_angle1 - self.measured_compass_angle())
                diff_angle = diff_angle1

            # if you are far away, you get closer
            # heading < 0: at left
            # heading > 0: at right
            # base.angular_vel_controller : -1:left, 1:right
            # we are trying to align : diff_angle -> 0
            command_comm["rotation"] = sign(diff_angle)

            # Desired distance between drones
            desired_dist = 60

            d1x = nearest_position1[0] - self.measured_gps_position()[0]
            d1y = nearest_position1[1] - self.measured_gps_position()[1]
            distance1 = math.sqrt(d1x ** 2 + d1y ** 2)

            d1 = distance1 - desired_dist
            # We use a logistic function. -1 < intensity1(d1) < 1 and  intensity1(0) = 0
            # intensity1(d1) approaches 1 (resp. -1) as d1 approaches +inf (resp. -inf)
            intensity1 = 2 / (1 + math.exp(-d1 / (desired_dist * 0.5))) - 1

            direction1 = math.atan2(d1y, d1x)
            heading1 = normalize_angle(direction1 - self.measured_compass_angle())

            # The drone will slide in the direction of heading
            longi1 = intensity1 * math.cos(heading1)
            lat1 = intensity1 * math.sin(heading1)

            # If we found only 1 drone
            if found_drone and len(received_messages) == 1:
                command_comm["forward"] = longi1
                command_comm["lateral"] = lat1

            # If we found at least 2 drones
            elif found_drone and len(received_messages) >= 2:
                d2x = nearest_position2[0] - self.measured_gps_position()[0]
                d2y = nearest_position2[1] - self.measured_gps_position()[1]
                distance2 = math.sqrt(d2x ** 2 + d2y ** 2)

                d2 = distance2 - desired_dist
                intensity2 = 2 / (1 + math.exp(-d2 / (desired_dist * 0.5))) - 1

                direction2 = math.atan2(d2y, d2x)
                heading2 = normalize_angle(direction2 - self.measured_compass_angle())

                longi2 = intensity2 * math.cos(heading2)
                lat2 = intensity2 * math.sin(heading2)

                command_comm["forward"] = 0.5 * (longi1 + longi2)
                command_comm["lateral"] = 0.5 * (lat1 + lat2)

        return found_drone, command_comm
        
    def update_command_search(self,command): #Ignorer
        command_lidar, collision_lidar = self.process_lidar_sensor(self.lidar())
        found, command_comm = self.process_communication_sensor()

        alpha = 0.4
        alpha_rot = 0.75
        if collision_lidar:
            alpha_rot = 0.1

        # The final command  is a combination of 2 commands
        command["forward"] = \
            alpha * command_comm["forward"] \
            + (1 - alpha) * command_lidar["forward"]
        command["lateral"] = \
            alpha * command_comm["lateral"] \
            + (1 - alpha) * command_lidar["lateral"]
        command["rotation"] = \
            alpha_rot * command_comm["rotation"] \
            + (1 - alpha_rot) * command_lidar["rotation"]
        
        return command

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move
        towards a wound person or the rescue center
        """
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_wounded = False
        if (self.state is self.Activity.SEARCHING_WOUNDED
            or self.state is self.Activity.GRASPING_WOUNDED) \
                and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and
                        not data.grasped):
                    found_wounded = True
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))

            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (self.state is self.Activity.SEARCHING_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
                and detection_semantic:
            for data in detection_semantic:
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance < 30)

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))

        # Gére le déplacement aprés avoir trouvée la personne à ignorer
        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max
            #command["rotation"] = self.pathTracker.control_angle(0,best_angle,1/30)["rotation"]

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_rescue_center and is_near:
            command["forward"] = 0.0
            command["rotation"] = -1.0

        return found_wounded, found_rescue_center, command
    
    def draw_bottom_layer(self):
        #self.draw_setpoint()
        self.draw_path(path=self.path, color=(255, 0, 255))
            
        self.draw_coordinate_system()
        #self.draw_obstacles()
        #self.draw_direction()
        #self.draw_antedirection()
        return None
    
    def draw_setpoint(self):
        half_width = self._half_size_array[0]
        half_height = self._half_size_array[1]
        pt1 = self.position_setpoint + np.array([half_width, 0])
        pt2 = self.position_setpoint + np.array([half_width, 2 * half_height])
        arcade.draw_line(float(pt2[0]),
                            float(pt2[1]),
                            float(pt1[0]),
                            float(pt1[1]),
                            color=arcade.color.GRAY)

    def draw_path(self, path, color: list[int, int, int]):
        length = path.length()
        # print(length)
        pt2 = None
        for ind_pt in range(length):
            pose = path.get(ind_pt)
            pt1 = pose.position + self._half_size_array
            # print(ind_pt, pt1, pt2)
            if ind_pt > 0:
                arcade.draw_line(float(pt2[0]),
                                 float(pt2[1]),
                                 float(pt1[0]),
                                 float(pt1[1]), color)
            pt2 = pt1
    
    def draw_direction(self):
        pt1 = np.array([self.true_position()[0], self.true_position()[1]])
        pt1 = pt1 + self._half_size_array
        pt2 = pt1 + 250 * np.array([math.cos(self.true_angle()),
                                    math.sin(self.true_angle())])
        color = (255, 64, 0)
        arcade.draw_line(float(pt2[0]),
                         float(pt2[1]),
                         float(pt1[0]),
                         float(pt1[1]),
                         color)

    def draw_antedirection(self):
        pt1 = np.array([self.true_position()[0], self.true_position()[1]])
        pt1 = pt1 + self._half_size_array
        pt2 = pt1 + 150 * np.array([math.cos(self.true_angle() + np.pi / 2),
                                    math.sin(self.true_angle() + np.pi / 2)])
        color = (255, 64, 0)
        arcade.draw_line(float(pt2[0]),
                         float(pt2[1]),
                         float(pt1[0]),
                         float(pt1[1]),
                         color)
    
    def draw_coordinate_system(self, color_axis=(0, 0, 0), color_ticks=(100, 100, 100), tick_interval=100):
        """
        Dessine un repère centré au milieu de la carte, avec des graduations.

        :param dimensions: Dimensions de la carte (width, height) en pixels.
        :param color_axis: Couleur des axes (RGB), par défaut noir.
        :param color_ticks: Couleur des graduations (RGB), par défaut gris clair.
        :param tick_interval: Distance entre les graduations (en pixels), par défaut 50.
        """
        width, height  = self.size_area
        center_x = width / 2
        center_y = height / 2

        # Dessin des axes
        arcade.draw_line(0, center_y, width, center_y, color_axis, 2)  # Axe X
        arcade.draw_line(center_x, 0, center_x, height, color_axis, 2)  # Axe Y

        # Ajout des graduations sur l'axe X
        for x in range(0, width + 1, tick_interval):
            x_world = x - center_x
            arcade.draw_line(x, center_y - 5, x, center_y + 5, color_ticks, 1)  # Petite graduation
            if x_world % (2 * tick_interval) == 0:  # Grande graduation avec label
                arcade.draw_text(f"{int(x_world)}", x + 5, center_y + 10, color_ticks, 10, anchor_x="center")

        # Ajout des graduations sur l'axe Y
        for y in range(0, height + 1, tick_interval):
            y_world = y - center_y
            arcade.draw_line(center_x - 5, y, center_x + 5, y, color_ticks, 1)  # Petite graduation
            if y_world % (2 * tick_interval) == 0:  # Grande graduation avec label
                arcade.draw_text(f"{int(y_world)}", center_x + 10, y - 5, color_ticks, 10, anchor_x="left")
   
    def draw_obstacles(self, color=(255, 0, 0), square_size=8):
        """
        Dessine les obstacles sur la carte sous forme de petits carrés.

        :param color: Couleur des obstacles (RGB), par défaut rouge.
        :param square_size: Taille des carrés représentant les obstacles (en pixels), par défaut 5.
        """
        # Obtenir la liste des obstacles en coordonnées du monde réel
        obstacles = self.grid.get_obstacles()


        # Convertir les coordonnées du monde réel en pixels pour dessiner
        width, height = self.size_area  # Dimensions de la carte en pixels
        center_x = width / 2
        center_y = height / 2

        for obstacle in obstacles:
            x_world, y_world = obstacle
            # Conversion des coordonnées du monde en coordonnées de la carte
            x_pixel = int(center_x + x_world)
            y_pixel = int(center_y + y_world)
            

            # Dessiner un carré représentant l'obstacle
            arcade.draw_rectangle_filled(
                x_pixel, y_pixel, square_size, square_size, color
            )

            arcade.draw_circle_filled(x_world + center_x, y_world + center_y, 4, (0,255,0))
            # Dessiner les coordonnées x, y en bleu juste à côté de chaque obstacle
            #arcade.draw_text(f"({x_world}, {y_world})", x_world + center_x - 50 ,y_world + center_y  - 50, color=(0, 0, 255), font_size=10)