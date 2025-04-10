import math
import numpy as np
#from simple_pid import PID
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.utils import clamp
from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose

class PID:
    """
    Kp : Augmente la réactivité. Plus Kp est élevé, plus le système répond vite à une erreur, mais attention aux oscillations si Kp est trop grand.
    Ki : Corrige les erreurs résiduelles (offset), mais peut ralentir le système ou causer un "windup".
    Kd : Aide à amortir les oscillations causées par un Kp élevé. Augmenter Kd améliore la stabilité pour un Kp plus grand.
    """
    def __init__(self, Kp, Ki, Kd,error_init=0):
        self.kp = Kp
        self.ki = Ki
        self.kd = Kd
        self.integral = 0
        self.previous_error = error_init

    def compute(self, error, delta_time):
        delta_time = 1/30 # A changer

        if delta_time <= 0:
            return 0
        
        self.integral += error * delta_time
        derivative = (error - self.previous_error) / delta_time
        self.previous_error = error
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class PathTracker:

    def __init__(self):

        # PID controllers for forward and lateral forces
        self.pid_forward = PID(Kp=0.03, Ki=0.0001, Kd=0.01) 
        self.pid_lateral = PID(Kp=0.03, Ki=0.0001, Kd=0.01)


        # PID pour l'angle de direction
        self.pid_steering = PID(Kp=5, Ki=0.0001, Kd=0.01)
        self.diff_angle = 0

        self.current_target_index = 0  # Indice du point cible actuel
        self.iter_path = 0
        self.error_distance = 0
        self.tolerance = 50

    # Contrôle d'anle
    def control_angle(self,current_angle,target_angle,delta_time):
        """
        The Drone will turn for a fix angle
        """

        self.diff_angle = normalize_angle(target_angle - current_angle)
        rotation = self.pid_steering.compute(self.diff_angle,delta_time)

        rotation = clamp(rotation, -1.0, 1.0)

        command = {"rotation": rotation}

        return command

    def control(self, current_pose : Pose ,path : Path,delta_time):

        """Aller d'un point A à B à l'aide d'un contrôleur PID"""

        # Débeuger au cas où il n'y a pas de valeurs ou que tous les points sont atteints
        if path.length() == 0 or self.isFinish(path):
            return {"forward": 0, "lateral": 0, "rotation": 0}
        
        
        #self.update_path_done(current_pose)

        theta = current_pose.orientation

        # Matrice de rotation
        rotation_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

        target_pose = path.get(self.current_target_index)

        target_position_translated = np.array([target_pose.position[0]-current_pose.position[0],target_pose.position[1]-current_pose.position[1]])

        target_position_robot = np.dot(rotation_matrix, target_position_translated)

        self.error_distance = np.linalg.norm(target_position_robot)

        # Update PID controllers
        forward = self.pid_forward.compute(target_position_robot[0], delta_time)
        lateral = self.pid_lateral.compute(target_position_robot[1], delta_time)
        rotation = 0



        # Limit PID outputs to [-1, 1]

        forward = max(min(forward, 1), -1)
        lateral = max(min(lateral, 1), -1)

        command = {"forward": forward,
            "lateral": lateral,
            "rotation": rotation}
        
        self.update_point_index()

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
            rotation_command = self.pid_steering.compute(error_distance, delta_time=1/30)
            rotation_command = clamp(rotation_command, -1.0, 1.0)

            # Contrôleur PID pour la vitesse
            forward_command = self.pid_forward.compute(front_dist - K, delta_time=1/30)
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


    def isFinish(self,path):
        if path is None:
            return True
        if self.current_target_index >= path.length():
             self.current_target_index = 0
             return True
        return False
    
    def isFinishPose(self,path : Path,pose : Pose):
        if path is None:
            return True
        
        path = path._poses[:, :2]
        target_poses = path - pose.position
        distances = np.linalg.norm(target_poses, axis=1)
        nearest_index = np.argmin(distances)

        if(nearest_index == np.size(path)-2):
            return True
        return False

    
    def update_point_index(self):
        # Si le drone est suffisamment proche du point cible autour de 3 pixel
        if(self.error_distance < self.tolerance):
            # Change de point
            self.current_target_index += 1
    
    def find_closest_point_index(self, current_pose, path):
        """Trouve le point du chemin le plus proche du drone."""
        min_distance = float('inf')
        closest_index = self.current_target_index
        for i in range(self.current_target_index,path.length()):
            point = path.get(i)
            distance = np.linalg.norm(np.array(point.position) - np.array(current_pose.position))
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index
    
    def update_current_path_index(self,n):
        self.current_target_index = n
    

                



