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
        self.path_done = Path()
        self.error_distance = 0
        self.tolerance = 30

    # Implémenter plus tard
    def control_with_stanley(self,current_pose:Pose,path:Path,limited_steering_angle,target_index,crosstrack_error,delta_time):

        if path.length() == 0 :#or self.isFinish(path):
            return {"forward": 0, "lateral": 0, "rotation": 0}
        
        if abs(limited_steering_angle - current_pose.orientation)< 0.3:
            rotation = 0
        # Utiliser un PID pour la rotation
        else:
            rotation = self.pid_steering(limited_steering_angle - current_pose.orientation, delta_time)

        # PID pour avancer et latéral
        forward = self.pid_forward(-crosstrack_error, delta_time)
        lateral = 0  # Peut être ajouté selon votre logique

        # Limiter les sorties
        forward = max(min(forward, 1), -1)
        rotation = max(min(rotation, 1), -1)

        command = {
        "forward": forward,
        "lateral": lateral,
        "rotation": rotation
        }

        return command
    
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
    
    # Implémenter plus tard
    def pure_pursuit_control(self, current_pose, path, lookahead_distance, delta_time):
        """
        Implémentation du contrôle Pure Pursuit.
        :param current_pose: Pose actuelle du drone (position et orientation).
        :param path: Trajectoire à suivre.
        :param lookahead_distance: Distance de regard vers l’avant pour trouver le point cible.
        :param delta_time: Temps écoulé depuis la dernière mise à jour.
        :return: Commande du drone (rotation pour suivre la courbure).
        """

        if path.length() == 0:
            return {"forward": 0, "lateral": 0, "rotation": 0}

        """# 1️⃣ Trouver le point du chemin le plus proche
        closest_index = self.current_target_index

        # 2️⃣ Trouver le point cible basé sur le lookahead
        goal_index = closest_index
        while goal_index < path.length() - 1:
            goal_point = path.get(goal_index)
            distance_to_goal = np.linalg.norm(np.array(goal_point.position) - np.array(current_pose.position))

            if distance_to_goal >= lookahead_distance:
                break
            goal_index += 1"""

        # 3️⃣ Transformation du point cible en coordonnées du véhicule
        goal_pose = path.get(self.current_target_index)
        self.update_point_index()
        position_error = np.array(goal_pose.position) - np.array(current_pose.position)
        
        theta = current_pose.orientation
        rotation_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        goal_vehicle_coords = np.dot(rotation_matrix, position_error)

        # 4️⃣ Calcul de la courbure (c = 2 * y / L^2) avec y = coordonnée latérale du point cible
        x_goal, y_goal = goal_vehicle_coords
        if abs(x_goal) < 1e-6:  # Éviter une division par zéro
            curvature = 0
        else:
            curvature = (2 * y_goal) / (lookahead_distance ** 2)

        # 5️⃣ Génération de la commande de rotation
        rotation_command = self.pid_steering.compute(curvature, delta_time)
        rotation_command = clamp(rotation_command, -1.0, 1.0)

        # 6️⃣ Retour de la commande Pure Pursuit
        return {
            "forward": 1.0,  # Peut être ajusté en fonction de la vitesse désirée
            "lateral": 0,
            "rotation": rotation_command,
            "grasper": 0
        }

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

    def isFinish(self,path):
        if self.current_target_index >= path.length():
             self.current_target_index = 0
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
    
    def update_path_done(self,current_pose):
        # Pour dessiner le chemin
        self.iter_path += 1
        if self.iter_path % 3 == 0:
            position = np.array([current_pose.position[0],
                                 current_pose.position[1]])
            angle = current_pose.orientation
            pose = Pose(position=position, orientation=angle)
            self.path_done.append(pose)
    
    def update_current_path_index(self,n):
        self.current_target_index = n
    

                



