import math
import numpy as np
from simple_pid import PID
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose

class PathTracker:

    def __init__(self):

        # PID controllers for forward and lateral forces
        """
        Kp : Augmente la réactivité. Plus Kp est élevé, plus le système répond vite à une erreur, mais attention aux oscillations si Kp est trop grand.
        Ki : Corrige les erreurs résiduelles (offset), mais peut ralentir le système ou causer un "windup".
        Kd : Aide à amortir les oscillations causées par un Kp élevé. Augmenter Kd améliore la stabilité pour un Kp plus grand.
        """
        self.pid_forward = PID(Kp=0.03, Ki=0.0001, Kd=0.01, setpoint=0) 
        self.pid_lateral = PID(Kp=0.03, Ki=0.0001, Kd=0.01, setpoint=0)

        # Limiter la commande alpha entre -1 et 1
        self.pid_forward_output_limits = (-1, 1) 
        self.pid_lateral_output_limits = (-1, 1)

        # PID pour l'angle de direction
        self.pid_steering = PID(5, 0.162, 100, setpoint=0)
        self.pid_steering.output_limits = (-1, 1)  # Limiter l'angle de rotation entre -1 et 1

        self.current_target_index = 0  # Indice du point cible actuel
        self.iter_path = 0
        self.path_done = Path()
        self.prev_diff_position = 0
        self.isturnRight = True
        self.error_distance = 0

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
    
    def control_pursuit(self, current_pose : Pose ,path : Path,delta_time):
        
         # Si le drone est suffisamment proche du point cible autour de 3 pixel
        if(self.error_distance < 3):
            # Change de point
            self.current_target_index += 1
        
        target_pose = path.get(self.current_target_index)

        # Calculez les erreurs latérales et d'orientation par rapport au point cible
        error_x = target_pose.position[0] - current_pose.position[0]
        error_y = target_pose.position[1] - current_pose.position[1]
        target_theta = np.arctan2(error_y,error_x)
        theta = current_pose.orientation

        # Calculez les erreurs dans la base B1(x1,y1) du drone
        error_theta = target_theta - theta 
        error_x1 = np.cos(theta)*error_x + np.sin(theta)*error_y

        self.error_distance = (error_x**2 + error_y**2)**0.5
        
        # Update PID controllers
        rotation = self.pid_steering(error_theta,delta_time)
        forward = 0

        if abs(error_theta)< 0.3:
            rotation = 0
            forward = self.pid_forward(-error_x1, delta_time)
        
        # Limit PID outputs to [-1, 1]
        forward = max(min(forward, 1), -1)
        rotation = max(min(rotation, 1), -1)
        
        command = {"forward": forward,
        "lateral": 0,
        "rotation": rotation,
        "grasper":0}

        return command


    def control(self, current_pose : Pose ,path : Path,delta_time):

        """Aller d'un point A à B à l'aide d'un contrôleur PID"""

        # Débeuger au cas où il n'y a pas de valeurs ou que tous les points sont atteints
        if path.length() == 0 or self.isFinish(path):
            return {"forward": 0, "lateral": 0, "rotation": 0}
        
         # Si le drone est suffisamment proche du point cible autour de 3 pixel
        if(self.error_distance < 8):
            # Change de point
            self.current_target_index += 1
        
        # Pour dessiner le chemin
        self.iter_path += 1
        if self.iter_path % 3 == 0:
            position = np.array([current_pose.position[0],
                                 current_pose.position[1]])
            angle = current_pose.orientation
            pose = Pose(position=position, orientation=angle)
            self.path_done.append(pose)

        target_pose = path.get(self.current_target_index)

        # Calculez les erreurs latérales et d'orientation par rapport au point cible
        error_x = target_pose.position[0] - current_pose.position[0]
        error_y = target_pose.position[1] - current_pose.position[1]
        target_theta = np.arctan2(error_y,error_x)
        theta = current_pose.orientation

        # Calculez les erreurs dans la base B1(x1,y1) du drone
        error_theta = target_theta - theta 
        error_x1 = np.cos(theta)*error_x + np.sin(theta)*error_y
        error_y1 = -np.sin(theta)*error_x + np.cos(theta)*error_y

        self.error_distance = (error_x**2 + error_y**2)**0.5

        # Update PID controllers
        forward = self.pid_forward(-error_x1, delta_time)
        lateral = self.pid_lateral(-error_y1, delta_time)

        # Limit PID outputs to [-1, 1]
        forward = max(min(forward, 1), -1)
        lateral = max(min(lateral, 1), -1)

        command = {"forward": forward,
            "lateral": lateral,
            "rotation": 0,
            "grasper":0}

        return command

    def isFinish(self,path):
        if self.current_target_index >= path.length():
             self.current_target_index = 0
             return True
        return False
    

                



