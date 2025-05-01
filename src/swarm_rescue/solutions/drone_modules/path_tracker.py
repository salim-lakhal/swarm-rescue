import math
import numpy as np
import time
#from simple_pid import PID
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.utils import clamp
from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.timer import Timer

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

        self.collision_cooldown = 5
        self.last_collision_time = 0

        self.lidar = None

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

    def control(self, current_pose : Pose ,path : Path,delta_time,lidar,velocity = np.array([1,1]),lidar_values=np.zeros(181),):

        self.lidar = lidar

        """Aller d'un point A à B à l'aide d'un contrôleur PID"""

        # Débeuger au cas où il n'y a pas de valeurs ou que tous les points sont atteints
        if path.length() == 0 or self.isFinish(path):
            return {"forward": 0, "lateral": 0, "rotation": 0}
        
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
        
        #current_time = time.time()
        #command,collision = self.process_lidar_sensor(lidar)


        """if collision :
            if (current_time - self.last_collision_time > self.collision_cooldown):
                self.last_collision_time = current_time
                self.finish(path)
                #command = {"forward": 0, "lateral": 0, "rotation": 0}
            return command # Aller contre le mur mettre ça dans attenuation"""

        command = {"forward": forward,
            "lateral": lateral,
            "rotation": rotation}
        
        command = self.handleCollision(command,lidar,path)
        
        self.update_point_index()

        return command


    def follow_target(self, current_pose: Pose, target_position, d_suivi=20.0):
        import numpy as np

        # 1. Position actuelle et orientation
        x, y = current_pose.position
        theta = current_pose.orientation  # en radians

        # 2. Vecteur vers la cible (dans B0)
        dx = target_position[0] - x
        dy = target_position[1] - y
        v_global = np.array([dx, dy])
        distance = np.linalg.norm(v_global)

        if distance < 1e-6:
            v_global = np.array([0.0, 0.0])
            distance = 0.0

        # 3. Transformation du vecteur dans la base locale B1
        R = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta),  np.cos(-theta)]
        ])
        v_local = R @ v_global

        # 4. Générer la commande forward/lateral avec atténuation de vitesse selon la distance
        if distance > d_suivi:
            v_local = v_local / distance  # direction unitaire
            intensity = (distance - d_suivi) / d_suivi  # plus on est loin, plus on va vite
            intensity = min(intensity, 1.0)
            forward = v_local[0] * intensity
            lateral = v_local[1] * intensity
        else:
            forward = 0.0
            lateral = 0.0

        # 5. Atténuation selon les obstacles détectés par le lidar
        #forward, lateral = self.attenuation(forward, lateral)

        command = {"forward": forward, "lateral": lateral}
        return command
    
    def handleCollision(self,command,lidar,path):
        
        # Atténuation

        u = np.array([command["forward"],command["lateral"]])

        values = lidar.get_sensor_values()
        ray_angles = lidar.ray_angles
        size = lidar.resolution
        seuil_ralentir = 70
        seuil_stop = 25
        k = 3
        current_time = time.time()

        if size != 0:
            d = min(values)
            # near_angle_raw : angle with the nearest distance
            alpha = ray_angles[np.argmin(values)]
            v_obs = np.array([np.cos(alpha),np.sin(alpha)])
            u_parallele = (u @ v_obs)*v_obs
            u_perp = u - u_parallele
        
        attenuation = min(1.0, np.exp(- (1 - d / seuil_ralentir) * k)) #min(1.0,(d/seuil_ralentir)**2)

        # Fonction d'atténuation non linéaire
        if d <= seuil_stop:
            
            if (current_time - self.last_collision_time > self.collision_cooldown):
                self.last_collision_time = current_time
                self.finish(path)

            attenuation = -0.5  # frein ou marche arrière

            attenuation = 0.0  # on bloque l'avancement vers l'obstacle

            # === Nouvelle stratégie : diriger vers zone la plus libre ===
            alpha_max = ray_angles[np.argmax(values)]
            v_free = np.array([np.cos(alpha_max), np.sin(alpha_max)])

            redirection_strength = 1.0  # intensité de la redirection
            u = redirection_strength * v_free  # on oriente totalement vers la zone libre
        else :
            u = attenuation*u_parallele + u_perp

        forward = max(min(u[0], 1), -1)
        lateral = max(min(u[1], 1), -1)

        command["forward"] = forward
        command["lateral"] = lateral

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

    def process_lidar_sensor(self, the_lidar_sensor): 
        command = {"forward": 1.0,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller = 1.0
        seuil_ralentir = 200
        seuil_stop = 30

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
        # Calcul projection avant/arrière
        front_projection = np.cos(near_angle)  # +1 si pile devant, 0 si latéral, -1 derrière
        if size != 0 and min_dist < seuil_stop:
            collision = True

            # Éloignement proportionnel, dirigé par la position angulaire
            command["forward"] = - front_projection * (1.0 - min_dist / seuil_stop)  # recule si obstacle en face
            command["lateral"] = - np.sin(near_angle) * (1.0 - min_dist / seuil_stop)  # pousse latéralement si côté

            if near_angle > 0:
                command["rotation"] = -angular_vel_controller
            else:
                command["rotation"] = angular_vel_controller
        elif size != 0 and min_dist < seuil_ralentir:
            # Distance proche, on ralentit proportionnellement
            facteur = (min_dist - seuil_stop) / (seuil_ralentir - seuil_stop)  # de 0 à   # de 0 (près) à 1 (loin)
            command["forward"] = front_projection * facteur  # ralenti en approche si obstacle en face
            command["lateral"] = np.sin(near_angle) * facteur  # ajuste latéral si besoin
            command["rotation"] = 0.0

        return command, collision

    def isFinish(self,path):
        if path is None:
            return True
        if self.current_target_index >= path.length():
             self.current_target_index = 0
             return True
        return False
    
    def finish(self,path):
        self.current_target_index = path.length()
    
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
    

                



