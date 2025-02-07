import numpy as np
import math
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.utils import clamp
from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose


class PurePursuitController:
    def __init__(self, lookahead_distance=30, max_speed=1.0):
        self.lookahead_distance = lookahead_distance  # Distance pour choisir le point cible
        self.max_speed = max_speed  # Vitesse maximale du drone
        self.current_target_index = 0  # Index du point cible actuel
        self.path_done = []  # Pour stocker le chemin parcouru
        self.iter_path = 0  # Compteur pour dessiner le chemin

    def control(self, current_pose: Pose, path: Path, delta_time):
        """Contrôle du drone en utilisant l'algorithme Pure Pursuit."""
        
        # Vérifier si le chemin est vide ou si tous les points sont atteints
        if path.length() == 0 or self.isFinish(path):
            return {"forward": 0, "lateral": 0, "rotation": 0}

        # Trouver le point cible (lookahead point)
        target_pose = self.find_lookahead_point(current_pose, path)

        # Calculer la courbure pour atteindre le point cible
        curvature = self.calculate_curvature(current_pose, target_pose)

        # Convertir la courbure en commandes de contrôle
        forward_speed = self.max_speed
        angular_velocity = curvature * forward_speed

        # Limiter les commandes de contrôle
        forward_speed = max(min(forward_speed, 1), -1)
        angular_velocity = max(min(angular_velocity, 1), -1)

        # Pour dessiner le chemin
        self.iter_path += 1
        if self.iter_path % 3 == 0:
            position = np.array([current_pose.position[0], current_pose.position[1]])
            angle = current_pose.orientation
            pose = Pose(position=position, orientation=angle)
            self.path_done.append(pose)

        # Retourner les commandes de contrôle
        return {"forward": -forward_speed, "lateral": 0, "rotation": angular_velocity, "grasper": 0}

    def find_lookahead_point(self, current_pose: Pose, path: Path):
        """Trouve le point cible à une distance lookahead_distance sur le chemin."""
        for i in range(self.current_target_index, path.length()):
            target_pose = path.get(i)
            distance = np.linalg.norm(target_pose.position - current_pose.position)
            if distance >= self.lookahead_distance:
                self.current_target_index = i
                return target_pose
        return path.get(-1)  # Retourne le dernier point si aucun point n'est trouvé

    def calculate_curvature(self, current_pose: Pose, target_pose: Pose):
        """Calcule la courbure nécessaire pour atteindre le point cible."""
        # Vecteur entre la position actuelle et le point cible
        target_vector = target_pose.position - current_pose.position

        # Angle entre la direction du drone et le vecteur cible
        angle_to_target = np.arctan2(target_vector[1], target_vector[0]) - current_pose.orientation

        # Courbure (inverse du rayon de courbure)
        curvature = 2 * np.sin(angle_to_target) / np.linalg.norm(target_vector)
        return curvature

    def isFinish(self, path: Path):
        """Vérifie si tous les points du chemin ont été atteints."""
        return self.current_target_index >= path.length() - 1