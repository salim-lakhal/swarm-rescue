import math
from math import pi
from copy import deepcopy

#TODO : Salim Exploration aléatoire Wall-Following
class Exploration:
    def __init__(self, slam):
        """+
        Initialise l'algorithme d'exploration.

        :param slam: Instance SLAM, contenant la carte actuelle et la position estimée.
        """
        self.slam = slam  # Utilise les données de la carte SLAM
        self.wall_distance_threshold = 50  # Distance cible pour suivre les murs (en pixels)
        self.lidar_resolution = 181  # Résolution du capteur LiDAR (par défaut)
        self.following_wall = False  # Indique si le drone suit un mur
        self.direction = 1  # 1 pour sens horaire, -1 pour anti-horaire

    def compute_next_move(self, position, lidar_data):
        """
        Calcule le prochain mouvement basé sur la logique de Wall-Following.

        :param position: La position actuelle du drone (x, y, orientation).
        :param lidar_data: Liste des distances relevées par le capteur LiDAR.
        :return: Dictionnaire {"forward": float, "lateral": float, "rotation": float}.
        """
        forward_speed = 1.0
        lateral_speed = 0.0
        rotation_speed = 0.0

        # Vérifie si des obstacles sont détectés
        min_distance, min_angle = self.get_closest_obstacle(lidar_data)

        if min_distance < self.wall_distance_threshold:
            # Active le mode de suivi de mur si un mur est proche
            self.following_wall = True

            # Ajuste la trajectoire pour suivre le mur
            if min_angle < 0:  # Obstacle à gauche
                rotation_speed = 0.5 * self.direction  # Tourne légèrement pour s'éloigner
            else:  # Obstacle à droite
                rotation_speed = -0.5 * self.direction

            forward_speed = 0.8  # Avance doucement
        else:
            # Pas de mur détecté à une distance proche
            if self.following_wall:
                # Continue à suivre le mur
                rotation_speed = -0.3 * self.direction
            else:
                # Exploration libre si aucun mur n'est détecté
                rotation_speed = 0.5 * self.direction  # Tourne pour scanner la zone

        return {
            "forward": forward_speed,
            "lateral": lateral_speed,
            "rotation": rotation_speed,
        }

    def get_closest_obstacle(self, lidar_data):
        """
        Analyse les données LiDAR pour trouver l'obstacle le plus proche.

        :param lidar_data: Liste des distances relevées par le capteur LiDAR.
        :return: Tuple (distance minimale, angle associé).
        """
        min_distance = float("inf")
        min_angle = 0

        for i, distance in enumerate(lidar_data):
            angle = -pi + (2 * pi / len(lidar_data)) * i
            if distance < min_distance:
                min_distance = distance
                min_angle = angle

        return min_distance, min_angle
