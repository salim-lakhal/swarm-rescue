import math
from math import sqrt
from copy import deepcopy
from typing import Optional
from heapq import heappush, heappop
from random import randrange
import random
import ompl.src.ompl.base as ob
import ompl.src.ompl.geometric as og
import numpy as np
from examples import example_mapping
from examples import example_return_area
from drone_modules import occupancy_grid

# TODO : Utiliser OMPL se rendre d'un point A à B connaissant les collisions
# Solution provisoire : Djikstra
class PathPlanner:
    def __init__(self,map,position,return_area):
        self.map = map # Nuage de points
        self.position = position # Position du drone
        self.return_area = return_area # Aire de retour 
        pass
    
    def distance(self, a, b):
        return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) #Calcul de la distance

    def mapping(self):
        graph = {} # graphe stocké sous forme de dictionnaire | clé : couple de deux points | valeur : distance entre les deux
        for i in self.map:
            for j in self.map:
                if i != j and self.distance(i, j) < 2: # Mur d'épaisseur minimal de 2 | Ainsi on évite tout les points qui dépasserait un mur
                    key1 = (i, j)
                    key2 = (j, i)
                    if key1 not in graph and key2 not in graph:
                        graph[key1] = self.distance(i, j)
        return graph
    
    def djikstra(self):
        """Algorithme de Dijkstra pour trouver le chemin le plus court."""
        graphe = self.mapping()
        queue = []
        path_distance = {self.position: 0}
        path = {self.position: None}
        heappush(queue, (0, self.position))

        while queue:
            current_distance, current_node = heappop(queue)

            if current_node in self.return_area:
                return path

            for (node1, node2), dist in graphe.items():
                if node1 == current_node:
                    voisin = node2
                    new_distance = current_distance + dist
                    if voisin not in path_distance or new_distance < path_distance[voisin]:
                        path_distance[voisin] = new_distance
                        path[voisin] = current_node
                        heappush(queue, (new_distance, voisin))

        return None
    
    def plan_path(self):
        """On trouve tout les points pour avoir le plus court chemin entre start et goal"""
        path = self.djikstra()
        result = []
        if path is None:
            return result
        current = self.return_area
        while current is not None:
            result.append(current)
            current = path[current]
        return result[::-1]
    
    def plan_path(self, start, goal, map_data): # map_data mettre dans map_data l'ensemble des collision de la premiére map meme si il est pas senser la connaitre
        """
        Implémente OMPL pour planifier un chemin.
        """        
        path = ''
        return path

    # Vérifie si un point est valide
    def validState(state, map_data):
        for obstacle in map_data:
            if state == obstacle:
                return False
        return True
    
    #Fonctionnant retournant une liste de 10 points voisins dans un rayon de 5x5 
    def randomPoint(point):
        resp = []
        for i in range(10):
            resp.append(point[0]+random.randint(0,10)-5,point[1]+random.randint(0,10)-5)
        return resp
    
    
