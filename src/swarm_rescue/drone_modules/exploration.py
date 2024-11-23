import math
from math import sqrt
from copy import deepcopy
from typing import Optional
from heapq import heappush, heappop
from random import randrange

import numpy as np

#TODO : Salim Exploration aléatoire Wall-Following
class Exploration:
    def __init__(self, slam):
        self.slam = slam  # Utilise les données de la carte SLAM

    def compute_next_move(self, position):
        """
        Calcule le prochain mouvement basé sur la logique d'exploration.
        """
        # Implémentez Wall-Following ou une autre stratégie
        return {"forward": 1.0, "lateral": 0.0, "rotation": 0.5}