import math
from math import sqrt
from copy import deepcopy
from typing import Optional
from heapq import heappush, heappop
from random import randrange

from ompl.base import *
from ompl.geometric import *

import numpy as np

# TODO : Utiliser OMPL se rendre d'un point A à B connaissant les collisions
class PathPlanner:
    def __init__(self):
        pass

    def plan_path(self, start, goal, map_data): # map_data mettre dans map_data l'ensemble des collision de la premiére map meme si il est pas senser la connaitre
        """
        Implémente OMPL pour planifier un chemin.
        """

        path = ''
        return path
