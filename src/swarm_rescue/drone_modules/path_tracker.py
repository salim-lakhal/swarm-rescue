import math
from math import sqrt
from copy import deepcopy
from typing import Optional
from heapq import heappush, heappop
from random import randrange

import numpy as np

# TODO : Salim ou Younes Utiliser Stanley Control pour liser la trajectoire
class PathTracker:
    def __init__(self):
        pass

    def follow_path(self, path, current_position): # comment suivre le chemin
        """
        Utiliser Stanley Control
        """
        return {"forward": 1.0, "lateral": 0.0, "rotation": 0.2}
