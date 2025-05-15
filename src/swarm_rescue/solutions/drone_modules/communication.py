import math
import numpy as np
import time
#from simple_pid import PID
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.utils import clamp
from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.timer import Timer


class Communication:

    def __init__(self,identifer):

        self.identifier = identifer
        self.iter = 0
        self.communicator = None
        self.lidar = None
        self.pose = None
    
    def update(self,communicator, pose : Pose):
        self.pose = pose
        self.communicator = communicator

    
    def findNearestDrone(self):
        # Calcul du plus proche voisin
        if self.communicator is None :
            return self.identifier
        
        min_dist = float("inf")
        nearest_id = None
        received_messages = self.communicator.received_messages
        for msg in received_messages:
            _, (sender_id, (other_pos, _)) = msg
            dist = ((self.pose.position[0] - other_pos[0]) ** 2 + (self.pose.position[1] - other_pos[1]) ** 2) ** 0.5
            if dist < min_dist and sender_id != self.identifier:
                min_dist = dist
                nearest_id = sender_id
        return nearest_id
    
    def asignfrontier(self,msg):
        # Retourne un dictionnaire de frontrière avec un id et la coordonnée de la frontiére à explorer

        return {"0":(10,10),"1":None}

    def getMessage(self, robot_id):
        if self.communicator is None:
            return None

        for msg in self.communicator.received_messages:
            _, (sender_id, (other_pos, _)) = msg
            if sender_id == robot_id:
                _,message = msg
                return msg  # ou message[1] si tu veux juste la position/orientation
        return None

    def getNearestDrone(self):
        return self.findNearestDrone()
    
    def getId(_,msg):
        if msg is None :
            return None
        _, (sender_id, (other_pos, _)) = msg
        return sender_id
    
    def getGPSData(self,robot_id):
        msg = self.getMessage(robot_id)
        if msg is None:
            return None
        _, (sender_id, (other_pos, _)) = msg
        return other_pos


