import math
import numpy as np
import time
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.utils import clamp
from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.timer import Timer


class SemanticManager:

    def __init__(self,identifer):

        self.identifier = identifer
        self.iter = 0
        self.semantic_values = None
        self.lidar = None
        self.pose = None
        self.list_wounded = []
        self.list_wounded_save = []
        self.current_grasped_entity_pose = None
        self.has_just_grasped = False
    
    def update(self,semantic_values,list_wounded,grasped_entities,communicator, pose : Pose):
        self.pose = pose
        self.semantic_values = semantic_values
        self.list_wounded = list_wounded
        self.grasped_entities = grasped_entities
        self.communicator = communicator

        if not self.grasped_entities:
            self.update_wounded_list()
        else:
            self.removeWoundedFromList(pos=self.current_grasped_entity_pose)
        
        self.handleCommunication()


    def handleCommunication(self):

        if self.communicator is None:
            return None
        
        received_messages = self.communicator.received_messages

        for msg in received_messages:
             _, (sender_id, (other_pos, _),wounded_list_by_other,wounded_saved_by_other,_) = msg

            # 1. Supprimer les blessés que l’autre a déjà sauvés
             for saved in wounded_saved_by_other:
                self.list_wounded = [w for w in self.list_wounded if self.compute_distance_gaussian(w, saved) < 0.5]

                    # Ajouter à list_wounded_save s’il n’y est pas déjà
                if all(self.compute_distance_gaussian(saved, own_saved) < 0.5 for own_saved in self.list_wounded_save):
                    self.list_wounded_save.append(saved)
            # 2. Ajouter les blessés détectés par l’autre drone si on ne les a pas déjà
             for detected in wounded_list_by_other:
                # Vérifie que le blessé n’est ni déjà dans list_wounded, ni dans list_wounded_save
                if all(self.compute_distance_gaussian(detected, w) < 0.5 for w in self.list_wounded) and \
                all(self.compute_distance_gaussian(detected, saved) < 0.5 for saved in self.list_wounded_save):
                    self.list_wounded.append(detected)
        return None


    def update_wounded_list(self):
        detection_semantic = self.semantic_values
        best_score = 0
        best_angle = 0
        best_distance = 0

        if detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and
                        not data.grasped):
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))
                    self.addWoundedToList(new_pos=self.get_wounded_position(data.distance,data.angle))
            
            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]
                    best_distance = score[2]
        
            
            self.current_grasped_entity_pose = self.get_wounded_position(best_distance,best_angle)

            """if not self.has_just_grasped:
                self.grasped_origin_pose = self.current_grasped_entity_pose
                self.has_just_grasped = True
            
            if self.grasped_entities:
                self.removeWoundedFromList(pos=self.current_grasped_entity_pose)"""

            # Rajouter une condition tant qu'on a pas attraper le mec le plus proche
            """if self.grasped_entities and not self.has_just_grasped:
                self.removeWoundedFromList(pos=self.current_grasped_entity_pose)
                self.has_just_grasped = True"""

    def notify_release(self):
        """
        À appeler une fois que le drone dépose le blessé au centre de secours.
        """
        self.has_just_grasped = False
        #self.current_grasped_entity_pose = None
              
    def is_in_woundedList(self, new_pos, sigma: float = 25) -> bool:
        """
        Vérifie si une nouvelle position correspond à un blessé déjà détecté
        en utilisant une distance gaussienne (tolérance par écart-type sigma).

        :param new_pos: Position estimée du nouveau blessé
        :param sigma: Écart-type de tolérance (en mètres)
        :return: True si c’est probablement le même blessé, False sinon
        """
        for known in self.list_wounded:
            distance = math.sqrt((new_pos[0] - known[0])**2 + (new_pos[1] - known[1])**2)
            probability = math.exp(- (distance ** 2) / (2 * sigma ** 2))

            # Seuil à ajuster en fonction de ton besoin ; ici proba > 0.5
            if probability > 0.5:
                return True  # blessé probablement déjà vu
        return False
    
    def compute_distance_gaussian(self, pos1, pos2, sigma: float = 25) -> float:
        """
        Calcule la probabilité que deux positions correspondent au même blessé
        selon un modèle de distribution gaussienne.

        :param pos1: Première position (x, y)
        :param pos2: Deuxième position (x, y)
        :param sigma: Écart-type de l'incertitude spatiale (en mètres)
        :return: Probabilité (entre 0 et 1)
        """
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        probability = math.exp(- (distance ** 2) / (2 * sigma ** 2))
        return probability
    
    def addWoundedToList(self,new_pos):

        if new_pos is None :
            return None

        if not self.is_in_woundedList(new_pos):
            #print(str(new_pos) + "n'était pas dans la list")
            self.list_wounded.append(new_pos)
        else : 
            #print(str(new_pos) + "est déja dans la list")
            return None
        
        return None

    def removeCurrentWoundedFromList(self):
        self.removeWoundedFromList(pos=self.get_wounded_grasped_position())
    
    def removeWoundedFromList(self, pos, sigma: float = 25):
        """
        Supprime un blessé de la liste s'il est suffisamment proche (même logique que is_in_woundedList).
        """
        for known in self.list_wounded:
            distance = math.sqrt((pos[0] - known[0])**2 + (pos[1] - known[1])**2)
            probability = math.exp(- (distance ** 2) / (2 * sigma ** 2))

            if probability > 0.5:
                self.list_wounded.remove(known)
                self.list_wounded_save.append(pos)
                #print(f"Blessé {known} retiré de la liste (grasped)")
                break  # On en retire un seul à la fois
    
    def removeWoundedFromListSave(self, pos, sigma: float = 20):
        """
        Supprime un blessé de la liste s'il est suffisamment proche (même logique que is_in_woundedList).
        """
        for known in self.list_wounded_save:
            distance = math.sqrt((pos[0] - known[0])**2 + (pos[1] - known[1])**2)
            probability = math.exp(- (distance ** 2) / (2 * sigma ** 2))

            if probability > 0.5:
                self.list_wounded_save.remove(known)
                #self.list_wounded_save.append(pos)
                #print(f"Blessé {known} retiré de la liste save (grasped)")
                break  # On en retire un seul à la fois
    

    def is_in_saveWoundedList(self, new_pos, sigma: float = 25) -> bool:
        """
        Vérifie si une nouvelle position correspond à un blessé déjà détecté
        en utilisant une distance gaussienne (tolérance par écart-type sigma).

        :param new_pos: Position estimée du nouveau blessé
        :param sigma: Écart-type de tolérance (en mètres)
        :return: True si c’est probablement le même blessé, False sinon
        """

        for known in self.list_wounded_save:
            distance = math.sqrt((new_pos[0] - known[0])**2 + (new_pos[1] - known[1])**2)
            probability = math.exp(- (distance ** 2) / (2 * sigma ** 2))

            # Seuil à ajuster en fonction de ton besoin ; ici proba > 0.5
            if probability > 0.5:
                return True  # blessé probablement déjà vu
        return False

    def saveWoundedPosition(self,pos):

        if pos is None :
            return None

        if not self.is_in_saveWoundedList(pos):
            #print(str(new_pos) + "n'était pas dans la list")
            self.list_wounded_save.append(pos)
        else : 
            #print(str(new_pos) + "est déja dans la list")
            return None
        
        return None

    def get_wounded_position(self, distance: float, angle_relative: float) -> tuple:
        """
        Calcule la position absolue du blessé à partir de la position du robot, 
        d'une distance et d'un angle relatif.

        :param distance: Distance entre le robot et le blessé
        :param angle_relative: Angle relatif entre le robot et le blessé (en radians)
        :return: Position absolue (x, y) du blessé
        """

        # Orientation absolue du robot (en radians)
        theta_robot = self.pose.orientation  

        # Calcul de l'angle absolu
        angle_absolute = theta_robot + angle_relative

        # Position actuelle du robot
        x_robot = self.pose.position[0]
        y_robot = self.pose.position[1]

        # Coordonnées absolues du blessé
        x_wounded = x_robot + distance * math.cos(angle_absolute)
        y_wounded = y_robot + distance * math.sin(angle_absolute)

        return (x_wounded, y_wounded)

    def get_wounded_grasped_position(self):
        return self.grasped_wounded_position
    
    def getWoundedList(self):
        return self.list_wounded
    
    def getWoundedSaveList(self):
        return self.list_wounded_save