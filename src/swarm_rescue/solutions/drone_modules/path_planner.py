import math
from math import sqrt
from copy import deepcopy
from typing import Optional
from heapq import heappush, heappop
from random import randrange
from spg_overlay.utils.grid import Grid
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
import random
import numpy as np
from solutions.drone_modules.occupancy_grid import OccupancyGrid
from random import uniform


class Node():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.children = []
        self.parent = None

class RRT():
    def __init__(self,start,goal,numIteration,grid,stepSize,x_max,y_max):
        self.randomTree = Node(start[0],start[1])
        self.goal = Node(goal[0],goal[1])
        self.nearestNode = None
        self.iterations = min(numIteration,200)
        self.grid = grid
        self.x_max_grid = x_max
        self.y_max_grid = y_max
        self.rho = stepSize
        self.pathDist = 0
        self.nearestDist = 100000
        self.numPath = 0
        self.path = []
    
    def distance(self,node1, node2):
        return math.sqrt((node1.x - node2[0])**2 + (node1.y - node2[1])**2)

    def addChild(self,x,y):
        if (x == self.goal.x):
            self.nearestNode.children.append(self.goal)
            self.goal.parent = self.nearestNode
        else:
            tempNode = Node(x,y)
            self.nearestNode.children.append(tempNode)
            tempNode.parent = self.nearestNode
    
    def sampleAPoint(self):
        x = random.randint(1,self.y_max_grid)
        y = random.randint(1,self.x_max_grid)
        point = np.array([x,y])
        return point
    def steerToPoint(self,locationStart, locationEnd):
        offset = self.rho*self.unitVector(locationStart,locationEnd)
        point = np.array([locationStart.x + offset[0], locationStart.y + offset[1]])
        if point[0] >= self.y_max_grid:
            point[0] = self.y_max_grid -1
        if point[1] >= self.x_max_grid:
            point[1] = self.x_max_grid -1
        return point
    def isInIbstacle(self,locationStart, locationEnd):
        u = self.unitVector(locationStart,locationEnd)
        testPoint = np.array([0.0,0.0])
        for i in range(self.rho):
            testPoint[0] = locationStart.x + i*u[0]
            testPoint[1] = locationStart.y + i*u[1]
            if self.grid[round(testPoint[1]).astype(np.int64),round(testPoint[0]).astype(np.int64)] == 1:
                return True
        return False
    def unitVector(self,locationStart, locationEnd):
        v = np.array([locationEnd[0]-locationStart.x, locationEnd[1] - locationStart.y])
        u = v/np.linalg.norm(v)
        return u
    def findNearest(self,root,point):
        if not root:
            return
        dist = self.distance(root,point)
        if dist <= self.nearestDist:
            self.nearestNode = root
            self.nearestDist = dist
        for child in root.children:
            self.findNearest(child,point)
        pass

    def goalFound(self,point):
        if self.distance(self.goal , point) <= self.rho:
            return True
        pass

    def resetNearestValues(self):
        self.nearestNode = None
        self.nearestDist = 10000

    def retraceRRTPath(self,goal):
        if goal.x == self.randomTree.x:
            return
        self.numPath += 1
        currentPoint = np.array([goal.x,goal.y])
        self.path.insert(0,currentPoint)
        self.pathDist += self.rho
        self.retraceRRTPath(goal.parent)

        