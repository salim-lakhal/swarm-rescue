import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class KalmanFilter:

    def __init__(self,initial_position):
        self.iter = 0
        # Paramètre du filtre exponentiel
        self.alpha = 0.7  # Facteur de lissage (ajuste selon tes besoins)
        # Paramètres
        self.phi = 0.98  # Matrice de coefficients AR(1)
        sigmaQ = sqrt(0.40)    # Covariance du bruit d'évolution (eta_k)
        self.sigmaR = 5 # Covariance du bruit blanc (epsilon_k)
        self.Q_kalman = np.array([[0.39,0],[0,30**2]])  # Covariance du bruit de processus
        # Matrices du modèle
        self.F = np.eye(2) 
        self.F[1,1] = self.phi
        self.B = np.array([1,0])
        self.H = np.array([1,1]).reshape(1, -1) 
        x0 = initial_position[0]  +  np.random.normal(0, 1)
        wx0 = np.random.normal(0, self.sigmaR)
        y0 = initial_position[1]  +  np.random.normal(0, 1)
        wy0 = np.random.normal(0, self.sigmaR)

        self.x_est = np.array([x0,wx0])
        self.y_est = np.array([y0,wy0])


        self.Px_est = np.eye(2)         
        self.Px_est[0,0] = 0.40
        self.Px_est[1,1] = 5

        self.Py_est = np.eye(2)         
        self.Py_est[0,0] = 0.40
        self.Py_est[1,1] = 5

        self.v_pred = np.array([0,0])
        self.v_filtre = np.array([0,0])
    
    def fk(self,p,v):
        return np.array([self.fkx(p[0],v[0]),self.fky(p[1],v[1])])


    def fkx(self,x,vx):
        # Prédiction
        x_pred = self.F @ self.x_est + vx*self.B
        P_pred = self.F @ self.Px_est @ self.F.T + self.Q_kalman

        # Mise à jour
        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T)  # Gain de Kalman
        self.x_est = x_pred + K @ (x - self.H @ x_pred)
        self.Px_est = (np.eye(2) - K @ self.H) @ P_pred

        return self.x_est[0]
    
    def fky(self,y,vy):
        # Prédiction
        y_pred = self.F @ self.y_est + vy*self.B
        P_pred = self.F @ self.Py_est @ self.F.T + self.Q_kalman

        # Mise à jour
        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T)  # Gain de Kalman
        self.y_est = y_pred + K @ (y - self.H @ y_pred)
        self.Py_est = (np.eye(2) - K @ self.H) @ P_pred

        return self.y_est[0]
    
    def fv(self,v):
        alpha = 0.01
        self.v_filtre =  (alpha/(alpha + 1))*v + (alpha/(alpha + 1)) * self.v_pred - ((alpha-1)/(alpha + 1)) * self.v_filtre
        self.v_pred = v
        return self.v_filtre
        
    

    


    