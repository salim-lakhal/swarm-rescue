#### Les classes fournis :

### constants.py :
## Que des constantes qui sont définis 
FRAME_RATE : Taux d'images par seconde 

LINEAR_SPEED_RATIO : Ratio de vitesse linéaire

ANGULAR_SPEED_RATIO : Ratio de vitesse angulaire

LINEAR_SPEED_RATIO_WOUNDED : Ratio de vitesse linéaire en cas de blessure

DRONE_INITIAL_HEALTH : Santé initiale du drone

RESOLUTION_SEMANTIC_SENSOR : Résolution du capteur sémantique

MAX_RANGE_SEMANTIC_SENSOR : Portée maximale du capteur sémantique

FOV_SEMANTIC_SENSOR : Champ de vision du capteur sémantique

RESOLUTION_LIDAR_SENSOR : Résolution du capteur LIDAR

MAX_RANGE_LIDAR_SENSOR : Portée maximale du capteur LIDAR

FOV_LIDAR_SENSOR : Champ de vision du capteur LIDAR

RANGE_COMMUNICATION : Portée de communication

### fps_dislay : 

Osef c'est pour les fps

### grid.py :

Class Grid :
- size_area_world : Couple donnant la dimension du monde
- resoluton : float donnant la résolution
- x_max_grid : Donne le x maximum
- y_max_grid : Donne le y maximum
- grid : Matrice de dimension x_max_grid X y_max_grid

Méthodes : 
- _conv_world_to_grid(self,x_world,y_world) -----> x_grid , y_max_grid
Converti les coordonnées réels du mondes dans les coordonnées de la grille

- _conv_grid_to_world(self,x_grid,y_grid)   -----> x_world , y_world
Converti les coorodnnées de la grille dans les coordonnées du monde

- add_value_along_line(self, x_0 : float, y_0 : float, x_1 : float, y_1 : float, val) -----> modifie le self.grid
Ajoute une valeur dans une ligne de point utilisant l'algo de Bresenham (osef je pense)

- add_points(self,points_x,points_y,val)   -----> modifie le self.grid
Ajoute une valeur dans une matrice de points avec une valeur particulière

- display(self,grid_to_display : np.ndarray, robot_pose : Pose, title="grid")  -----> Affichage de la grille et du robot


### misc_data.py :

class MiscData :
- size_area : Couple de flottant indiquant la taille de l'aire où le drone est en train d'opérer
- number_drones : donne le nombre de drones
- max_timestep_limit : limite de temps 
- max_walltime_limite : limite de temps

### mouse_measure.py : 
osef

### path.py :

Class Path : 
- poses : tableau à 3 colonnes | X , Y , Orientation

Méthodes : 

- Append(self, pose : Pose) -----> Ajoute une position à poses

- length(self) -----> Dimensions du tableau poses

- get(self, index : int) -----> Récupère un Pose en particulier en fonction de l'index

- reset(self) ----> réintialise poses

### pose.py :

Class Positon :
- data : données de la posiiton sous la forme d'un tableau numpy 1x2

Méhodes :

- __getitem__(self,key) -----> self.data[key]

- __setitem__(self,key,value) ----> change la valeur de key

- __repr__(self) ----> Affiche la position

- set(self,x,y) ----> Initialise la position avec x et y

- x(self) ----> renvoie x

- y(self) -----> renvoie y

Class Pose :
- position = np.zeros(2, )  
- orientation = 0.0

### timer.py :

osef

### utils.py :

- normalize_angle(angle, __(zero_2_2pi) optionnel__ ) ----> normalise l'angle

- sign(x) -----> retourne le signe de x

- rad2deg(angle) -----> converti les radians en degrés

- deg2rad(angle) -----> converti les degrés en radians

- circular_mean(angles) ----> retourne l'angle moyen

- bresenham(start,end) ----> renvoie une liste de points
implémente l'algorithme de bresenham afin de tracer une ligne entre start et end en revnoyant une liste de points.

- circular_kernel(radius) ----> Aucune idée

- clamp(val, min_val, max_val) -----> retourne la valeur intermédiaire entre 3 valeurs

## Modules des drones :

### drone_distance_sensors.py :

