import scipy.interpolate as inte
import numpy as np
import matplotlib.pyplot as plt
from define_experiment import experiment1

experiment, end_event = experiment1()
alpha_dist = experiment['alpha_dist']
alpha_deg = experiment['alpha_deg']

alpha_fun = inte.interp1d(alpha_dist,alpha_deg,kind='cubic',fill_value='extrapolate')

dist = np.linspace(alpha_dist[0],alpha_dist[-1],100)
deg = alpha_fun(dist)

plt.plot(dist,deg)
plt.scatter(alpha_dist,alpha_deg,marker="*")
plt.xlabel('Position (m)')
plt.ylabel('Terrain Angle (deg)')
plt.title('Terrain Angle vs. Position of Rover')
plt.savefig('terrain_angle_vs_position.png',format='png')
plt.show()

