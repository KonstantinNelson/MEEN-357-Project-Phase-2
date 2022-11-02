import scipy.interpolate as inte
import numpy as np
import matplotlib.pyplot as plt
from define_experiment import experiment1

#get the experiment and end_event dicts from experiment1
experiment, end_event = experiment1()
#get the alpha_dist array from the experiment dict
alpha_dist = experiment['alpha_dist']
#get the alpha_deg array fromt the experiment dict
alpha_deg = experiment['alpha_deg']

#define alpha_fun to interpolate the values of alpha_dist and alpha_deg
alpha_fun = inte.interp1d(alpha_dist,alpha_deg,kind='cubic',fill_value='extrapolate')

#define dist as an array of zeros
dist = np.linspace(alpha_dist[0],alpha_dist[-1],100)
#approximate the values of deg with dist and the interpolation function
deg = alpha_fun(dist)

#plot the data from alpha_dist and alpha_deg, and plot the interpolated values of deg
plt.plot(dist,deg)
plt.scatter(alpha_dist,alpha_deg,marker="*")
plt.xlabel('Position (m)')
plt.ylabel('Terrain Angle (deg)')
plt.title('Terrain Angle vs. Position of Rover')
plt.savefig('terrain_angle_vs_position.png',format='png')
plt.show()

