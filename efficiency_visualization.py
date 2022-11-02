import scipy.interpolate as inte
import numpy as np
import matplotlib.pyplot as plt
from subfunctions import *

#retrieve the the rover and planet dict from the define_rover_1 function in subfunctions
rover,planet = define_rover_1()
#get the effcy array from the rover dictionary
effcy = rover['wheel_assembly']['motor']['effcy']
#get the effcy_tau array from teh rover dict
effcy_tau = rover['wheel_assembly']['motor']['effcy_tau']

#define a function effcy_fun that interpolates the values of effcy_tau and effcy
effcy_fun = inte.interp1d(effcy_tau,effcy,kind='cubic')

#create an array of 100 values ranging from the minimum torque value to the maximum torque value
tau = np.linspace(effcy_tau[0],effcy_tau[-1],100)
#approximate the intermediate values of eff using the interpolation function and tau
eff = effcy_fun(tau)

#graph the values from the effcy_tau and affcy dicts, and the values of eff from effcy_fun
plt.plot(tau,eff)
plt.scatter(effcy_tau,effcy,marker='*')
plt.xlabel('Torque (N*m)')
plt.ylabel('Efficiency')
plt.title('Motor Efficiency vs. Motor Torque')
plt.savefig('eff_vs_torque.png',format='png')
plt.show()
