import scipy.interpolate as inte
import numpy as np
import matplotlib.pyplot as plt
from rover_dict import *

effcy = rover['wheel_assembly']['motor']['effcy']
effcy_tau = rover['wheel_assembly']['motor']['effcy_tau']

effcy_fun = inte.interp1d(effcy_tau,effcy,kind='cubic')

tau = np.linspace(effcy_tau[0],effcy_tau[-1],100)
eff = effcy_fun(tau)

plt.plot(tau,eff)
plt.scatter(effcy_tau,effcy,marker='*')
plt.xlabel('Torque (N*m)')
plt.ylabel('Efficiency')
plt.title('Motor Efficiency vs. Motor Torque')
plt.savefig('eff_vs_torque.png',format='png')
plt.show()
