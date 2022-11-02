import math
import numpy as np
import scipy.interpolate as inte
from scipy.integrate import simps, solve_ivp
from define_experiment import *
import matplotlib.pyplot as plt
from subfunctions import *

end_event['max_distance']=1000
end_event['max_time']=10000
end_event['min_velocity']=0.1

rover = simulate_rover(rover,planet,experiment,end_event)

te = rover['telemetry']


f = plt.figure(figsize=(8,7))
ax=f.add_subplot(311,xlabel='Time [s]',ylabel='Position [m]')
ax.title.set_text('Rover Position vs. Time')
ax2=f.add_subplot(312,xlabel='Time [s]',ylabel='Velocity [m/s]')
ax2.title.set_text('Rover Velocity vs. Time')
ax3=f.add_subplot(313,xlabel='Time [s]',ylabel='Mechanical Power [W]')
ax3.title.set_text('Mechanical Power vs. Time')
ax.plot(te['Time'],te['position'])
ax2.plot(te['Time'],te['velocity'])
ax3.plot(te['Time'],te['power'])
plt.tight_layout()
plt.savefig('Task8.png',format='png')
