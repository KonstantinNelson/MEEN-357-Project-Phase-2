import math
import numpy as np
import scipy.interpolate as inte
from scipy.integrate import simps, solve_ivp
from define_experiment import *
import matplotlib.pyplot as plt
from subfunctions import *

#import the experiment and end_event dictionaries from the experiment1 function in define_experiment
experiment,end_event = experiment1()
#import the rover and planet dictionaries from the define_rover_1 function in subfunctions
rover,planet = define_rover_1()
#redefine the max_distance termination condition
end_event['max_distance']=1000
#redefine the max_time termination condition
end_event['max_time']=10000
#redefine the min_velocity termination condition
end_event['min_velocity']=0.01

#update the rover dict with telemetry information
rover = simulate_rover(rover,planet,experiment,end_event)

#store the data in the telemetry dictionary in te
te = rover['telemetry']

#make a subplot of position vs time, velocity vs time, and power vs time
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
