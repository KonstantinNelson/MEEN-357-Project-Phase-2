
import math
import numpy as np
import scipy.interpolate as inte
from scipy.integrate import simps, solve_ivp
from define_experiment import *
import matplotlib.pyplot as plt

def define_rover_1():   # or define_rover()
    # Initialize Rover dict for testing
    wheel = {'radius':0.30,
             'mass':1}
    speed_reducer = {'type':'reverted',
                     'diam_pinion':0.04,
                     'diam_gear':0.07,
                     'mass':1.5}
    motor = {'torque_stall':170,
             'torque_noload':0,
             'speed_noload':3.80,
             'mass':5.0, 'effcy_tau': np.array([0,10,20,40,75,165]),
             'effcy': np.array([0,.6,.75,.73,.55,.05])
             }
        
    chassis = {'mass':659}
    science_payload = {'mass':75}
    power_subsys = {'mass':90}
    
    wheel_assembly = {'wheel':wheel,
                      'speed_reducer':speed_reducer,
                      'motor':motor}
    
    rover = {'wheel_assembly':wheel_assembly,
             'chassis':chassis,
             'science_payload':science_payload,
             'power_subsys':power_subsys}
    
    planet = {'g':3.72}
    
    # return everything we need
    return rover, planet

def get_mass(rover):
    """
    Inputs:  rover:  dict      Data structure containing rover parameters
    
    Outputs:     m:  scalar    Rover mass [kg].
    """
    
    # Check that the input is a dict
    if type(rover) != dict:
        raise Exception('Input must be a dict')
    
    # add up mass of chassis, power subsystem, science payload, 
    # and components from all six wheel assemblies
    m = rover['chassis']['mass'] \
        + rover['power_subsys']['mass'] \
        + rover['science_payload']['mass'] \
        + 6*rover['wheel_assembly']['motor']['mass'] \
        + 6*rover['wheel_assembly']['speed_reducer']['mass'] \
        + 6*rover['wheel_assembly']['wheel']['mass'] \
    
    return m


def get_gear_ratio(speed_reducer):
    """
    Inputs:  speed_reducer:  dict      Data dictionary specifying speed
                                        reducer parameters
    Outputs:            Ng:  scalar    Speed ratio from input pinion shaft
                                        to output gear shaft. Unitless.
    """
    
    # Check that 1 input has been given.
    #   IS THIS NECESSARY ANYMORE????
    
    # Check that the input is a dict
    if type(speed_reducer) != dict:
        raise Exception('Input must be a dict')
    
    # Check 'type' field (not case sensitive)
    if speed_reducer['type'].lower() != 'reverted':
        raise Exception('The speed reducer type is not recognized.')
    
    # Main code
    d1 = speed_reducer['diam_pinion']
    d2 = speed_reducer['diam_gear']
    
    Ng = (d2/d1)**2
    
    return Ng


def tau_dcmotor(omega, motor):
    """
    Inputs:  omega:  numpy array      Motor shaft speed [rad/s]
             motor:  dict             Data dictionary specifying motor parameters
    Outputs:   tau:  numpy array      Torque at motor shaft [Nm].  Return argument
                                      is same size as first input argument.
    """
    
    # Check that 2 inputs have been given
    #   IS THIS NECESSARY ANYMORE????
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception ('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')

    # Check that the second input is a dict
    if type(motor) != dict:
        raise Exception('Second input must be a dict')
        
    # Main code
    tau_s    = motor['torque_stall']
    tau_nl   = motor['torque_noload']
    omega_nl = motor['speed_noload']
    
    # initialize
    tau = np.zeros(len(omega),dtype = float)
    for i in range(len(omega)):
        if omega[i] >= 0 and omega[i] <= omega_nl:
            tau[i] = tau_s - (tau_s-tau_nl)/omega_nl *omega[i]
        elif omega[i] < 0:
            tau[i] = tau_s
        elif omega[i] > omega_nl:
            tau[i] = 0
        
    return tau
    
    


def F_rolling(omega, terrain_angle, rover, planet, Crr):
    """
    Inputs:           omega:  numpy array     Motor shaft speed [rad/s]
              terrain_angle:  numpy array     Array of terrain angles [deg]
                      rover:  dict            Data structure specifying rover 
                                              parameters
                    planet:  dict            Data dictionary specifying planetary 
                                              parameters
                        Crr:  scalar          Value of rolling resistance coefficient
                                              [-]
    
    Outputs:           Frr:  numpy array     Array of forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the second input is a scalar or a vector
    if (type(terrain_angle) != int) and (type(terrain_angle) != float) and (not isinstance(terrain_angle, np.ndarray)):
        raise Exception('Second input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(terrain_angle, np.ndarray):
        terrain_angle = np.array([terrain_angle],dtype=float) # make the scalar a numpy array
    elif len(np.shape(terrain_angle)) != 1:
        raise Exception('Second input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the first two inputs are of the same size
    if len(omega) != len(terrain_angle):
        raise Exception('First two inputs must be the same size')
    
    # Check that values of the second input are within the feasible range  
    if max([abs(x) for x in terrain_angle]) > 75:    
        raise Exception('All elements of the second input must be between -75 degrees and +75 degrees')
        
    # Check that the third input is a dict
    if type(rover) != dict:
        raise Exception('Third input must be a dict')
        
    # Check that the fourth input is a dict
    if type(planet) != dict:
        raise Exception('Fourth input must be a dict')
        
    # Check that the fifth input is a scalar and positive
    if (type(Crr) != int) and (type(Crr) != float):
        raise Exception('Fifth input must be a scalar')
    if Crr <= 0:
        raise Exception('Fifth input must be a positive number')
        
    # Main Code
    m = get_mass(rover)
    g = planet['g']
    r = rover['wheel_assembly']['wheel']['radius']
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    
    v_rover = r*omega/Ng
    
    Fn = np.array([m*g*math.cos(math.radians(x)) for x in terrain_angle],dtype=float) # normal force
    Frr_simple = -Crr*Fn # simple rolling resistance
    
    Frr = np.array([math.erf(40*v_rover[ii]) * Frr_simple[ii] for ii in range(len(v_rover))], dtype = float)
    
    return Frr


def F_gravity(terrain_angle, rover, planet):
    """
    Inputs:  terrain_angle:  numpy array   Array of terrain angles [deg]
                     rover:  dict          Data structure specifying rover 
                                            parameters
                    planet:  dict          Data dictionary specifying planetary 
                                            parameters
    
    Outputs:           Fgt:  numpy array   Array of forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(terrain_angle) != int) and (type(terrain_angle) != float) and (not isinstance(terrain_angle, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(terrain_angle, np.ndarray):
        terrain_angle = np.array([terrain_angle],dtype=float) # make the scalar a numpy array
    elif len(np.shape(terrain_angle)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that values of the first input are within the feasible range  
    if max([abs(x) for x in terrain_angle]) > 75:    
        raise Exception('All elements of the first input must be between -75 degrees and +75 degrees')

    # Check that the second input is a dict
    if type(rover) != dict:
        raise Exception('Second input must be a dict')
    
    # Check that the third input is a dict
    if type(planet) != dict:
        raise Exception('Third input must be a dict')
        
    # Main Code
    m = get_mass(rover)
    g = planet['g']
    
    Fgt = np.array([-m*g*math.sin(math.radians(x)) for x in terrain_angle], dtype = float)
        
    return Fgt


def F_drive(omega, rover):
    """
    Inputs:  omega:  numpy array   Array of motor shaft speeds [rad/s]
             rover:  dict          Data dictionary specifying rover parameters
    
    Outputs:    Fd:  numpy array   Array of drive forces [N]
    """
    
    # Check that 2 inputs have been given.
    #   IS THIS NECESSARY ANYMORE????
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')

    # Check that the second input is a dict
    if type(rover) != dict:
        raise Exception('Second input must be a dict')
    
    # Main code
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    
    tau = tau_dcmotor(omega, rover['wheel_assembly']['motor'])
    tau_out = tau*Ng
    
    r = rover['wheel_assembly']['wheel']['radius']
    
    # Drive force for one wheel
    Fd_wheel = tau_out/r 
    
    # Drive force for all six wheels
    Fd = 6*Fd_wheel
    
    return Fd


def F_net(omega, terrain_angle, rover, planet, Crr):
    """
    Inputs:           omega:  list     Motor shaft speed [rad/s]
              terrain_angle:  list     Array of terrain angles [deg]
                      rover:  dict     Data structure specifying rover 
                                      parameters
                     planet:  dict     Data dictionary specifying planetary 
                                      parameters
                        Crr:  scalar   Value of rolling resistance coefficient
                                      [-]
    
    Outputs:           Fnet:  list     Array of forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')
        
     #Check that the second input is a scalar or a vector
    if (type(terrain_angle) != int) and (type(terrain_angle) != float) and (not isinstance(terrain_angle, np.ndarray)):
        raise Exception('Second input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(terrain_angle, np.ndarray):
        terrain_angle = np.array([terrain_angle],dtype=float) # make the scalar a numpy array
    elif len(np.shape(terrain_angle)) != 1:
        raise Exception('Second input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the first two inputs are of the same size
    if len(omega) != len(terrain_angle):
        raise Exception('First two inputs must be the same size')
    
    # Check that values of the second input are within the feasible range  
    if max([abs(x) for x in terrain_angle]) > 75:    
        raise Exception('All elements of the second input must be between -75 degrees and +75 degrees')
        
    # Check that the third input is a dict
    if type(rover) != dict:
        raise Exception('Third input must be a dict')
        
    # Check that the fourth input is a dict
    if type(planet) != dict:
        raise Exception('Fourth input must be a dict')
        
    # Check that the fifth input is a scalar and positive
    if (type(Crr) != int) and (type(Crr) != float):
        raise Exception('Fifth input must be a scalar')
    if Crr <= 0:
        raise Exception('Fifth input must be a positive number')
    
    # Main Code
    Fd = F_drive(omega, rover)
    Frr = F_rolling(omega, terrain_angle, rover, planet, Crr)
    Fg = F_gravity(terrain_angle, rover, planet)
    
    Fnet = Fd + Frr + Fg # signs are handled in individual functions
    
    return Fnet


def motorW(v,rover):
    
    #check that the first input is a scalar or a numpy array
    if (type(v) != int) and (type(v) != float) and (not isinstance(v, np.ndarray)):
        raise Exception('The first input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
        
    #check if the second input is a dict    
    if type(rover) != dict:
        raise Exception('Second input must be a dict')
        
    #retrieve gear ratio from get_gear_ratio function    
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    #get wheel radius from rover dict
    r = rover['wheel_assembly']['wheel']['radius']
    #calculate the rotational velocity of the wheel using the translational velocity and the wheel's radius
    w_out = v/r
    #calculate the rotational velocity of the motor shaft using w_out and the gear ratio
    w = w_out*Ng    
        
    return w


def rover_dynamics(t,y,rover,planet,experiment):
    
    #check if the first input is a scalar 
    if (type(t) != int) and (type(t) != float) and (type(t)!=np.float64):
        raise Exception('The first input must be a scalar.')
        
    #check if the second input is a numpy array    
    if not isinstance(y, np.ndarray):
        raise Exception('The second input must be a numpy array.')
    
    #check if the third input is a dict
    if type(rover) != dict:
        raise Exception('The third input must be a dict.')
        
    #check if the fourth input is a dict    
    if type(planet) != dict:
        raise Exception('The fourth input must be a dict.')
    
    #check if the fifth input is a dict
    if type(experiment) != dict:
        raise Exception('The fifth input must be a dict.')
       
    #retrieve the array of degree values from the experiment dict
    alpha_deg = experiment['alpha_deg']
    #get the array of distance values from the experiment dict
    alpha_dist = experiment['alpha_dist']
    #define a function alpha_fun that interpolates the data from alpha_deg and alpha_dist
    alpha_fun = inte.interp1d(alpha_dist,alpha_deg,kind='cubic',fill_value='extrapolate')
    #retrieve the coefficient of rolling resistance from the experiment dict
    Crr = experiment['Crr']
    #approximate the terrain angle using the interpolation function and a value of position
    terrain_angle = alpha_fun(y[1])
    #calculate the corresponding omega value using the motorW function and a value of velocity
    w = motorW(float(y[0]),rover) 
    #calculate the net force on the rover using the F_net function with the omega and terrain angle values
    F = F_net(w,float(terrain_angle),rover,planet,Crr)
    #retrieve the mass of the rover
    mass = get_mass(rover)
        
    dydt=np.array([float(F/mass),y[0]])
    return dydt

def mechpower(v,rover):
    
    #check if the first input is a scalar or a numpy array
    if (type(v) != int) and (type(v) != float) and (not isinstance(v, np.ndarray)):
        raise Exception('The first input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    
    #check if the second input is a dict
    if type(rover) != dict:
        raise Exception('Second input must be a dict')
    
    #calculate the value(s) of motor shaft speed using the motorW function
    w = motorW(v,rover)
    #calculate the value(s) of tau using the tau_dcmotor function
    tau = tau_dcmotor(w,rover['wheel_assembly']['motor'])
    
    #calculate power using the corresponding tau and omega values
    P = tau*w    
    
    return P
    
    
def battenergy(t,v,rover):
    
    #check if the first input is a numpy array
    if not isinstance(t, np.ndarray):
        raise Exception('The first input should be defined as a numpy array.')
    
    #check if the second input is a numpy array
    if not isinstance(v, np.ndarray):
        raise Exception('The second input should be defined as a numpy array.')    
    
    #test if the length of the first and second input are the same
    if len(t) != len(v):
        raise Exception('The first and second input must have the same length.')
    
    #check if the third input is a dict
    if type(rover) != dict:
        raise Exception('Third input must be a dict')
    
    #calculate the mechanical power using the mechpower function and the second input
    P = mechpower(v,rover)
    #calculate the motor shaft speed using the motorW function
    w = motorW(v,rover)
    #calculate the torque using the tau_dcmotor function
    tau = tau_dcmotor(w,rover['wheel_assembly']['motor'])   
    #get the array of torque values from the rover dict
    effcy_tau = rover['wheel_assembly']['motor']['effcy_tau']
    #get the array of efficiency values from the rover dict
    effcy = rover['wheel_assembly']['motor']['effcy']
    
    #define a function effcy_fun that interpolates the values of effcy_tau and effcy
    effcy_fun = inte.interp1d(effcy_tau,effcy,kind='cubic')
    #interpolate a value for efficiency using the interpolation function and a value for tau
    eff = effcy_fun(tau)
    
    #intialize an array of zeroes
    P_batt = np.zeros(len(eff))
    for i in range(len(eff)):
        #if the efficiency value is 0, the resulting P_batt value is NaN
        if eff[i] == 0:
            P_batt[i] = 0
        else:
            #calculate the value of P_batt using the mechanical power of 6 wheels and the motor efficiency
            P_batt = 6*P/eff
    #integrate the array of power values to get the value of electrical energy
    E = simps(P_batt,t)
    
    return E


def simulate_rover(rover,planet,experiment,end_event):
    
    #check if the first input is a dict
    if type(rover) != dict:
        raise Exception('The first input must be a dict.')
    
    #check if the second input is a dict
    if type(planet) != dict:
        raise Exception('The second input must be a dict.')
    
    #check if the third input is a dict
    if type(experiment) != dict:
        raise Exception('The third input must be a dict.')
      
    #check if the fourth input is a dict
    if type(end_event) != dict:
        raise Exception('The fourth input must be a dict.')
    
    #get the array of distance values from the experiment dict
    alpha_dist = experiment['alpha_dist']
    #get the array of degree values from the experiment dict
    alpha_deg = experiment['alpha_deg']
    #get the initial conditions from the experiment dict
    y0 = experiment['initial_conditions']
    #get the time span from the experiment dict
    tspan = experiment['time_range']
    #call the end_of_mission_event function and get the events that will terminate the ode solver
    events = end_of_mission_event(end_event)
    
    #define the rover_dynamics function as a function of t and y
    fun=lambda t,y: rover_dynamics(float(t),y,rover,planet,experiment)
    #use the rk45 ode solver method to solve the rover dynamics function for time, position, and velocity
    sol=solve_ivp(fun,tspan,y0,method='RK45',events=events)
    
    #store the values of time in T
    T=sol.t
    #store the values of velocity in Y0
    Y0=sol.y[0,:]
    #store the values of position in Y1
    Y1=sol.y[1,:]
    
    #update the rover dictionary with new values about the rover's perfomance over the given experiment conditions
    rover['telemetry']={
            'Time': T,
            'completion_time': T[-1],
            'velocity': Y0,
            'position': Y1,
            'distance_traveled': Y1[-1],
            'max_velocity': max(Y0),
            'average_velocity': sum(Y0)/len(Y0),
            'power': mechpower(Y0,rover),
            'batteryenergy': battenergy(T,Y0,rover),
            'energy_per_distance': battenergy(T,Y0,rover)/Y1[-1]}
    
    return rover

def end_of_mission_event(end_event):
    """
    Defines an event that terminates the mission simulation. Mission is over
    when rover reaches a certain distance, has moved for a maximum simulation 
    time or has reached a minimum velocity.            
    """
    
    mission_distance = end_event['max_distance']
    mission_max_time = end_event['max_time']
    mission_min_velocity = end_event['min_velocity']
    
    # Assume that y[1] is the distance traveled
    distance_left = lambda t,y: mission_distance - y[1]
    distance_left.terminal = True
    
    time_left = lambda t,y: mission_max_time - t
    time_left.terminal = True
    
    velocity_threshold = lambda t,y: y[0] - mission_min_velocity;
    velocity_threshold.terminal = True
    velocity_threshold.direction = -1
    
    # terminal indicates whether any of the conditions can lead to the
    # termination of the ODE solver. In this case all conditions can terminate
    # the simulation independently.
    
    # direction indicates whether the direction along which the different
    # conditions is reached matter or does not matter. In this case, only
    # the direction in which the velocity treshold is arrived at matters
    # (negative)
    
    events = [distance_left, time_left, velocity_threshold]
    
    return events
       
