import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.linalg import expm, sinm, cosm
from math import log10, floor

# propogate forward in time
def propagate_forward(t, dt, state):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective_hamiltonian(t)*dt).dot(state) # phi(t+dt)
    dp = 1 - inner(state_1) # prob to jump
    
    # no jump
    if(random_one >= dp):

        no_jump_prefactor = (I-j*dt*effective_hamiltonian(t))/(np.sqrt(1-dp))
        new_state = no_jump_prefactor.dot(state)
        
    # jump
    elif(random_one < dp):
        
        random_two = np.random.uniform(0, 1)
        
        dp_plus = (state.conj().T).dot(sigma_plus(t).dot(sigma_minus(t).dot(state)))
        dp_minus = (state.conj().T).dot(sigma_minus(t).dot(sigma_plus(t).dot(state)))
        
        dp_plus_normalised = dp_plus / (dp_plus + dp_minus)
        dp_minus_normalised = dp_minus / (dp_plus + dp_minus)
        
        # jump up
        if(random_two >= dp_plus_normalised):
            jump_up_prefactor = sigma_plus(t)/np.sqrt(dp_minus/dt)
            new_state = jump_up_prefactor.dot(state)
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus(t)/np.sqrt(dp_plus/dt)
            new_state = jump_down_prefactor.dot(state)

    return(new_state, t+dt)

# calculate outer product
def outer(state1, state2):
  return(state1.dot(state2.conj().T))

# calculate inner product
def inner(state):
  return((state.conj().T).dot(state))

def single_MC(initial_state, t):
    
    record_states = []  # blank array to record states
    record_time = []  # blank array to record time
    
    # record initial values
    record_states.append(initial_state)
    record_time.append(t)
    
    state = initial_state
    
    # propgate forward in time
    for i in range(int(tmax/dt)):
                        
        # get new state and time
        (state, t) = propagate_forward(t, dt, state)
        
        # normalise the states
        normalised_state = (1/np.sqrt(inner(state)))*state
    
        # record states
        record_states.append(normalised_state)
        record_time.append(t)
        
        # set the state to the normalised state
        state = normalised_state
    
    reduced_density_op_matrices = []  # blank array to store density ops at each step
    
    # calculate the density ops at each step
    for i in range(len(record_states)):
        reduced_density_op_matrices.append(outer(record_states[i], record_states[i]))
    
    population_plus = []  # blank array to store population numbers
    population_minus = []
    
    # extract the population
    for i in range(len(reduced_density_op_matrices)):
        population_minus.append(((minus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(minus(record_time[i]))[0][0])
        population_plus.append(((plus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(plus(record_time[i]))[0][0])
        
    population_plus = np.real(population_plus)
    population_minus = np.real(population_minus)
    
    return population_minus, population_plus, record_time


# calculate effective hamitonian
def effective_hamiltonian(time):
    
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))
    
    # Effective Hamiltonian
    #H_eff = H_sys - j*(gamma_0/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time)/2)*(1+N)*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time)/2)*N*P_eta(time).dot(P_eta(time).conj().T)
    H_eff = H_sys

    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):
       
    return 0.01*detuning(tmax)
    
# calculate time dependent bias
def detuning(time):  
    
    return 20*(time/tmax-(1./2.))

# calculate time depedent P_0 operator
def P_0(time):
    return np.cos(theta(time))*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))

# calculate time depedent P_eta operator
def P_eta(time):
    return np.sin(theta(time))*(outer(minus(time), plus(time)))
    
#calculate time depedent minus state
def minus(time):
    return np.cos(theta(time)/2)*g-np.sin(theta(time)/2)*e
    
# calculate time depedent plus state
def plus(time):
    return np.sin(theta(time)/2)*g+np.cos(theta(time)/2)*e
    
# calculate time dependent theta
def theta(time):
    if(detuning(time) == 0):
        return np.pi/2
    else:
        return np.arctan2(tunnelling_coeff(time),detuning(time))

# calculate eta
def eta(time):
    return(np.sqrt(tunnelling_coeff(time)**2+detuning(time)**2))

# calculate time dep rate
def gamma_eta(time):
    return 2*np.pi*alpha*eta(time)

# calculate time dep sigma+
def sigma_plus(time):
    return outer(plus(time),minus(time))

# calculate time dep sigma-
def sigma_minus(time):
    return outer(minus(time),plus(time))

# calculate time dep sigma_x
def sigma_x(time):
    return outer(plus(time), minus(time)) + outer(minus(time), plus(time))

# calculate time dep sigma_z
def sigma_z(time):
    return outer(plus(time), plus(time)) - outer(minus(time), minus(time))


#### BEGIN MAIN PROGRAM

# Define some constants
j = 1j  # imaginary unit
little_omega = 1
k = 8.617e-5 # boltzman constant
T = 5e3 # temperature 
alpha = 1/(12*np.pi) # coupling strength

# Time indep rate
gamma_0= 4*np.pi*alpha*k*T

# states in z basis
e = np.array([[1],[0]])
g = np.array([[0],[1]])

#states in +/- basis time indep
#plus = np.sin(theta/2)*g+np.cos(theta/2)*e
#minus = np.cos(theta/2)*g-np.sin(theta/2)*e

# bosonic occupation number
N = 1/(np.exp(little_omega/(k*T)) - 1)

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
I = np.array([[1,0], [0,1]])

# time data
tmax_LZ = np.logspace(-15,20, 100)

# master arry to store end probability
master_pop = []
master_std = []

# monte carlo for each time
for i in range(len(tmax_LZ)):

    t = 0  # start time
    tmax = tmax_LZ[i]  # end time
    dt = tmax/200.  # stepsize
    
    number_of_traj = 1
    
    initial_state = minus(t)
        
    multiple_traj_excited_pop = []  
    multiple_traj_ground_pop = []
    
    # loop over all trajectories
    for i in range(number_of_traj):
        
        # do the monte carlo for each traj
        temp_excited, temp_ground, temp_time = single_MC(initial_state, t)
        
        # append to mastere arrays
        multiple_traj_excited_pop.append(temp_excited)
        multiple_traj_ground_pop.append(temp_ground)
        
    # calculate average pop
    avg_excited_pop = np.mean(multiple_traj_excited_pop, axis = 0)
    avg_ground_pop = np.mean(multiple_traj_ground_pop, axis = 0)
    
    # calculate error of average dist
    std_excited = np.std(multiple_traj_excited_pop, axis = 0) / np.sqrt(number_of_traj)
    std_ground = np.std(multiple_traj_ground_pop, axis = 0) / np.sqrt(number_of_traj)

    # store the end result
    master_pop.append(avg_excited_pop[-1])
    master_std.append(std_excited[-1])


# Landua Zener

# lz calculations
bias_rate_of_change = [20./tmax_LZ[i] for i in range(len(tmax_LZ))]
LZ_gamma = [tunnelling_coeff(tmax_LZ[i])**2/abs(bias_rate_of_change[i]) for i in range(len(tmax_LZ))]
LZ_prob = [1-np.exp(-np.pi*LZ_gamma[i]/2) for i in range(len(tmax_LZ))]


# plot results
plt.figure()
plt.plot(np.log10(bias_rate_of_change), master_pop, label = 'Numerical approx \nto ground state \npopulation')
plt.plot(np.log10(bias_rate_of_change), LZ_prob, label = 'Landau-Zener \nprobability to \ntransition')
plt.xlabel('$log\\left(\\frac{\partial\epsilon}{\partial t}\\right)$')
plt.ylabel('Probability')
plt.legend()
plt.title('Landau-Zener vs Numerical Approximation \nin energy-basis starting in ground state')
























