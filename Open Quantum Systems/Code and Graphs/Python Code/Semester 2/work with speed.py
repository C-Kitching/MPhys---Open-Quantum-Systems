import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import expm, sinm, cosm, eig, logm
import sys
import multiprocessing
import timeit
from multiprocessing import Pool, cpu_count
from numba import jit, cuda

# propogate forward in time
def propagate_forward(t, dt, state, jump):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective_hamiltonian(t)*dt).dot(state) # phi(t+dt)
    
    dp = 1 - inner(state_1) # prob to jump
    
    # no jump
    if(random_one >= dp):
        
        jump = False
        
        no_jump_prefactor = (I-j*dt*effective_hamiltonian(t))/(np.sqrt(1-dp))
        new_state = no_jump_prefactor.dot(state)
        
    # jump
    elif(random_one < dp):
        
        jump = True
        
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

    return(new_state, t+dt, jump)

# calculate outer product
def outer(state1, state2):
  return(state1.dot(state2.conj().T))

# calculate inner product
def inner(state):
  return((state.conj().T).dot(state))

# calculate trace of a 2d array
def trace(matrix):
    
    trace = 0
    
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
                if(i == j):
                    trace += matrix[i][j]
                    
    return trace
    

def single_MC(core_traj):
    
    # create master arrays
    master_work = []
    master_heat = []
    master_energy = []
    master_pop_plus = []
    master_pop_minus = []
    
    # loop over all trajectories
    for i in range(core_traj):
    
        # reset time and initial state 
        t = 0  # start time    
        initial_state = minus(0) # set the initial states initial state
        
        record_states = []  # blank array to record states
        record_time = []  # blank array to record time
        record_heat= []
        record_work = []
        record_total_energy = []
        
        # record initial values
        record_states.append(initial_state)
        record_time.append(t)
        record_work.append(0)
        record_heat.append(0)
        record_total_energy.append(0)
        
        old_state = initial_state
        
        # initialise jump counts to 0
        jump = False
        
        # propgate forward in time
        for i in range(int(tmax/dt)):
          
            # get new state and time
            (new_state, t, jump) = propagate_forward(t, dt, old_state, jump)

            # normalise the states
            normalised_state = (1/np.linalg.norm(new_state))*new_state
            
            # if jump then heat exchanged
            if jump:
                heat = expectation(system_hamiltonian(t), normalised_state)- expectation(system_hamiltonian(t), old_state)
            # if no jump then work done
            else:
                heat = 0
            
            # calculate work
            work = expectation(system_hamiltonian(t), normalised_state)- expectation(system_hamiltonian(t-dt), normalised_state)
                    
            # record heat and work
            record_heat.append(heat)
            record_work.append(work)
        
            # get the total energy
            record_total_energy.append(expectation(system_hamiltonian(t), normalised_state)- expectation(system_hamiltonian(t-dt), old_state))
        
            # record states
            record_states.append(normalised_state)
            record_time.append(t)
            
            # set the state to the normalised state
            old_state = normalised_state
                        
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
                        
        # add each traj heat and work to master array
        master_work.append(record_work)
        master_heat.append(record_heat)
        master_energy.append(record_total_energy)
        master_pop_plus.append(population_plus)
        master_pop_minus.append(population_minus)
        
    # average over trajectories
    avg_pop_plus = np.mean(master_pop_plus, axis = 0)
    avg_pop_minus = np.mean(master_pop_minus, axis = 0)
    avg_work = np.mean(master_work, axis = 0)
    avg_heat = np.mean(master_heat, axis = 0)
    avg_energy = np.mean(master_energy, axis = 0)
        
    return(avg_work, avg_heat, avg_energy, record_time, avg_pop_plus, avg_pop_minus)

# calculate expectation of matrix and state
def expectation(matrix, state):
    return np.real((state.conj().T).dot(matrix.dot(state)))[0][0]


# calculate system hamiltonian
def system_hamiltonian(time):   
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))
    
    return H_sys
        
# calculate effective hamitonian
def effective_hamiltonian(time):       
    # Effective Hamiltonian
    H_eff = system_hamiltonian(time) - j*(gamma_0/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time)/2)*(1+N(time))*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time)/2)*N(time)*P_eta(time).dot(P_eta(time).conj().T)
    #H_eff = system_hamiltonian(time)

    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):
    return 0.01*detuning(tmax)
    
# calculate time dependent bias
def detuning(time):  
    
    return 10*(time/tmax - 0.5)

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

# calculate occupation number
def N(time):
    return 1/(np.exp(eta(time)/(k*T)) - 1)



j = 1j  # imaginary unit
little_omega = 1 # oprtical transition
k = 1 # boltzman constant 
T = 1 # temperature
alpha = 1/(12*np.pi) # coupling strength

# Time indep rate
gamma_0= 4*np.pi*alpha*k*T

# bosonic occupation number
#N = 1/(np.exp(little_omega/(k*T)) - 1)

# excited and ground states
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
I = np.array([[1,0], [0,1]])


# tmax
tmax = 1
dt = 0.01

# ramp gradient
ramp = []

# store max work
master_work = []

# loop over times 
for i in range(20):

    ramp.append((detuning(tmax)-detuning(0))/tmax)
    
    # parallel map
    results = single_MC(100)
    
    # work
    work = results[0]
        
    # cummulative arrays
    cum_work = np.cumsum(work)
            
    # append to master array
    master_work.append(cum_work[-1])
    
    # incremement the max time
    tmax += 1

plt.figure()
plt.xlabel('$\\frac{\partial\epsilon(t)}{\partial t}$')
plt.ylabel('Work')
plt.title('Work done with ramp speed')
plt.plot(ramp, master_work)
    



