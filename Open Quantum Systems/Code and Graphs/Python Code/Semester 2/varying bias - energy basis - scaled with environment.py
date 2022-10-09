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

def single_MC(core_traj):
    
    master_plus = []
    master_minus = []
    
    for i in range(core_traj):
        
        # reset time and initial state 
        t = 0  # start time    
        initial_state = minus(0) # set the initial states initial state
    
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
        
        master_plus.append(population_plus)
        master_minus.append(population_minus)
        
    # average over trajectories
    avg_pop_plus = np.mean(master_plus, axis = 0)
    avg_pop_minus = np.mean(master_minus, axis = 0)
    
    return avg_pop_plus, avg_pop_minus, record_time


# calculate effective hamitonian
def effective_hamiltonian(time):
    
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))
    
    # Effective Hamiltonian
    H_eff = H_sys - j*(gamma_0/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time)/2)*(1+N(time))*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time)/2)*N(time)*P_eta(time).dot(P_eta(time).conj().T)

    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):
    return 0.1*detuning(tmax)
    
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
        if(tunnelling_coeff((time)) < 0):
            return np.arctan2(tunnelling_coeff(time)+ np.pi,detuning(time))
        else:
            return np.arctan2(tunnelling_coeff(time),detuning(time))

# calculate eta
def eta(time):
    return(np.sqrt((tunnelling_coeff(time))**2+detuning(time)**2))

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


#### BEGIN MAIN PROGRAM

# Define some constants
j = 1j  # imaginary unit
k = 1 # boltzman constant
T = 1 # temperature 
alpha = 0.1/(12*np.pi) # coupling strength

# Time indep rate
gamma_0= 4*np.pi*alpha*k*T

# states in z basis
e = np.array([[1],[0]])
g = np.array([[0],[1]])

#states in +/- basis time indep
#plus = np.sin(theta/2)*g+np.cos(theta/2)*e
#minus = np.cos(theta/2)*g-np.sin(theta/2)*e

# bosonic occupation number
#N = 1/(np.exp(little_omega/(k*T)) - 1)

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
I = np.array([[1,0], [0,1]])

tmin = 0
t = 0  # end time 
tmax = 2000 # start 
dt = 0.1  # stepsize


# being main program
if __name__ == '__main__':
    
    start = timeit.default_timer() # start timer
    
    # check how many cores we have
    core_number = multiprocessing.cpu_count()
    print('You have {0:1d} CPU cores'.format(core_number))

    # traj number
    number_of_traj = 20
    
    core_traj = [int(number_of_traj/core_number) for i in range(core_number)]
    
    #Create the worker pool
    pool = Pool(processes=core_number) 

    # parallel map
    results = pool.map(single_MC, core_traj)
    
    # close the pool
    pool.close()
    
    plus_data = [results[i][0] for i in range(len(results))]  
    minus_data = [results[i][1] for i in range(len(results))] 
    time = results[0][2]
    
    # normalise the time
    time = [time[i]/tmax for i in range(len(time))]
    
    # average each cores results
    minus_avg = np.mean(minus_data, axis = 0)
    plus_avg = np.mean(plus_data, axis = 0)

    # calculate error of average dist
    std_plus = np.std(plus_data, axis = 0) / np.sqrt(number_of_traj)
    std_minus = np.std(minus_data, axis = 0) / np.sqrt(number_of_traj)

    # Graph details
    plt.figure()
    # plot errorbar plot
    plt.errorbar(time, plus_avg, yerr = std_plus, capsize = 2.5, errorevery = len(time)//20, ecolor = 'm', label = '$\\rho_{++}$')
    plt.errorbar(time, minus_avg, yerr = std_minus, capsize = 2.5, errorevery = len(time)//20, ecolor = 'm', label = '$\\rho_{--}$')
    plt.title('Adibatic limit for %d trajectories with environment' %number_of_traj)
    plt.ylabel('$\overline{P}$')
    plt.xlabel('$\\frac{t}{t_{max}}$')   
    plt.legend()













