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
def propagate_forward(t, dt, state, down_jumps, up_jumps):
    
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
        
        dp_plus = (state.conj().T).dot(sigma_plus.dot(sigma_minus.dot(state)))
        dp_minus = (state.conj().T).dot(sigma_minus.dot(sigma_plus.dot(state)))
        
        dp_plus_normalised = dp_plus / (dp_plus + dp_minus)
        dp_minus_normalised = dp_minus / (dp_plus + dp_minus)
        
        # jump up
        if(random_two >= dp_plus_normalised):
            jump_up_prefactor = sigma_plus/np.sqrt(dp_minus/dt)
            new_state = jump_up_prefactor.dot(state)
            up_jumps += 1
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus/np.sqrt(dp_plus/dt)
            new_state = jump_down_prefactor.dot(state)
            down_jumps += 1

    return(new_state, t+dt, down_jumps, up_jumps)

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
    master_down_jumps = []
    master_up_jumps = []    
    master_non_cum_down = []
    master_non_cum_up = []
    
    # loop over all trajectories
    for i in range(core_traj):
    
        # reset time and initial state 
        t = 0  # start time    
        initial_state = g # set the initial states initial state
        
        record_states = []  # blank array to record states
        record_time = []  # blank array to record time
        
        # record initial values
        record_states.append(initial_state)
        record_time.append(t)
        
        state = initial_state
        
        # initialise jump counts to 0
        old_up_jumps= 0
        old_down_jumps = 0
        
        # record jump counts at each step
        record_up_jumps = []
        record_down_jumps = []
        record_non_cum_up = []
        record_non_cum_down = []
        
        # append initial jump counts
        record_up_jumps.append(old_up_jumps)
        record_down_jumps.append(old_down_jumps)
        record_non_cum_up.append(old_up_jumps)
        record_non_cum_down.append(old_down_jumps)
        
        # propgate forward in time
        for i in range(int(tmax/dt)):
          
            # get new state and time
            (state, t, new_down_jumps, new_up_jumps) = propagate_forward(t, dt, state, old_down_jumps, old_up_jumps)
            
            # record jumps
            record_up_jumps.append(new_up_jumps)
            record_down_jumps.append(new_down_jumps)
            
            # record non cum jumps
            record_non_cum_up.append(new_up_jumps - old_up_jumps)
            record_non_cum_down.append(new_down_jumps - old_down_jumps)
            
            # update values
            old_up_jumps = new_up_jumps
            old_down_jumps = new_down_jumps
             
            # normalise the states
            normalised_state = (1/np.linalg.norm(state))*state
        
            # record states
            record_states.append(normalised_state)
            record_time.append(t)
            
            # set the state to the normalised state
            state = normalised_state
        
        # array to store work
        work =[]
        
        # calculate work
        for i in range(len(record_states)):
            work.append(np.real(((record_states[i].conj().T).dot((system_hamiltonian(record_time[i])).dot(record_states[i])))[0][0]))
                        
        master_work.append(work)
        master_up_jumps.append(record_up_jumps)
        master_down_jumps.append(record_down_jumps)
        master_non_cum_up.append(record_non_cum_up)
        master_non_cum_down.append(record_non_cum_down)
        
    avg_work = np.mean(master_work, axis = 0)
    avg_up_jumps = np.mean(master_up_jumps, axis = 0)
    avg_down_jumps = np.mean(master_down_jumps, axis = 0)
    avg_up_non_cum = np.mean(master_non_cum_up, axis = 0)
    avg_down_non_cum = np.mean(master_non_cum_down, axis = 0)
        
    return(avg_work, record_time, avg_up_jumps, avg_down_jumps, avg_up_non_cum, avg_down_non_cum)
        
# calculate effective hamitonian
def effective_hamiltonian(time):
    
    # System Hamiltonian
    H_sys = system_hamiltonian(time)
    
    # Effective Hamiltonian
    H_eff = H_sys - j*gamma(time)*N*sigma_minus.dot(sigma_plus) - j*gamma(time)*(N+1)*sigma_plus.dot(sigma_minus)  

    return(H_eff)

# calculate system hamiltonian
def system_hamiltonian(time):
    H_sys = (detuning(time)/2)*sigma_z + (tunnelling_coeff(time)/2)*sigma_x
    return H_sys

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):       
    return 0.01*detuning(tmax)
    
# calculate time dependent bias
def detuning(time):      
    return 20*((time/tmax) + 1)

# calculate time dep rate
def gamma(time):
    return tunnelling_coeff(time)/6




j = 1j  # imaginary unit
little_omega = 1 # oprtical transition
k = 1 # boltzman constant 
T = 8.617e-5 * 4000 # temperature
alpha = 1/(12*np.pi) # coupling strength

# Time indep rate
gamma_0= 4*np.pi*alpha*k*T

# excited and ground states
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# bosonic occupation number
N = 1/(np.exp(little_omega/(k*T)) - 1)

# Pauli matricies
sigma_plus = np.array([[0,1],[0,0]])
sigma_minus = np.array([[0,0],[1,0]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0, -j], [-j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
I = np.array([[1,0], [0,1]])

# time parameters
dt = 0.1  # stepsize
t = 0  # start time 
tmax = 200  # end time 

# being main program
if __name__ == '__main__':
    
    start = timeit.default_timer() # start timer
    
    # check how many cores we have
    core_number = multiprocessing.cpu_count()
    print('You have {0:1d} CPU cores'.format(core_number))

    # traj number
    number_of_traj = 100
    
    core_traj = [int(number_of_traj/core_number) for i in range(core_number)]

    #Create the worker pool
    pool = Pool(processes=core_number) 

    # parallel map
    results = pool.map(single_MC, core_traj)
    
    # close the pool
    pool.close()
    
    # combine results of each cores calculations
    core_work = [results[i][0] for i in range(len(results))]
    time = results[0][1]
    core_up = [results[i][2] for i in range(len(results))]
    core_down = [results[i][3] for i in range(len(results))]    
    core_up_non_cum = [results[i][4] for i in range(len(results))]
    core_down_non_cum = [results[i][5] for i in range(len(results))]

    # average each cores results
    avg_work = np.mean(core_work, axis = 0)
    avg_up_jumps = np.mean(core_up, axis = 0)
    avg_down_jumps = np.mean(core_down, axis = 0)
    avg_non_cum_up = np.mean(core_up_non_cum, axis = 0)
    avg_non_cum_down = np.mean(core_down_non_cum, axis = 0)
    
    
    # calculate jump difference
    jump_diff = [avg_down_jumps[i] - avg_up_jumps[i] for i in range(len(avg_down_jumps))]
    
    stop = timeit.default_timer()

    print('Time: ', stop - start) 
    
    # heat and work in time
    plt.figure()
    plt.plot(time, avg_work, label = 'Work')
    plt.plot(time, jump_diff, label = 'Heat')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Energy / eV')
    plt.title('Heat and Work on %d trajectories\n Time dependent' %number_of_traj)
    
    # heat histogram
    plt.figure()
    plt.title('Heat exchange between system and environment')
    plt.xlabel('Heat transfered/ eV')
    plt.ylabel('Frequency')
    bins=np.histogram(np.hstack((avg_non_cum_up,avg_non_cum_down)), bins=40)[1] #get the bin edges
    plt.hist(avg_non_cum_up, bins, alpha = 0.5, label = "Heat transfered to system", edgecolor='black')
    plt.hist(avg_non_cum_down, bins, alpha = 0.5, label = "Heat transfered to environment", edgecolor='black')
    plt.legend()
   


