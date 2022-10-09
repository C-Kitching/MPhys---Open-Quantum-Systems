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
    
    state_1 = (I - j*H_eff*dt).dot(state) # phi(t+dt)
    dp = 1 - inner(state_1) # prob to jump
    
    print(inner(state_1))

    # no jump
    if(random_one >= dp):
        no_jump_prefactor = (I-j*dt*H_eff)/(np.sqrt(1-dp))
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
        
        # append initial jump counts
        record_up_jumps.append(old_up_jumps)
        record_down_jumps.append(old_down_jumps)
        
        # record the points the jumps occur
        jump_points = []
        jump_points.append(0)
        
        # propgate forward in time
        for i in range(int(tmax/dt)):
          
            # get new state and time
            (state, t, new_down_jumps, new_up_jumps) = propagate_forward(t, dt, state, old_down_jumps, old_up_jumps)
            
            # record jumps
            record_up_jumps.append(new_up_jumps)
            record_down_jumps.append(new_down_jumps)
            
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
            work.append(np.real(((record_states[i].conj().T).dot((H_sys).dot(record_states[i])))[0][0]))
                        
        master_work.append(work)
        master_up_jumps.append(record_up_jumps)
        master_down_jumps.append(record_down_jumps)
        
    avg_work = np.mean(master_work, axis = 0)
    avg_up_jumps = np.mean(master_up_jumps, axis = 0)
    avg_down_jumps = np.mean(master_down_jumps, axis = 0)
        
    return(avg_work, record_time, avg_up_jumps, avg_down_jumps)

    

j = 1j  # imaginary unit


delta = 1 # tunnelling 
epsilon = 0  # bias
gamma = delta / 6 # rates
little_omega = 1 # oprtical transition
eta = np.sqrt(epsilon**2+delta**2)
k = 1 # boltzman constant 
T = 8.617e-5 * 4000 # temperature
alpha = 1/(12*np.pi) # coupling strength

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
tmax = 20  # end time 

# Hamiltonian
H_sys = (epsilon/2)*sigma_z + (delta/2)*sigma_x
H_eff = H_sys - j*gamma*N*sigma_minus.dot(sigma_plus) - j*gamma*(1 + N)*sigma_plus.dot(sigma_minus)
#H_eff = H_sys


results = single_MC(100)


















