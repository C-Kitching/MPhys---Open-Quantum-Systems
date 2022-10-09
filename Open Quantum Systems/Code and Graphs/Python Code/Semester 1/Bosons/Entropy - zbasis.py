import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import expm, sinm, cosm, eig
import sys


# propogate forward in time
def propagate_forward(t, dt, state, down_jumps, up_jumps):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*H_eff*dt).dot(state) # phi(t+dt)
    dp = 1 - inner(state_1) # prob to jump

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
    

def single_MC(initial_state, t):
    
    record_states = []  # blank array to record states
    record_time = []  # blank array to record time
    
    # record initial values
    record_states.append(initial_state)
    record_time.append(t)
    
    state = initial_state
    
    # initialise jump counts to 0
    up_jumps= 0
    down_jumps = 0
    
    # record jump counts at each step
    record_up_jumps = []
    record_down_jumps = []
    
    # append initial jump counts
    record_up_jumps.append(up_jumps)
    record_down_jumps.append(down_jumps)
    
    # propgate forward in time
    for i in range(int(tmax/dt)):
      
        # get new state and time
        (state, t, down_jumps, up_jumps) = propagate_forward(t, dt, state, down_jumps, up_jumps)
        
        # record jumps
        record_up_jumps.append(up_jumps)
        record_down_jumps.append(down_jumps)
                
        # normalise the states
        normalised_state = (1/np.linalg.norm(state))*state
    
        # record states
        record_states.append(normalised_state)
        record_time.append(t)
        
        # set the state to the normalised state
        state = normalised_state
    
    reduced_density_op_matrices = []  # blank array to store density ops at each step
    
    # calculate the density ops at each step
    for i in range(len(record_states)):
        reduced_density_op_matrices.append(outer(record_states[i], record_states[i]))
            
    # calculate von neumann entropy
    S = []  
    for i in range(2, len(reduced_density_op_matrices)):
        s1 = -trace(reduced_density_op_matrices[i].dot(np.log(reduced_density_op_matrices[i])))
        s2= -trace(reduced_density_op_matrices[1].dot(np.log(reduced_density_op_matrices[1])))
        S.append((s2-s1)/record_time[i])   
                     
    # calculate entropy flux
    J = []
    for i in range(1, len(record_down_jumps)):
        J.append((little_omega/T)*(record_down_jumps[i] - record_up_jumps[i])/record_time[i]-record_time[0])
        
    # calculate total entropy flow
    sigma = []
    for i in range(len(S)):
        sigma.append(np.real(S[i] + J[i]))
        
    # remove the first 2 time elements
    record_time = np.delete(record_time, 0)
    record_time = np.delete(record_time, 1)
    
    return(sigma, record_time)
    

j = 1j  # imaginary unit

delta = 1  
epsilon = 0  # Detuning
gamma = delta / 6
little_omega = 1
eta = np.sqrt(epsilon**2+delta**2)
k = 8.617e-5 # boltzman constant eV/K
T = 8000
alpha = 1/(12*np.pi)

# theta
if(epsilon != 0):
    theta = np.arctan(delta/epsilon)
else:
    theta = np.pi/2

cos= epsilon/eta
sin = epsilon/eta

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

dt = 0.01  # stepsize
t = 0  # end time 
tmax = 20  # start time

# Hamiltonian
H_sys = (epsilon/2)*sigma_z + (delta/2)*sigma_x
H_eff = H_sys - j*gamma*N*sigma_minus.dot(sigma_plus) - j*gamma*(1 + N)*sigma_plus.dot(sigma_minus)

initial_state = g # set the initial states initial state

# Do the MC for a single tragetory
entropy_flow, time = single_MC(initial_state, t)

# graph details
plt.figure()
plt.plot(time, entropy_flow)
plt.title('Entropy Flow, dt = %s' %dt)
plt.xlabel('$t$')
plt.ylabel('$\sigma$')
plt.show()












