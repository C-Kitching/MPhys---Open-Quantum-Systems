import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, sinm, cosm


# propogate forward in time
def propagate_forward(t, dt, state):
    
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
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus/np.sqrt(dp_plus/dt)
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
    
    population = []  # blank array to store population numbers
    
    # extract the population
    for i in range(len(reduced_density_op_matrices)):
        population.append((reduced_density_op_matrices[i])[0][0])
    
    population = np.real(population)
    
    return(population, record_time)


j = 1j  # imaginary unit

delta = 1  
epsilon = 0  # Detuning
gamma = delta / 6
little_omega = 1.602e-19
eta = np.sqrt(epsilon**2+delta**2)
k = 1.381e-23
T = 25
alpha = 1/(12*np.pi)

if(epsilon != 0):
    theta = np.arctan(delta/epsilon)
else:
    theta = np.pi/2

cos= epsilon/eta
sin = epsilon/eta

# states in z basis
old_e = np.array([[1],[0]])
old_g = np.array([[0],[1]])

#states in +/- basis
plus = np.sin(theta/2)*old_g+np.cos(theta/2)*old_e
minus = np.cos(theta/2)*old_g-np.sin(theta/2)*old_e

# e and g in terms of +/- basis
g = np.sin(theta/2)*plus + np.cos(theta/2)*minus
e = np.cos(theta/2)*plus - np.sin(theta/2)*minus

# bosonic occupation number
N = 1/(np.exp(little_omega/(k*T)) - 1)

# Pauli matricies
sigma_plus = outer(e, g)
sigma_minus = outer(g, e)
sigma_x = outer(e, g) + outer(g, e)
sigma_y = np.array([[0, -j], [-j, 0]])
sigma_z = outer(e, e) - outer(g, g)
I = np.array([[1,0], [0,1]])

dt = 0.01  # stepsize
t = 0  # end time 
tmax = 200  # start time

# Hamiltonian
H_sys = (epsilon/2)*sigma_z + (delta/2)*sigma_x
H_eff = H_sys - j*gamma*N*sigma_minus.dot(sigma_plus) - j*gamma*(1 + N)*sigma_plus.dot(sigma_minus)

initial_state = g # set the initial state

# Do the MC for a single tragetory
plt.figure()
(pop, time) = single_MC(initial_state, t)
(pop2, time2) = single_MC(initial_state, t)

plt.plot(time2, pop2, 'r--')
plt.plot(time, pop)
plt.title('Single MC Quantum Trajectories - Temp Dependence \n Energy Eigenbasis, dt = %s' %dt)
plt.xlabel('t')
plt.ylabel('$P_{e}$')
plt.show()


