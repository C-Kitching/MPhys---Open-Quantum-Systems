import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# calculate outer product
def outer(state):
  return(np.outer(state, state.conj()))

# calculate inner product
def inner(state):
  return(np.vdot(state, state))

# propogate forward in time
def propagate_forward(state):
    
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

    return(new_state)

def single_MC(initial_state):
    
    record_states = []  # blank array to record states
    
    # record initial values
    record_states.append(initial_state)
    
    state = initial_state
    
    # propgate forward in time
    for i in range(int(t_end/dt) - 1):
      
        # get new state and time
        state = propagate_forward(state)
        
        # normalise the states
        normalised_state = (1/np.linalg.norm(state))*state
    
        # record states
        record_states.append(normalised_state)
        
        # set the state to the normalised state
        state = normalised_state
    
    reduced_density_op_matrices = []  # blank array to store density ops at each step
    
    # calculate the density ops at each step
    for i in range(len(record_states)):
        reduced_density_op_matrices.append(outer(record_states[i]))
    
    population = []  # blank array to store population numbers
    
    # extract the population
    for i in range(len(reduced_density_op_matrices)):
        population.append((reduced_density_op_matrices[i])[0][0])
    population = np.real(population)
    
    return(population)


j = 1j  # imaginary unit

omega = 1  # Rabi Frequency
nu = 0  # Detuning
gamma = omega / 6
little_omega = 1.602e-19
k = 1.381e-23
T = 2500

N = 1/(np.exp(little_omega/(k*T)) - 1)


# Pauli matricies
sigma_plus = np.array([[0,1],[0,0]])
sigma_minus = np.array([[0,0],[1,0]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0, -j], [-j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
I = np.array([[1,0], [0,1]])

# Hamiltonians
H_eff = (nu/2)*sigma_z + (omega/2)*sigma_x - j*((gamma/2)*N*sigma_minus.dot(sigma_plus) + (gamma/2)*(N+1)*sigma_plus.dot(sigma_minus))  

# different trajectories
trajs = np.arange(100,5000,100)

# hold the errors
statistical_errors = []
absolute_errors = []

for i in range(len(trajs)):
  
    dt = 0.01  # stepsize
    t_start = 0  # start time 
    t_end = 20  # end time
    
    
    initial_state = np.array([[0],[1]])  # initial state
    
    number_of_traj = trajs[i]
    
    time = np.arange(t_start, t_end, dt) # time
    time_omega = np.array([omega * t for t in time]) # adjusted time
    
    multiple_traj_pop = []
    
    # get populations for each trajectory
    for i in range(number_of_traj):
        multiple_traj_pop.append(single_MC(initial_state))
        
    # calculate average pop
    avg_pop = np.mean(multiple_traj_pop, axis = 0)
    
    # calculate error of average dist
    std = np.std(multiple_traj_pop, axis = 0) / np.sqrt(number_of_traj)
    
    # EXACT SOLUTION
    
    H = (nu/2) * sigmaz() + (omega/2) * sigmax() # Hamiltonian
    psi0 = basis(2, 1) # initial state [[0],[1]]
    
    times = np.linspace(t_start, t_end, int((t_end - t_start)/dt)) # time
    
    # solve exactly
    result = mesolve(H, psi0, times, [np.sqrt(gamma * N) * sigmap(), np.sqrt(gamma * (N+1)) * sigmam()], [sigmap() * sigmam()])
    
    temp_absolute_error = []
    
    # error calculation
    for i in range(len(avg_pop)):
        temp_absolute_error.append(abs(result.expect[0][i] - avg_pop[i]))
        
    absolute_errors.append(np.mean(temp_absolute_error))
    statistical_errors.append(np.mean(std))
        
        

# graph details
plt.figure()
plt.plot(trajs, absolute_errors, 'b', label = 'Absolute error')
plt.plot(trajs, statistical_errors, 'r--', label = 'Statistical error')
plt.xlabel('Number of Trajectories')
plt.ylabel('Error')
plt.title('Error convergence with trajectory number')
plt.legend()
plt.show()