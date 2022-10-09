import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import fsolve

def calculate_jump_time(state, random):
    
    func = lambda tau: random - np.float(np.real((state.conj().T).dot(expm(tau*j*(H_eff.conj().T - H_eff)).dot(state))[0]))
    
    tau_initial_guess = 1
    tau_solution = fsolve(func, tau_initial_guess)
    
    tau_solution = round(tau_solution[0], 2)
    
    return(tau_solution)

def propagate(initial_state, t_start, t_end):
    
    # arrays to record values
    record_states = []
    record_times = []
    
    t = t_start # set initial time
    state = initial_state # set initial state
    
    # record initial values
    record_states.append(state)
    record_times.append(t)
    
    # while not at the end time
    while(t < t_end):

        random = np.random.uniform(0, 1) # select a random number     

        current_time = t
        current_state = state
        jump_time = current_time + calculate_jump_time(state, random) # determine the jump time
        
        if(jump_time <= t_end):
            time_range = np.arange(current_time, jump_time + dt, dt)
        else:
            time_range = np.arange(current_time, t_end + dt, dt)
        
        # until the jump, propagte usually
        for time in time_range:
            
            # find the new state            
            prefactor = 1/(state.conj().T).dot(expm(j*(time - current_time)*(H_eff.conj().T - H_eff)).dot(current_state))
            new_state = prefactor*expm(-j*(time - current_time)*H_eff).dot(current_state)
            
            normalised_state = (1/np.linalg.norm(new_state))*new_state # normalise the new state
            
            record_states.append(normalised_state) # record the state
            
            t += dt # incrememnt the time
            record_times.append(t) # record the time
            
        if(t > t_end): break
    
        state = normalised_state
            
        random_two = np.random.uniform(0, 1) # select a second random number
        
        # calculate expectation values
        dp_minus = dt*(state.conj().T).dot(sigma_plus.dot(sigma_minus.dot(state)))
        dp_plus = dt*(state.conj().T).dot(sigma_minus.dot(sigma_plus.dot(state)))
        
        # normalise expectation values
        dp_plus_normalised = dp_plus / (dp_plus + dp_minus)
        dp_minus_normalised = dp_minus / (dp_plus + dp_minus)
        
        # jump up
        if(random_two >= dp_plus_normalised):
            jump_up_prefactor = sigma_plus/np.sqrt(dp_plus/dt)
            new_state = jump_up_prefactor.dot(state)
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus/np.sqrt(dp_minus/dt)
            new_state = jump_down_prefactor.dot(state)
            
        normalised_state = (1/np.linalg.norm(new_state))*new_state # normalise the state   
        state = normalised_state # update the state      
        record_states.append(state) # record the state
            
        record_times.append(t) # record the time
        
    reduced_density_op_matrices = []  # blank array to store density ops at each step
    
    # calculate the density ops at each step
    for i in range(len(record_states)):
        reduced_density_op_matrices.append(outer(record_states[i]))
    
    population = []  # blank array to store population numbers
    
    # extract the population
    for i in range(len(reduced_density_op_matrices)):
        population.append((reduced_density_op_matrices[i])[0][0])
    
    record_time_omega = [dt * omega for dt in record_times]
    population = np.real(population)
    
    return(population, record_time_omega)
            
# calculate outer product
def outer(state):
  return(state.dot(state.conj().T))

# calculate inner product
def inner(state):
  return((state.conj().T).dot(state))           
            
        




j = 1j  # imaginary unit

omega = 1  # Rabi Frequency
epsilon = 0  # Detuning
gamma = omega / 6
little_omega = 1.602e-19
k = 1.381e-23
T = 8000

N = 1/(np.exp(little_omega/(k*T)) - 1)

dt = 0.01  # stepsize
t_start = 0  # end time 
t_end = 20  # start time

# Pauli matricies
sigma_plus = np.array([[0,1],[0,0]])
sigma_minus = np.array([[0,0],[1,0]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0, -j], [-j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
I = np.array([[1,0], [0,1]])

# Hamiltonians
H_eff = (epsilon/2)*sigma_z + (omega/2)*sigma_x - j*(gamma*N*sigma_minus.dot(sigma_plus) + gamma*(N+1)*sigma_plus.dot(sigma_minus)) 

initial_state = np.array([[0],[1]])  # initial state

# Do the MC for a single tragetory
plt.figure()
(pop, time) = propagate(initial_state, t_start, t_end)
(pop2, time2) = propagate(initial_state, t_start, t_end)

plt.plot(time2, pop2, 'r--')
plt.plot(time, pop)
plt.title('Single MC Quantum Trajectories')
plt.xlabel('t$\Omega$')
plt.ylabel('$P_{e}$')
plt.show()