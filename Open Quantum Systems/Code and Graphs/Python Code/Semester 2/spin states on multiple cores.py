import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import expm, sinm, cosm, eig, logm
import sys
import multiprocessing
import timeit
from multiprocessing import Pool, cpu_count

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

def single_MC(core_traj):
    
    master_excited = []
    master_ground = []
    
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
        
        excited_population = []  # blank array to store population numbers
        ground_population = []
        
        # extract the population
        for i in range(len(reduced_density_op_matrices)):
            ground_population.append(((g.conj().T).dot(reduced_density_op_matrices[i])).dot(g)[0][0])
            excited_population.append(((e.conj().T).dot(reduced_density_op_matrices[i])).dot(e)[0][0])
            
        master_excited.append(np.real(excited_population))
        master_ground.append(np.real(ground_population))
        
    avg_excited = np.mean(master_excited, axis = 0)
    avg_ground = np.mean(master_ground, axis = 0)
    
    return avg_excited, avg_ground, record_time


# calculate effective hamitonian
def effective_hamiltonian(time):
    
    # System Hamiltonian
    H_sys = (detuning(time)/2)*sigma_z + (tunnelling_coeff(time)/2)*sigma_x
    
    # Effective Hamiltonian
    H_eff = H_sys - j*((gamma(time)/2)*N(time)*sigma_minus.dot(sigma_plus) + (gamma(time)/2)*(N(time)+1)*sigma_plus.dot(sigma_minus))  
    #H_eff = H_sys

    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):
       
    return 0.01*detuning(tmax)
    
# calculate time dependent bias
def detuning(time):  
    
    return 10*(time/tmax-(1./2.))

# calculate time dep rate
def gamma(time):
    return tunnelling_coeff(time)/6

# calculate occupation number
def N(time):
    return 1/(np.exp(detuning(time)/(k*T)) - 1)



#### BEGIN MAIN PROGRAM

# Define some constants
j = 1j  # imaginary unit
k = 1 # boltzman constant eV/K
T = 1 # temperature 
alpha = 1/(12*np.pi) # coupling strength

# states in z basis
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# Pauli matrices
sigma_x = outer(e,g) + outer(g,e)
sigma_z = outer(e,e) - outer(g,g)

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
sigma_plus = outer(e,g)
sigma_minus = outer(g,e)
I = np.array([[1,0], [0,1]])

t = 0  # end time 
tmax = 20000 # start 
dt = 0.1  # stepsize

# being main program
if __name__ == '__main__':
    
    start = timeit.default_timer() # start timer
    
    # check how many cores we have
    core_number = multiprocessing.cpu_count()
    print('You have {0:1d} CPU cores'.format(core_number))

    # traj number
    number_of_traj = 200
    
    core_traj = [int(number_of_traj/core_number) for i in range(core_number)]

    #Create the worker pool
    pool = Pool(processes=core_number) 

    # parallel map
    results = pool.map(single_MC, core_traj)
    
    # close the pool
    pool.close()
    
    # combine results of each cores calculations
    core_up = [results[i][0] for i in range(len(results))]
    core_down = [results[i][1] for i in range(len(results))]
    time = results[0][2]
    
    # average each cores results
    avg_up_pop = np.mean(core_up, axis = 0)
    avg_down_pop = np.mean(core_down, axis = 0)
    
    # calculate error of average dist
    std_up = np.std(avg_up_pop, axis = 0) / np.sqrt(number_of_traj)
    std_down = np.std(avg_down_pop, axis = 0) / np.sqrt(number_of_traj)
    
    # scale the time to be between 0 and 1
    for i in range(len(time)):
        time[i] = time[i]/tmax
    

    plt.figure(figsize = (9,5))

    # Population plots
    plt.subplot(1,2,2)
    plt.errorbar(time, avg_up_pop, yerr = std_up, capsize = 2.5, errorevery = len(time)//20, ecolor = 'm', label = r'$\left|\uparrow \right>$')
    plt.errorbar(time, avg_down_pop, yerr = std_down, capsize = 2.5, errorevery = len(time)//20, ecolor = 'm', label = r'$\left|\downarrow \right>$')
    plt.title('%d trajectories' %number_of_traj)
    plt.ylabel('$\overline{P}$')
    plt.xlabel('$\\frac{t}{t_{max}}$')   
    plt.legend()
    
    
    # single traj
    single = single_MC(1)
    up = single[0]
    down = single[1]
    time = single[2]
    
    # scale the time to be between 0 and 1
    for i in range(len(time)):
        time[i] = time[i]/tmax
    
    plt.subplot(1,2,1)
    plt.plot(time, up)
    plt.plot(time, down)
    plt.title('1 trajectory')
    plt.ylabel('$\overline{P}$')
    plt.xlabel('$\\frac{t}{t_{max}}$')  
    
    # common title
    plt.suptitle('Spin-state populations with environment T = %d' %T)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    
    
    