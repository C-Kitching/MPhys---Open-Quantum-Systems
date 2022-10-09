import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import expm, sinm, cosm, eig, logm
import sys
import multiprocessing
import timeit
from multiprocessing import Pool, cpu_count
from functools import partial
from numba import jit, cuda

# propogate forward in time
def propagate_forward(t, dt, state, jump, temperature, alpha):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective_hamiltonian(t, temperature, alpha)*dt).dot(state) # phi(t+dt)
    
    dp = 1 - inner(state_1) # prob to jump
    
    # no jump
    if(random_one >= dp):
        
        jump = False
        
        no_jump_prefactor = (I-j*dt*effective_hamiltonian(t, temperature, alpha))/(np.sqrt(1-dp))
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
    

def single_MC(core_traj, temperature, alpha):
    
    # create master arrays
    master_time = []
    master_spin_up = []
    master_spin_down = []
        
    # loop over all trajectories
    for i in range(core_traj):
    
        # reset time and initial state 
        t = 0  # start time    
        initial_state = minus(0) # set the initial states initial state
                
        old_state = initial_state
        
        # data containers
        record_states = []
        record_time = []
        
        # append initial values
        record_states.append(initial_state)
        record_time.append(t)
        
        # initialise jump counts to 0
        jump = False
        
        # propgate forward in time
        for i in range(int(tmax/dt)):
          
            # get new state and time
            (new_state, t, jump) = propagate_forward(t, dt, old_state, jump, temperature, alpha)

            # normalise the states
            normalised_state = (1/np.linalg.norm(new_state))*new_state
            
            # record states
            record_states.append(normalised_state)
            record_time.append(t)
            
            # set the state to the normalised state
            old_state = normalised_state
            
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
            
        excited_population = np.real(excited_population)
        ground_population = np.real(ground_population)
                                    
        # add each traj variables to master array
        master_spin_up.append(excited_population)
        master_spin_down.append(ground_population)
              
    # average over trajectories
    avg_spin_up = np.mean(master_spin_up, axis = 0)
    avg_spin_down = np.mean(master_spin_down, axis =0)
        
    return(record_time, avg_spin_up, avg_spin_down)

# calculate expectation of matrix and state
def expectation(matrix, state):
    return np.real((state.conj().T).dot(matrix.dot(state)))[0][0]


# calculate system hamiltonian
def system_hamiltonian(time):   
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))
    
    return H_sys
        
# calculate effective hamitonian
def effective_hamiltonian(time, temperature, alpha):   
    H_eff = system_hamiltonian(time) - j*(gamma_0(temperature, alpha)/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time, alpha)/2)*(1+N(time, temperature))*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time, alpha)/2)*N(time, temperature)*P_eta(time).dot(P_eta(time).conj().T)
    #H_eff = system_hamiltonian(time, tmax)
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
def gamma_eta(time, alpha):
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
def N(time, temperature):
    return 1/(np.exp(eta(time)/(k*temperature)) - 1)

# gamma 0
def gamma_0(temperature, alpha):
    return 4*np.pi*alpha*k*temperature



j = 1j  # imaginary unit
little_omega = 1 # oprtical transition
k = 1 # boltzman constant 


# excited and ground states
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
I = np.array([[1,0], [0,1]])

# time parameters
tmax = 200
dt = 0.1  # stepsize

# traj number
number_of_traj = 100


# being main program
if __name__ == '__main__':
    
    start = timeit.default_timer() # start timer
    
    # no environment
    results = single_MC(1, 0.1 , 0)
    no_environment_spin_up = results[1]
    
    # check how many cores we have
    core_number = multiprocessing.cpu_count()
    print('You have {0:1d} CPU cores'.format(core_number))

    core_traj = [int(number_of_traj/core_number) for i in range(core_number)]

    #Create the worker pool
    pool = Pool(processes=core_number) 
    
    # data containers
    master_up = []
    master_down = []
    
    # time
    time = np.linspace(0, tmax, int(tmax/dt))
    
    # different regimes
    temp = [0.01, 0.05, 0.1]
    
    # alpha 
    a = 1/(12*np.pi)
    
    # loop over different regimes
    for T in temp:
        
        print(T)
                
        MC = partial(single_MC, temperature = T, alpha = a)
        
        # parallel map
        results = pool.map(MC, core_traj)
    
        # combine results of each cores calculations
        time = results[0][0]
        spin_up = [results[i][1] for i in range(len(results))]
        spin_down = [results[i][2] for i in range(len(results))]
        
        # average each cores results
        up_avg = np.mean(spin_up, axis = 0)
        down_avg = np.mean(spin_down, axis = 0)

        # store results on each regime
        master_up.append(up_avg)
        master_down.append(down_avg)
        
    # close the pool
    pool.close()
    
    # normalise time
    time = [time[i]/max(time) for i in range(len(time))]
    
    # extract different temp data
    upT1 = master_up[0]
    upT2 = master_up[1]
    upT3 = master_up[2]
    downT1 = master_down[0]
    downT2 = master_down[1]
    downT3 = master_down[2]
         
    # Population plots
    plt.figure(figsize = (9,5))
    plt.subplot(1,2,1)
    plt.plot(time,upT1, label = r'$T = $%s' %temp[0])
    plt.plot(time,upT2, label = r'$T = $%s' %temp[1])
    plt.plot(time,upT3, label = r'$T = $%s' %temp[2])
    plt.plot(time, no_environment_spin_up, label = 'No environment', linestyle = '--')
    plt.title(r'Vary $T$', fontsize = 14)
    plt.legend()
    plt.ylabel('$\overline{P}$', fontsize = 14)
    plt.xlabel('$\\frac{t}{t_{max}}$', fontsize = 14) 
    
    
    
    
    
    
    #Create the worker pool
    pool = Pool(processes=core_number) 
    
    # data containers
    master_up = []
    master_down = []
    
    # time
    time = np.linspace(0, tmax, int(tmax/dt))
    
    # different regimes
    alphas = [0.1*a, 0.5*a, a]
    
    # temp
    T = 0.1
    
    # loop over different regimes
    for a in alphas:
        
        print(a)
                
        MC = partial(single_MC, temperature = T, alpha = a)
        
        # parallel map
        results = pool.map(MC, core_traj)
    
        # combine results of each cores calculations
        time = results[0][0]
        spin_up = [results[i][1] for i in range(len(results))]
        spin_down = [results[i][2] for i in range(len(results))]
        
        # average each cores results
        up_avg = np.mean(spin_up, axis = 0)
        down_avg = np.mean(spin_down, axis = 0)

        # store results on each regime
        master_up.append(up_avg)
        master_down.append(down_avg)
        
    # close the pool
    pool.close()
    
    # normalise time
    time = [time[i]/max(time) for i in range(len(time))]
    
    # extract different temp data
    upT1 = master_up[0]
    upT2 = master_up[1]
    upT3 = master_up[2]
    downT1 = master_down[0]
    downT2 = master_down[1]
    downT3 = master_down[2]
         
    # Population plots
    plt.subplot(1,2,2)
    plt.plot(time,upT1, label = r'$\frac{0.1}{12\pi}$')
    plt.plot(time,upT2, label = r'$\frac{0.5}{12\pi}$')
    plt.plot(time,upT3, label = r'$\frac{1}{12\pi}$')
    plt.plot(time, no_environment_spin_up, label = 'No environment', linestyle = '--')
    plt.title(r'Vary $\alpha$', fontsize = 14)
    plt.legend()
    plt.ylabel('$\overline{P}$', fontsize = 14)
    plt.xlabel('$\\frac{t}{t_{max}}$', fontsize = 14) 
    
    # common title
    plt.suptitle('Environment parameters on diabatic state populations', fontsize = 16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # output calculation time
    stop = timeit.default_timer()
    print('Time: ', stop - start) 


