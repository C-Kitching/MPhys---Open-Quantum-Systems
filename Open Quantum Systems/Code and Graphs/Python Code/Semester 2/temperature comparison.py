import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import expm, sinm, cosm, eig, logm
import sys
import multiprocessing
import timeit
from multiprocessing import Pool, cpu_count
from numba import jit, cuda
from functools import partial

# propogate forward in time
def propagate_forward(t, dt, state, jump, T):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective_hamiltonian(t, T)*dt).dot(state) # phi(t+dt)
    
    dp = 1 - inner(state_1) # prob to jump
    
    # no jump
    if(random_one >= dp):
        
        jump = False
        
        no_jump_prefactor = (I-j*dt*effective_hamiltonian(t, T))/(np.sqrt(1-dp))
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
    

def single_MC(core_traj, T):
    
    # create master arrays
    master_work = []
    master_heat = []
    master_energy = []
    master_pop_plus = []
    master_pop_minus = []
    master_spin_up = []
    master_spin_down = []
    
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
            (new_state, t, jump) = propagate_forward(t, dt, old_state, jump, T)

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
        spin_up = []
        spin_down = []
        
        # extract the population
        for i in range(len(reduced_density_op_matrices)):
            population_minus.append(((minus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(minus(record_time[i]))[0][0])
            population_plus.append(((plus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(plus(record_time[i]))[0][0])
            spin_up.append((e.conj().T).dot(reduced_density_op_matrices[i].dot(e))[0][0])
            spin_down.append((g.conj().T).dot(reduced_density_op_matrices[i].dot(g))[0][0])
            
        population_plus = np.real(population_plus)
        population_minus = np.real(population_minus)
        spin_up = np.real(spin_up)
        spin_down = np.real(spin_down)
                        
        # add each traj heat and work to master array
        master_work.append(record_work)
        master_heat.append(record_heat)
        master_energy.append(record_total_energy)
        master_pop_plus.append(population_plus)
        master_pop_minus.append(population_minus)
        master_spin_up.append(spin_up)
        master_spin_down.append(spin_down)
        
    # average over trajectories
    avg_pop_plus = np.mean(master_pop_plus, axis = 0)
    avg_pop_minus = np.mean(master_pop_minus, axis = 0)
    avg_spin_up = np.mean(master_spin_up, axis = 0)
    avg_spin_down = np.mean(master_spin_down, axis = 0)
    avg_work = np.mean(master_work, axis = 0)
    avg_heat = np.mean(master_heat, axis = 0)
    avg_energy = np.mean(master_energy, axis = 0)
        
    return(avg_work, avg_heat, avg_energy, record_time, avg_pop_plus, avg_pop_minus, avg_spin_up, avg_spin_down)

# calculate expectation of matrix and state
def expectation(matrix, state):
    return np.real((state.conj().T).dot(matrix.dot(state)))[0][0]


# calculate system hamiltonian
def system_hamiltonian(time):   
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))
    
    return H_sys
        
# calculate effective hamitonian
def effective_hamiltonian(time, T):   
    H_eff = system_hamiltonian(time) - j*(gamma_0(T)/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time)/2)*(1+N(time, T))*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time)/2)*N(time, T)*P_eta(time).dot(P_eta(time).conj().T)
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

# calculate occupation number
def N(time, T):
    return 1/(np.exp(eta(time)/(k*T)) - 1)

# time indep rate
def gamma_0(T):
    return 4*np.pi*alpha*k*T



j = 1j  # imaginary unit
little_omega = 1 # oprtical transition
k = 1 # boltzman constant 
alpha = 1/(12*np.pi) # coupling strength

# excited and ground states
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
I = np.array([[1,0], [0,1]])

# time parameters
t = 0  # start time 
tmax = 2000 # end time 
dt = 0.1  # stepsize


# being main program
if __name__ == '__main__':
    
    start = timeit.default_timer() # start timer
    
    # check how many cores we have
    core_number = multiprocessing.cpu_count()
    print('You have {0:1d} CPU cores'.format(core_number))

    # traj number
    number_of_traj = 40
    
    core_traj = [int(number_of_traj/core_number) for i in range(core_number)]

    #Create the worker pool
    pool = Pool(processes=core_number) 
    
    #temeratures
    temps = [0.1, 0.001]
    
    #data
    data = []
    
    for temp in temps:
        
        MC = partial(single_MC, T = temp)
    
        results = pool.map(MC, core_traj)
    
        # combine results of each cores calculations
        core_work = [results[i][0] for i in range(len(results))]
        core_heat = [results[i][1] for i in range(len(results))]  
        core_energy = [results[i][2] for i in range(len(results))] 
        time = results[0][3]
        plus_pop = [results[i][4] for i in range(len(results))]
        minus_pop = [results[i][5] for i in range(len(results))]
        spin_up = [results[i][6] for i in range(len(results))]
        spin_down = [results[i][7] for i in range(len(results))]
        
        # normalise the time
        time = [time[i]/tmax for i in range(len(time))]
        
        # average each cores results
        minus_avg = np.mean(minus_pop, axis = 0)
        plus_avg = np.mean(plus_pop, axis = 0)
        up_avg = np.mean(spin_up, axis = 0)
        down_avg = np.mean(spin_down, axis = 0)
        work = np.mean(core_work, axis = 0)
        heat = np.mean(core_heat, axis = 0)
        energy = np.mean(core_energy, axis = 0)
        
        # cummulative arrays
        cum_work = np.cumsum(work)
        cum_heat = np.cumsum(heat)
        cum_energy = np.cumsum(energy)
    
        # append master array
        data.append([up_avg, down_avg, cum_work, cum_heat, time])
    

    # close the pool
    pool.close()
    
    stop = timeit.default_timer()

    print('Time: ', stop - start) 
    
    #combined figure
    plt.figure(figsize = (9,5))
    
    plt.subplot(1,2,1)
    plt.plot(data[0][4], data[0][2], label = 'Work')
    plt.plot(data[0][4], data[0][3], label = 'Heat')
    plt.xlabel('$\\frac{t}{t_{max}}$', fontsize = 16)
    plt.ylabel('Energy scale', fontsize = 12)
    plt.title('T = 0.1', fontsize = 12)
    
    plt.subplot(1,2,2)
    plt.plot(data[1][4], data[1][2], label = 'Work')
    plt.plot(data[1][4], data[1][3], label = 'Heat')
    plt.xlabel('$\\frac{t}{t_{max}}$', fontsize = 16)
    plt.ylabel('Energy scale', fontsize = 12)
    plt.legend()
    plt.title('T = 0.001', fontsize = 12)
    
    # common title
    plt.suptitle('Effects of environment temperature on heat and work', fontsize = 16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    
    #combined figure
    plt.figure(figsize = (9,5))
    
    plt.subplot(1,2,1)
    plt.plot(data[0][4], data[0][0], label = r'$\rho_{\uparrow\hspace{-0.5}\uparrow}$')
    plt.plot(data[0][4], data[0][1], label = r'$\rho_{\downarrow\hspace{-0.5}\downarrow}$')
    plt.xlabel('$\\frac{t}{t_{max}}$', fontsize = 16)
    plt.ylabel('Energy scale', fontsize = 12)
    plt.title('T = 0.1', fontsize = 12)
    
    plt.subplot(1,2,2)
    plt.plot(data[1][4], data[1][0], label = r'$\rho_{\uparrow\hspace{-0.5}\uparrow}$')
    plt.plot(data[1][4], data[1][1], label = r'$\rho_{\downarrow\hspace{-0.5}\downarrow}$')
    plt.xlabel('$\\frac{t}{t_{max}}$', fontsize = 16)
    plt.ylabel('Energy scale', fontsize = 12)
    plt.legend()
    plt.title('T = 0.001', fontsize = 12)
    
    # common title
    plt.suptitle('Effects of environment temperature on final spin-state', fontsize = 16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
        

