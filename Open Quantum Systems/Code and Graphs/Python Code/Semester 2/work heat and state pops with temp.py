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
def propagate_forward(t, dt, state, jump, temperature):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective_hamiltonian(t, temperature)*dt).dot(state) # phi(t+dt)
    
    dp = 1 - inner(state_1) # prob to jump
    
    # no jump
    if(random_one >= dp):
        
        jump = False
        
        no_jump_prefactor = (I-j*dt*effective_hamiltonian(t, temperature))/(np.sqrt(1-dp))
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
    

def single_MC(core_traj, temperature):
    
    # create master arrays
    master_work = []
    master_heat = []
    master_pop_plus = []
    master_pop_minus = []
    master_spin_up = []
    master_spin_down = []
        
    # loop over all trajectories
    for i in range(core_traj):
    
        # reset time and initial state 
        t = 0  # start time    
        initial_state = minus(0) # set the initial states initial state
                
        old_state = initial_state
        
        # reset heat to zero
        heat = 0
        
        # initialise jump counts to 0
        jump = False
        
        # propgate forward in time
        for i in range(int(tmax/dt)):
          
            # get new state and time
            (new_state, t, jump) = propagate_forward(t, dt, old_state, jump, temperature)

            # normalise the states
            normalised_state = (1/np.linalg.norm(new_state))*new_state
            
            # if jump then heat exchanged
            if jump:
                heat += expectation(system_hamiltonian(t), normalised_state)- expectation(system_hamiltonian(t), old_state)
            # if no jump then work done
            
            # set the state to the normalised state
            old_state = normalised_state
            
        # calculate energy change over trajectory
        total_energy_change = expectation(system_hamiltonian(tmax), old_state)- expectation(system_hamiltonian(0), initial_state)
          
        # calculate work done on traj
        work = total_energy_change - heat
        
        # get final density operator
        density_op = outer(old_state, old_state)
        
        # get populations
        minus_pop = np.real(((minus(tmax).conj().T).dot(density_op)).dot(minus(tmax))[0][0])
        plus_pop = np.real(((plus(tmax).conj().T).dot(density_op)).dot(plus(tmax))[0][0])
        up_pop = np.real(((e.conj().T).dot(density_op)).dot(e)[0][0])
        down_pop = np.real(((g.conj().T).dot(density_op)).dot(g)[0][0])
                                    
        # add each traj variables to master array
        master_work.append(work)
        master_heat.append(heat)
        master_pop_plus.append(plus_pop)
        master_pop_minus.append(minus_pop)
        master_spin_up.append(up_pop)
        master_spin_down.append(down_pop)
        
    # average over trajectories
    avg_pop_plus = np.mean(master_pop_plus)
    avg_pop_minus = np.mean(master_pop_minus)
    avg_spin_up = np.mean(master_spin_up)
    avg_spin_down = np.mean(master_spin_down)
    avg_work = np.mean(master_work)
    avg_heat = np.mean(master_heat)
        
    return(avg_work, avg_heat, avg_pop_plus, avg_pop_minus, avg_spin_up, avg_spin_down)

# calculate expectation of matrix and state
def expectation(matrix, state):
    return np.real((state.conj().T).dot(matrix.dot(state)))[0][0]


# calculate system hamiltonian
def system_hamiltonian(time):   
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))
    
    return H_sys
        
# calculate effective hamitonian
def effective_hamiltonian(time, temperature):   
    H_eff = system_hamiltonian(time) - j*(gamma_0(temperature)/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time)/2)*(1+N(time, temperature))*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time)/2)*N(time, temperature)*P_eta(time).dot(P_eta(time).conj().T)
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
def N(time, temperature):
    return 1/(np.exp(eta(time)/(k*T)) - 1)

# gamma 0
def gamma_0(temperature):
    return 4*np.pi*alpha*k*temperature



j = 1j  # imaginary unit
little_omega = 1 # oprtical transition
k = 1 # boltzman constant 
T = 0.1 # temperature
alpha = 1/(12*np.pi) # coupling strength


# excited and ground states
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
I = np.array([[1,0], [0,1]])

# time parameters
tmax = 20000
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
    
    # different regimes
    temp = [0.001*pow(10, i) for i in range(3)]
    
    # containers for each regime
    master_minus = []
    master_plus = []
    master_up = []
    master_down = []
    master_work= []
    master_heat = []
    
    # loop over different regimes
    for T in temp:
        
        print(T)
                
        MC = partial(single_MC, temperature = T)
        
        # parallel map
        results = pool.map(MC, core_traj)
    
        # combine results of each cores calculations
        core_work = [results[i][0] for i in range(len(results))]
        core_heat = [results[i][1] for i in range(len(results))]  
        plus_pop = [results[i][2] for i in range(len(results))]
        minus_pop = [results[i][3] for i in range(len(results))]
        spin_up = [results[i][4] for i in range(len(results))]
        spin_down = [results[i][5] for i in range(len(results))]
 
        # average each cores results
        minus_avg = np.mean(minus_pop, axis = 0)
        plus_avg = np.mean(plus_pop, axis = 0)
        up_avg = np.mean(spin_up, axis = 0)
        down_avg = np.mean(spin_down, axis = 0)
        work = np.mean(core_work, axis = 0)
        heat = np.mean(core_heat, axis = 0)

        # store results on each regime
        master_plus.append(plus_avg)
        master_minus.append(minus_avg)
        master_up.append(up_avg)
        master_down.append(down_avg)
        master_work.append(work)
        master_heat.append(heat)
   
       
    # close the pool
    pool.close()
        
    # output calculation time
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    
    # spin states
    plt.figure()
    plt.plot(np.log(temp), master_up, label = r'$\rho_{\uparrow}$')
    plt.plot(np.log(temp), master_down, label = r'$\rho_{\downarrow}$')
    plt.xlabel(r'$t_{max}$')
    plt.ylabel('$\overline{P}$')
    plt.legend()
    plt.title('Final spin-state population with ramp speed')
    
    # energy states
    plt.figure()
    plt.plot(np.log(temp), master_plus, label = r'$\rho_{++}$')
    plt.plot(np.log(temp), master_minus, label = r'$\rho_{--}$')
    plt.xlabel(r'$log(T)$')
    plt.ylabel('$\overline{P}$')
    plt.legend()
    plt.title('Final energy-state population with temperature')
    
    # heat and work
    plt.figure()
    plt.plot(np.log(temp), master_work, label = r'Work')
    plt.plot(np.log(temp), master_heat, label = r'Heat')
    plt.xlabel(r'$log(T)$')
    plt.ylabel('Energy scale')
    plt.legend()
    plt.title('Heat transferred and work done with temperature')
    
    #combined figure
    plt.figure(figsize = (9,5))
    
    plt.subplot(1,2,2)
    plt.plot(np.log(temp), master_up, label = r'$\rho_{\uparrow\hspace{-0.5}\uparrow}$')
    plt.plot(np.log(temp), master_down, label = r'$\rho_{\downarrow\hspace{-0.5}\downarrow}$')
    plt.xlabel(r'$log(T)$', fontsize = 14)
    plt.ylabel('$\overline{P}$', fontsize = 14)
    plt.legend()
    plt.title('Final spin-state population', fontsize = 14)
    
    plt.subplot(1,2,1)
    plt.plot(np.log(temp), master_work, label = r'Work on system')
    plt.plot(np.log(temp), master_heat, label = r'Heat to system')
    plt.xlabel(r'$log(T)$', fontsize = 14)
    plt.ylabel('Energy scale', fontsize = 14)
    plt.legend()
    plt.title('Heat and work', fontsize = 14)
    
    # common title
    plt.suptitle('Effects of environment temperature', fontsize = 16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    
    
    
    
    