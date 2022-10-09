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
def propagate_forward(t, dt, state, jump):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective_hamiltonian(t)*dt).dot(state) # phi(t+dt)
    
    dp = abs(1 - inner(state_1)) # prob to jump
    
    # no jump
    if(random_one >= dp):
        
        jump = False
        
        no_jump_prefactor = (I-j*dt*effective_hamiltonian(t))/(np.sqrt(1-dp))
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
    

def single_MC(core_traj):
    
    # create master arrays
    master_work = []
    master_heat = []
    master_spin_up = []
    master_spin_down = []
    
    # loop over all trajectories
    for i in range(core_traj):
    
        # reset time and initial state 
        t = 0 
        initial_state = minus(0) 
        
        old_state = initial_state
        
        # initialise jump counts to 0
        jump = False
        
        # reset heat to 0
        heat = 0
        
        # propgate forward in time
        for i in range(int(tmax/dt)):
          
            # get new state and time
            (new_state, t, jump) = propagate_forward(t, dt, old_state, jump)

            # normalise the states
            normalised_state = (1/np.linalg.norm(new_state))*new_state
            
            # if jump then heat exchanged
            if jump:
                heat += expectation(system_hamiltonian(t), normalised_state)- expectation(system_hamiltonian(t), old_state)

            # set the state to the normalised state
            old_state = normalised_state
            
        # calculate system energy change over the trajectory
        sys_energy_change = expectation(system_hamiltonian(tmax), normalised_state)- expectation(system_hamiltonian(0), initial_state)
                        
        # calculate work done on traj
        work = sys_energy_change - heat
        
        # get final density matix
        density_matrix = outer(old_state, old_state)
        
        spin_up = (e.conj().T).dot(density_matrix.dot(e))[0][0]
        spin_down = (g.conj().T).dot(density_matrix.dot(g))[0][0]
                          
        # remove complex part
        spin_up = np.real(spin_up)
        spin_down = np.real(spin_down)
        
        # append to arrays
        master_work.append(work)
        master_heat.append(heat)
        master_spin_up.append(spin_up)
        master_spin_down.append(spin_down)
        
    # average over trajectories
    avg_work = np.mean(master_work)
    avg_heat = np.mean(master_heat)
    avg_up = np.mean(master_spin_down)
    avg_down = np.mean(master_spin_up)
        
    return(avg_work, avg_heat, avg_up, avg_down)

# calculate expectation of matrix and state
def expectation(matrix, state):
    return np.real((state.conj().T).dot(matrix.dot(state)))[0][0]


# calculate system hamiltonian
def system_hamiltonian(time):   
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))    
    return H_sys
        
# calculate effective hamitonian
def effective_hamiltonian(time):   
    H_eff = system_hamiltonian(time) - j*(gamma_0/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time)/2)*(1+N(time))*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time)/2)*N(time)*P_eta(time).dot(P_eta(time).conj().T)
    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):       
    return 0.01*detuning(tmax)
    
# calculate time dependent bias
def detuning(time):  
    return 20*(time/tmax + 1)

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
def N(time):
    return 1/(np.exp(eta(time)/(k*T)) - 1)



j = 1j  # imaginary unit
little_omega = 1 # oprtical transition
k = 1 # boltzman constant 
T = 1 # temperature
alpha = 1/(12*np.pi) # coupling strength

# Time indep rate
gamma_0= 4*np.pi*alpha*k*T

# bosonic occupation number
#N = 1/(np.exp(little_omega/(k*T)) - 1)

# excited and ground states
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
I = np.array([[1,0], [0,1]])

# time parameters
t = 0  # start time 
tmax = 1 # end time 
dt = 0.01  # stepsize


results = single_MC(1)


# combine core results
core_work = results[0]
core_heat = results[1]
core_up = results[2]
core_down = results[2]



'''
# being main program
if __name__ == '__main__':
    
    start = timeit.default_timer() # start timer
    
    # check how many cores we have
    core_number = multiprocessing.cpu_count()
    print('You have {0:1d} CPU cores'.format(core_number))

    # traj number
    number_of_traj = 20
    
    # split traj over cores
    core_traj = [int(number_of_traj/core_number) for i in range(core_number)]
    
    # record speeds
    ramp = []
    
    # total work and heat on traj
    master_work = []
    master_heat = []
    
    # final spin states
    master_up = []
    master_down = []
    
    #Create the worker pool
    pool = Pool(processes=core_number) 
    
    # loop over speeds 
    for i in range(20):
    
        ramp.append((detuning(tmax)-detuning(0))/tmax)
        
        # parallel map
        results = pool.map(single_MC, core_traj)
        
        # combine core results
        core_work = results[0]
        core_heat = results[1]
        core_up = results[2]
        core_down = results[2]
    
        # average core results
        work = np.mean(core_work, axis = 0)
        heat = np.mean(core_heat, axis = 0)
        spin_up = np.mean(core_up, axis = 0)
        spin_down = np.mean(core_down, axis = 0)
               
        # record values for each speed
        master_work.append(work)
        master_heat.append(heat)
        master_up.append(spin_up)
        master_down.append(spin_down)
        
        # incremement the max time
        tmax += 1
        
    # close the pool
    pool.close()
    
    # spin states plot
    plt.figure()
    plt.plot(ramp, master_up, label = r'$\left|\uparrow \right>$')
    plt.plot(ramp, master_up, label = r'$\left|\downarrow \right>$')
    plt.xlabel('$\\frac{\partial\epsilon(t)}{\partial t}$')
    plt.ylabel('Final spin state populations')
    plt.title('Final spin state with ramp speed')
    
    plt.figure()
    plt.plot(ramp, master_work)
    plt.xlabel('$\\frac{\partial\epsilon(t)}{\partial t}$')
    plt.ylabel('Work')
    plt.title('Work done with ramp speed')
''' 




