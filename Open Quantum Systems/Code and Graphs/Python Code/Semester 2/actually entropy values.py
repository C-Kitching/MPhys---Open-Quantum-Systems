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
def propagate_forward(t, dt, state, jump, tmax):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective_hamiltonian(t, tmax)*dt).dot(state) # phi(t+dt)
    
    dp = 1 - inner(state_1) # prob to jump
    
    # no jump
    if(random_one >= dp):
        
        jump = False
        
        no_jump_prefactor = (I-j*dt*effective_hamiltonian(t, tmax))/(np.sqrt(1-dp))
        new_state = no_jump_prefactor.dot(state)
        
    # jump
    elif(random_one < dp):
        
        jump = True
        
        random_two = np.random.uniform(0, 1)
        
        dp_plus = (state.conj().T).dot(sigma_plus(t, tmax).dot(sigma_minus(t, tmax).dot(state)))
        dp_minus = (state.conj().T).dot(sigma_minus(t, tmax).dot(sigma_plus(t, tmax).dot(state)))
        
        dp_plus_normalised = dp_plus / (dp_plus + dp_minus)
        dp_minus_normalised = dp_minus / (dp_plus + dp_minus)
        
        # jump up
        if(random_two >= dp_plus_normalised):
            jump_up_prefactor = sigma_plus(t, tmax)/np.sqrt(dp_minus/dt)
            new_state = jump_up_prefactor.dot(state)
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus(t, tmax)/np.sqrt(dp_plus/dt)
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
    

def single_MC(core_traj, tmax):
    
    # create master arrays
    master_heat = []
    master_density_op = []
    master_plus = []
    master_minus = []
    
    # set dt
    if(tmax < 20):
        dt = tmax/200
    else:
        dt = 0.1
    
    # loop over all trajectories
    for i in range(core_traj):
    
        # reset time and initial state 
        t = 0  # start time    
        initial_state = minus(0, tmax) # set the initial states initial state
        
        old_state = initial_state
        
        # reset heat to zero
        heat = 0
        
        # initialise jump counts to 0
        jump = False
                        
        # propgate forward in time
        for i in range(int(tmax/dt)):
          
             # get new state and time
            (new_state, t, jump) = propagate_forward(t, dt, old_state, jump, tmax)

            # normalise the states
            normalised_state = (1/np.linalg.norm(new_state))*new_state
            
            # if jump then heat exchanged
            if jump:
                heat += expectation(system_hamiltonian(t, tmax), normalised_state)- expectation(system_hamiltonian(t, tmax), old_state)
            
            # set the state to the normalised state
            old_state = normalised_state
            
            
        # calculate final density matrix
        density_op_final = outer(old_state, old_state)
        master_density_op.append(density_op_final)
        
        # populations
        plus_pop = (plus(tmax, tmax).conj().T).dot(density_op_final.dot(plus(tmax,tmax)))[0][0]
        minus_pop = (minus(tmax, tmax).conj().T).dot(density_op_final.dot(minus(tmax,tmax)))[0][0]
        
        # record        
        master_heat.append(heat)
        master_plus.append(plus_pop)
        master_minus.append(minus_pop)

    
    # average =
    avg_final_density_op = np.mean(master_density_op, axis = 0)
    avg_plus = np.mean(master_plus)
    avg_minus = np.mean(master_minus)
    
    # purity
    purity = np.real(trace(avg_final_density_op.dot(avg_final_density_op)))
    
    # calculate S
    S_final = np.real(-trace(avg_final_density_op.dot(logm(avg_final_density_op))))
    S = abs(S_final)
    
    # calculate J
    avg_heat = np.mean(master_heat)
    J = abs(avg_heat/T)
    
        
    return(avg_final_density_op, np.real(S), np.real(J), avg_plus, avg_minus, purity)

# calculate expectation of matrix and state
def expectation(matrix, state):
    return np.real((state.conj().T).dot(matrix.dot(state)))[0][0]


# calculate system hamiltonian
def system_hamiltonian(time, tmax):   
    # System Hamiltonian
    H_sys = (eta(time, tmax)/2)*(outer(plus(time, tmax), plus(time, tmax)) - outer(minus(time, tmax), minus(time, tmax)))
    
    return H_sys
        
# calculate effective hamitonian
def effective_hamiltonian(time, tmax):   
    H_eff = system_hamiltonian(time, tmax) - j*(gamma_0/2)*P_0(time, tmax).dot(P_0(time, tmax)) - j*(gamma_eta(time, tmax)/2)*(1+N(time, tmax))*P_eta(time, tmax).conj().T.dot(P_eta(time, tmax)) - j*(gamma_eta(time, tmax)/2)*N(time, tmax)*P_eta(time, tmax).dot(P_eta(time, tmax).conj().T)
    #H_eff = system_hamiltonian(time, tmax)
    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time, tmax):
    return 0.01*detuning(tmax, tmax)
    
# calculate time dependent bias
def detuning(time, tmax):  
    
    return 10*(time/tmax - 0.5)

# calculate time depedent P_0 operator
def P_0(time, tmax):
    return np.cos(theta(time, tmax))*(outer(plus(time, tmax), plus(time, tmax)) - outer(minus(time, tmax), minus(time, tmax)))

# calculate time depedent P_eta operator
def P_eta(time, tmax):
    return np.sin(theta(time, tmax))*(outer(minus(time, tmax), plus(time, tmax)))
    
#calculate time depedent minus state
def minus(time, tmax):
    return np.cos(theta(time, tmax)/2)*g-np.sin(theta(time, tmax)/2)*e
    
# calculate time depedent plus state
def plus(time, tmax):
    return np.sin(theta(time, tmax)/2)*g+np.cos(theta(time, tmax)/2)*e
    
# calculate time dependent theta
def theta(time, tmax):
    if(detuning(time, tmax) == 0):
        return np.pi/2
    else:
        return np.arctan2(tunnelling_coeff(time, tmax),detuning(time, tmax))

# calculate eta
def eta(time, tmax):
    return(np.sqrt(tunnelling_coeff(time, tmax)**2+detuning(time, tmax)**2))

# calculate time dep rate
def gamma_eta(time, tmax):
    return 2*np.pi*alpha*eta(time, tmax)

# calculate time dep sigma+
def sigma_plus(time, tmax):
    return outer(plus(time, tmax),minus(time, tmax))

# calculate time dep sigma-
def sigma_minus(time, tmax):
    return outer(minus(time, tmax),plus(time, tmax))

# calculate time dep sigma_x
def sigma_x(time, tmax):
    return outer(plus(time, tmax), minus(time, tmax)) + outer(minus(time, tmax), plus(time, tmax))

# calculate time dep sigma_z
def sigma_z(time, tmax):
    return outer(plus(time, tmax), plus(time, tmax)) - outer(minus(time, tmax), minus(time, tmax))

# calculate occupation number
def N(time, tmax):
    return 1/(np.exp(eta(time, tmax)/(k*T)) - 1)


j = 1j  # imaginary unit
k = 1 # boltzman constant 
T = 0.1 # temperature
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


# being main program
if __name__ == '__main__':
    
    
    start = timeit.default_timer() # start timer
    
    # check how many cores we have
    core_number = multiprocessing.cpu_count()
    print('You have {0:1d} CPU cores'.format(core_number))

    # traj number
    number_of_traj = 4
    
    core_traj = [int(number_of_traj/core_number) for i in range(core_number)]
    
    # different regimes
    regime = 2*np.logspace(-3,4, 8, base=10.0)

    #Create the worker pool
    pool = Pool(processes=core_number) 
    
    # containers for each regime
    master_J = []
    master_S = []
    master_op = []
    master_sigma = []
    master_plus = []
    master_minus = []
    master_purity = []
    
    # loop over different regimes
    for t in regime:
        
        # check point
        print(t)
             
        # parallel map
        MC = partial(single_MC, tmax = t)
        results = pool.map(MC, core_traj)
    
        # combine results of each cores calculations
        core_op = [results[i][0] for i in range(len(results))]
        core_S = [results[i][1] for i in range(len(results))]
        core_J = [results[i][2] for i in range(len(results))]
        core_plus = [results[i][3] for i in range(len(results))]
        core_minus = [results[i][4] for i in range(len(results))]
        core_purity = [results[i][5] for i in range(len(results))]
 
        # average each cores results
        avg_op = np.mean(core_op, axis = 0)
        avg_S = np.mean(core_S, axis = 0)
        avg_J = np.mean(core_J, axis = 0)
        avg_plus = np.mean(core_plus, axis = 0)
        avg_minus = np.mean(core_minus, axis = 0)
        avg_purity = np.mean(core_purity, axis = 0)
            
        # calculate total entropy production
        sigma = avg_S + avg_J
        
        # record on master array
        master_op.append(avg_op)
        master_J.append(avg_J)
        master_sigma.append(sigma)
        master_S.append(avg_S)
        master_plus.append(avg_plus)
        master_minus.append(avg_minus)
        master_purity.append(avg_purity)
        
    # close the pool
    pool.close()
        
    # output calculation time
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    
    # populations
    plt.figure()
    plt.title('Adiabiatc state populations')
    plt.plot(np.log(regime), master_minus, label = r'$\rho_{--}$')
    plt.plot(np.log(regime), master_plus, label = r'$\rho_{++}$')
    plt.legend()
    plt.xlabel(r'$log(t_{max})$')
    plt.ylabel('$\overline{P}$')
        
    # purity
    plt.figure()
    plt.title('Purity')
    plt.plot(np.log(regime), master_purity)
    plt.xlabel(r'$log(t_{max})$')
    
    
    #combined figure
    plt.figure(figsize = (9,5))
    plt.subplot(1,2,2)
    cum_S = np.cumsum(master_S)
    #master_S = [master_S[i]/max(master_S) for i in range(len(master_S))]
    plt.plot(np.log10(regime), cum_S, label = r'$|\hspace{0.1}S\hspace{0.1}|$')
    plt.legend(fontsize = 14)
    plt.xlabel(r'$log(t_{max})$', fontsize = 14)
    plt.ylabel('Normalised entropy scale', fontsize = 14)
    plt.title('von-Neumann entropy', fontsize = 14)
    
    plt.subplot(1,2,1)
    cum_J = np.cumsum(master_J)
    #master_J = [master_J[i]/max(master_J) for i in range(len(master_J))]
    plt.plot(np.log10(regime), cum_J)
    plt.xlabel(r'$log(t_{max})$', fontsize = 14)
    plt.ylabel('Normalised entropy scale', fontsize = 14)
    plt.legend(fontsize = 14)
    plt.title('Entropy flow', fontsize = 14)
    
    # common title
    plt.suptitle('Entropy components', fontsize = 16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    
    
    
    
    
    