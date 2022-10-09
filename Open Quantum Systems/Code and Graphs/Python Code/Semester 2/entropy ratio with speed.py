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
def propagate_forward(t, dt, state, down_jumps, up_jumps, tmax):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective_hamiltonian(t, tmax)*dt).dot(state) # phi(t+dt)
    dp = 1 - inner(state_1) # prob to jump

    # no jump
    if(random_one >= dp):
        no_jump_prefactor = (I-j*dt*effective_hamiltonian(t, tmax))/(np.sqrt(1-dp))
        new_state = no_jump_prefactor.dot(state)
        
    # jump
    elif(random_one < dp):
        
        random_two = np.random.uniform(0, 1)
        
        dp_plus = (state.conj().T).dot(sigma_plus(t, tmax).dot(sigma_minus(t, tmax).dot(state)))
        dp_minus = (state.conj().T).dot(sigma_minus(t, tmax).dot(sigma_plus(t, tmax).dot(state)))
        
        dp_plus_normalised = dp_plus / (dp_plus + dp_minus)
        dp_minus_normalised = dp_minus / (dp_plus + dp_minus)
        
        # jump up
        if(random_two >= dp_plus_normalised):
            jump_up_prefactor = sigma_plus(t, tmax)/np.sqrt(dp_minus/dt)
            new_state = jump_up_prefactor.dot(state)
            up_jumps += 1
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus(t, tmax)/np.sqrt(dp_plus/dt)
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
    

def single_MC(core_traj, tmax):
    
    # create master arrays
    master_density_op = []
    master_down_jumps = []
    master_up_jumps = []
    master_omega = []
    
    # set dt
    if(tmax < 0.2):
        dt = tmax/200
    else:
        dt = 0.1
    
    # loop over all trajectories
    for i in range(core_traj):
    
        # reset time and initial state 
        t = 0  # start time    
        initial_state = minus(0, tmax) # set the initial states initial state
        
        record_states = []  # blank array to record states
        record_time = []  # blank array to record time
        
        # record initial values
        record_states.append(initial_state)
        record_time.append(t)
        
        state = initial_state
        
        # initialise jump counts to 0
        old_up_jumps= 0
        old_down_jumps = 0
        
        # record jump counts at each step
        record_up_jumps = []
        record_down_jumps = []
        
        # append initial jump counts
        record_up_jumps.append(old_up_jumps)
        record_down_jumps.append(old_down_jumps)
        
        # record the points the jumps occur
        jump_points = []
        jump_points.append(0)
        
        # propgate forward in time
        for i in range(int(tmax/dt)):
          
            # get new state and time
            (state, t, new_down_jumps, new_up_jumps) = propagate_forward(t, dt, state, old_down_jumps, old_up_jumps, tmax)
            
            # record jumps
            record_up_jumps.append(new_up_jumps)
            record_down_jumps.append(new_down_jumps)
            
            # update values
            old_up_jumps = new_up_jumps
            old_down_jumps = new_down_jumps
             
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
            
        # calculate omega
        omega = []
        for i in range(len(record_states)):
            omega.append(expectation(system_hamiltonian(t, tmax),plus(t, tmax))-expectation(system_hamiltonian(t, tmax),minus(t, tmax)))
            
        master_omega.append(omega)
        master_density_op.append(reduced_density_op_matrices)
        master_up_jumps.append(record_up_jumps)
        master_down_jumps.append(record_down_jumps)

    avg_omega = np.mean(master_omega, axis = 0)
    avg_density_op = np.mean(master_density_op, axis = 0)
    avg_up_jumps = np.mean(master_up_jumps, axis = 0)
    avg_down_jumps = np.mean(master_down_jumps, axis = 0)
        
    return(avg_density_op, record_time, avg_up_jumps, avg_down_jumps, avg_omega)

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
    number_of_traj = 40
    
    core_traj = [int(number_of_traj/core_number) for i in range(core_number)]
    
    # different regimes
    regime = [0.02*pow(10, i) for i in range(5)]
    regime = np.logspace(-1, 2, 20)

    #Create the worker pool
    pool = Pool(processes=core_number) 
    
    # containers for each regime
    master_J = []
    master_S = []
    master_s = []
    master_sigma = []
    master_psigma = []
    
    # loop over different regimes
    for t in regime:
        
        # check point
        print(t)
        
        # set dt
        if(t < 20):
            dt = t/200
        else:
            dt = 0.1
        
        MC = partial(single_MC, tmax = t)
        
        # parallel map
        results = pool.map(MC, core_traj)
    
        # combine results of each cores calculations
        core_op = [results[i][0] for i in range(len(results))]
        core_up = [results[i][2] for i in range(len(results))]
        core_down = [results[i][3] for i in range(len(results))]
        core_omega = [results[i][4] for i in range(len(results))]
 
        # average each cores results
        avg_omega = np.mean(core_omega, axis = 0)
        avg_density_op = np.mean(core_op, axis = 0)
        avg_up_jumps = np.mean(core_up, axis = 0)
        avg_down_jumps = np.mean(core_down, axis = 0)
        
        # calculate jump difference
        jump_diff = [avg_down_jumps[i] - avg_up_jumps[i] for i in range(len(avg_down_jumps))]

        # calculate S
        S = []
        s=[]
        for i in range(1, len(avg_density_op)):
            s1 = -trace(avg_density_op[i-1].dot(logm(avg_density_op[i-1])))
            s2 = -trace(avg_density_op[i].dot(logm(avg_density_op[i])))
            S.append((s2-s1)/dt) 
            s.append(-trace(avg_density_op[i].dot(logm(avg_density_op[i]))))
            
        # calculate J
        J = []
        for i in range(1, len(avg_density_op)):
            J.append(np.real((avg_omega[i]/T)*((jump_diff[i])-(jump_diff[i-1]))/dt))
            
        # calculate total entropy production
        sigma = []
        for i in range(len(S)):
            sigma.append(np.real(S[i] + J[i]))
            
        # cummulative sum
        #J_final = np.cumsum(J)[-1]
        #sigma_final = np.cumsum(sigma)[-1]
        #S_final = np.cumsum(S)[-1]
        #s_final = np.cumsum(s)[-1]
        
        # mean
        J_final = np.mean(J)
        sigma_final = np.mean(sigma)
        s_final = np.mean(s)
        S_final = np.mean(S)
        
        p_sigma = abs(S_final) + abs(J_final)
        
        # record on master array
        master_J.append(abs(J_final))
        master_sigma.append(abs(sigma_final))
        master_S.append(abs(S_final))
        master_s.append(abs(s_final))
        master_psigma.append(p_sigma)
        
        
        
       
    # close the pool
    pool.close()
        
    # output calculation time
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    
    #figure
    S_sigma = [master_S[i]/master_sigma[i] for i in range(len(master_sigma))]
    J_sigma = [master_J[i]/master_sigma[i] for i in range(len(master_sigma))]
    plt.plot(np.log(regime), S_sigma, label = r'$\frac{\frac{dS}{dt}}{\sigma}$')
    plt.plot(np.log(regime), J_sigma, label = r'$\frac{J}{\sigma}$')
    plt.xlabel(r'$log(t_{max})$')
    plt.legend()
    plt.ylabel('Entropy')
    plt.title('Entropy ratios with ramp speed')  
    
    plt.figure()
    plt.plot(np.log(regime), master_S, label = r'$\overline{\frac{dS}{dt}}$')
    plt.plot(np.log(regime), master_J, label = '$\overline{J}$')
    plt.plot(np.log(regime), master_sigma, label = r'$\overline{\sigma}$')
    plt.xlabel(r'$log(t_{max})$')
    plt.legend()
    plt.ylabel('Entropy')
    plt.title('Entropy balance components with ramp speed')
    
    norm_scale = [(master_S[i]**2+master_J[i]**2)**(1/2) for i in range(len(master_S))]
    #figure
    plt.figure()
    S_sigma = [master_S[i]/norm_scale[i] for i in range(len(norm_scale))]
    J_sigma = [master_J[i]/norm_scale[i] for i in range(len(norm_scale))]
    plt.plot(np.log(regime), S_sigma, label = r'$\frac{dS}{dt}$')
    plt.plot(np.log(regime), J_sigma, label = r'$j$')
    plt.xlabel(r'$log(t_{max})$')
    plt.legend()
    plt.ylabel('Entropy')
    plt.title('Entropy ratios with ramp speed') 
    
    
    
    
    
    
    
    
    
    
    
    
    
    