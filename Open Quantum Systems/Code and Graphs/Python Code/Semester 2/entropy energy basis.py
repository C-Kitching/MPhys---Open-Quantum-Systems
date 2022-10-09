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
def propagate_forward(t, dt, state, down_jumps, up_jumps):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective_hamiltonian(t)*dt).dot(state) # phi(t+dt)
    dp = 1 - inner(state_1) # prob to jump

    # no jump
    if(random_one >= dp):
        no_jump_prefactor = (I-j*dt*effective_hamiltonian(t))/(np.sqrt(1-dp))
        new_state = no_jump_prefactor.dot(state)
        
    # jump
    elif(random_one < dp):
        
        print('jump')
        
        random_two = np.random.uniform(0, 1)
        
        dp_plus = (state.conj().T).dot(sigma_plus(t).dot(sigma_minus(t).dot(state)))
        dp_minus = (state.conj().T).dot(sigma_minus(t).dot(sigma_plus(t).dot(state)))
        
        dp_plus_normalised = dp_plus / (dp_plus + dp_minus)
        dp_minus_normalised = dp_minus / (dp_plus + dp_minus)
        
        # jump up
        if(random_two >= dp_plus_normalised):
            jump_up_prefactor = sigma_plus(t)/np.sqrt(dp_minus/dt)
            new_state = jump_up_prefactor.dot(state)
            up_jumps += 1
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus(t)/np.sqrt(dp_plus/dt)
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
    

def single_MC(core_traj):
    
    # create master arrays
    master_density_op = []
    master_down_jumps = []
    master_up_jumps = []
    master_omega = []
    master_plus = []
    master_minus = []
    
    # loop over all trajectories
    for i in range(core_traj):
    
        # reset time and initial state 
        t = 0  # start time    
        initial_state = minus(0) # set the initial states initial state
        
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
            (state, t, new_down_jumps, new_up_jumps) = propagate_forward(t, dt, state, old_down_jumps, old_up_jumps)
            
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
            omega.append(expectation(system_hamiltonian(record_time[i]),plus(record_time[i]))-expectation(system_hamiltonian(record_time[i]),minus(record_time[i])))
            
        population_plus = []  # blank array to store population numbers
        population_minus = []
        
        # extract the population
        for i in range(len(reduced_density_op_matrices)):
            population_minus.append(((minus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(minus(record_time[i]))[0][0])
            population_plus.append(((plus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(plus(record_time[i]))[0][0])
            
        population_plus = np.real(population_plus)
        population_minus = np.real(population_minus)
            
        master_plus.append(population_plus)
        master_minus.append(population_minus)
        master_omega.append(omega)
        master_density_op.append(reduced_density_op_matrices)
        master_up_jumps.append(record_up_jumps)
        master_down_jumps.append(record_down_jumps)
        
    avg_minus = np.mean(master_minus, axis = 0)
    avg_plus = np.mean(master_plus, axis = 0)
    avg_omega = np.mean(master_omega, axis = 0)
    avg_density_op = np.mean(master_density_op, axis = 0)
    avg_up_jumps = np.mean(master_up_jumps, axis = 0)
    avg_down_jumps = np.mean(master_down_jumps, axis = 0)
        
    return(avg_density_op, record_time, avg_up_jumps, avg_down_jumps, avg_omega, avg_plus, avg_minus)

# calculate expectation of matrix and state
def expectation(matrix, state):
    return np.real((state.conj().T).dot(matrix.dot(state)))[0][0]


# calculate system hamiltonian
def system_hamiltonian(time):   
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))
    
    return H_sys
        
# calculate effective hamitonian
def effective_hamiltonian(time):   
    H_eff = system_hamiltonian(time) - j*(gamma_0/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time)/2)*(1+N(time))*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time)/2)*N(time)*P_eta(time).dot(P_eta(time).conj().T)
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

# calculate time dep sigma_z
def sigma_z(time):
    return outer(plus(time), plus(time)) - outer(minus(time), minus(time))

# calculate occupation number
def N(time):
    return 1/(np.exp(eta(time)/(k*T)) - 1)


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


# time parameters
t = 0  # start time 
tmax = 200 # end time 
dt = 0.1  # stepsize


# being main program
if __name__ == '__main__':
    
    """
    regime = np.logspace(-3,3,20)
    
    master_S = []
    master_J = []
    
    for t in regime:
    
        tmax = t
        
        # set dt
        if(tmax < 20):
            dt = tmax/200
        else:
            dt = 0.1
        
        results = single_MC(1)
        
        op = results[0]
        up = results[2]
        down = results[3]
        omega = results[4]
        
        jump_diff = [down[i]-up[i] for i in range(len(up))]
        
        # calculate S
        S = []
        s=[]
        time_S = []
        for i in range(1, len(op)):
            s1 = -trace(op[i-1].dot(logm(op[i-1])))
            s2 = -trace(op[i].dot(logm(op[i])))
            S.append(np.real((s2-s1)/dt)) 
            s.append(np.real(-trace(op[i].dot(logm(op[i])))))
        master_S.append(s[-1])
            
        # calculate J
        J = []
        for i in range(1, len(op)):
            J.append((omega[i]/T)*((jump_diff[i])-(jump_diff[i-1]))/dt)
        J = [abs(J[i]) for i in range(len(J))]
        J = np.cumsum(J)
        master_J.append(J[-1])
        
    plt.figure()
    plt.plot(np.log(regime), master_S)
    plt.figure()
    plt.plot(np.log(regime), master_J)
    
    
    """
    for i in range(1):
    
        start = timeit.default_timer() # start timer
        
        # check how many cores we have
        core_number = multiprocessing.cpu_count()
        print('You have {0:1d} CPU cores'.format(core_number))
    
        # traj number
        number_of_traj = 40
        
        core_traj = [int(number_of_traj/core_number) for i in range(core_number)]
    
        #Create the worker pool
        pool = Pool(processes=core_number) 
    
        # parallel map
        results = pool.map(single_MC, core_traj)
        
        # close the pool
        pool.close()
        
        # combine results of each cores calculations
        core_op = [results[i][0] for i in range(len(results))]
        time = results[0][1]
        core_up = [results[i][2] for i in range(len(results))]
        core_down = [results[i][3] for i in range(len(results))]
        core_omega = [results[i][4] for i in range(len(results))]
        core_plus = [results[i][5] for i in range(len(results))]
        core_minus = [results[i][6] for i in range(len(results))]
    
        # average each cores results
        avg_omega = np.mean(core_omega, axis = 0)
        avg_density_op = np.mean(core_op, axis = 0)
        avg_up_jumps = np.mean(core_up, axis = 0)
        avg_down_jumps = np.mean(core_down, axis = 0)
        avg_plus = np.mean(core_minus, axis = 0)
        avg_minus = np.mean(core_plus, axis = 0)
        
        # calculate jump difference
        jump_diff = [avg_down_jumps[i] - avg_up_jumps[i] for i in range(len(avg_down_jumps))]
        
        stop = timeit.default_timer()
    
        print('Time: ', stop - start) 
        
        # calculate S
        S = []
        s=[]
        time_S = []
        for i in range(1, len(avg_density_op)):
            s1 = -trace(avg_density_op[i-1].dot(logm(avg_density_op[i-1])))
            s2 = -trace(avg_density_op[i].dot(logm(avg_density_op[i])))
            S.append(np.real((s2-s1)/dt)) 
            s.append(np.real(-trace(avg_density_op[i].dot(logm(avg_density_op[i])))))
            time_S = np.delete(time,0)
        print(s[-1])
            
        # calculate J
        J = []
        for i in range(1, len(avg_density_op)):
            J.append((avg_omega[i]/T)*((jump_diff[i])-(jump_diff[i-1]))/dt)
        J = [abs(J[i]) for i in range(len(J))]
        J = np.cumsum(J)
            
        # calculate total entropy production
        sigma = []
        for i in range(len(S)):
            sigma.append(np.real(S[i] + J[i]))
            
        # calculate state purity
        pure = []
        for i in range(len(avg_density_op)):  
            pure.append(np.real(trace(avg_density_op[i].dot(avg_density_op[i]))))
            
        # normalise time
        norm_time = [time_S[i]/max(time_S) for i in range(len(time_S))]
        time = [time[i]/max(time) for i in range(len(time))]
        
        # plot populations
        plt.figure()
        plt.plot(time, avg_plus, label = r'$\rho_{++}$')
        plt.plot(time, avg_minus, label = r'$\rho_{--}$')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel(r'$\overline{P}$')
        plt.title('Adiabiatc state populations')
        
        # purity
        plt.figure()
        plt.plot(time, pure)
        plt.xlabel("$t$")
        plt.ylim(0,1.2)
        plt.ylabel("$Tr(\\rho^{2})$")
        plt.title("Purity of state")
        plt.show()
        
        # von Neuman entropy
        plt.figure()
        plt.plot(norm_time, s)
        plt.xlabel('$\\frac{t}{t_{max}}$')
        plt.ylabel('$S$')
        plt.title('von-Neuman entropy')
        plt.show()
        
        # cumm J
        plt.figure()
        plt.plot(norm_time, J)
        plt.title('Entropy flux')
        plt.xlabel('$\\frac{t}{t_{max}}$')
        plt.ylabel('$J$')
        plt.show()
        
        print(s[-1])
        print(J[-1])    
    
    """
    # graph
    plt.figure()
    plt.plot(norm_time, S)
    plt.title('von-Neuman entropy change for %s trajectories' %number_of_traj)
    plt.xlabel('$\\frac{t}{t_{max}}$')
    plt.ylabel('$\\frac{dS}{dt}$')
    plt.show()
    
    # graph
    plt.figure()
    plt.plot(norm_time, J)
    plt.title('Entropy flux for %s trajectories' %number_of_traj)
    plt.xlabel('$\\frac{t}{t_{max}}$')
    plt.ylabel('$J$')
    plt.show()
    
    # graph
    plt.figure()
    plt.plot(norm_time, sigma)
    plt.title('Entropy production rate for %s trajectories' %number_of_traj)
    plt.xlabel('$\\frac{t}{t_{max}}$')
    plt.ylabel('$\sigma$')
    plt.show()
    """





















