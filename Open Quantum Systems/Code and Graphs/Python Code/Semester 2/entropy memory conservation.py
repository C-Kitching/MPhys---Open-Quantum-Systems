import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import expm, sinm, cosm, eig, logm
import sys
import multiprocessing
import timeit
from multiprocessing import Pool, cpu_count
from functools import partial

# propogate forward in time
def propagate_forward(state, up_jumps, down_jumps, core_traj):
    
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
            up_jumps += 1
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus/np.sqrt(dp_plus/dt)
            new_state = jump_down_prefactor.dot(state)
            down_jumps += 1

    return(new_state, up_jumps, down_jumps)

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
    
    

j = 1j  # imaginary unit


delta = 1 # tunnelling 
epsilon = 0  # bias
gamma = delta / 6 # rates
little_omega = 1 # oprtical transition
eta = np.sqrt(epsilon**2+delta**2)
k = 8.617e-5 # boltzman constant eV/K
T = 4000 # temperature
alpha = 1/(12*np.pi) # coupling strength

# excited and ground states
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# bosonic occupation number
N = 1/(np.exp(little_omega/(k*T)) - 1)

# Pauli matricies
sigma_plus = np.array([[0,1],[0,0]])
sigma_minus = np.array([[0,0],[1,0]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0, -j], [-j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
I = np.array([[1,0], [0,1]])

# time parameters
dt = 0.01  # stepsize
t = 0  # start time 
tmax = 20  # end time 

# Hamiltonian
H_sys = (epsilon/2)*sigma_z + (delta/2)*sigma_x
H_eff = H_sys - j*gamma*N*sigma_minus.dot(sigma_plus) - j*gamma*(1 + N)*sigma_plus.dot(sigma_minus)
#H_eff = H_sys


# being main program
if __name__ == '__main__':
    
    start = timeit.default_timer() # start timer
    
    # create the time array
    time = np.linspace(t, tmax, int(tmax/dt))
    
    # check how many cores we have
    core_number = multiprocessing.cpu_count()
    print('You have {0:1d} CPU cores'.format(core_number))
    
    # set the initial state
    initial_state = g 

    # traj number
    number_of_traj = 1000
    
    # each core calls function once
    core_traj = [int(number_of_traj/core_number) for i in range(core_number)]
    
    # set initial up and down jumps to zero
    up_jump = 0
    down_jump = 0
    
    # set current state to initial state
    state = initial_state
    
    #Create the worker pool
    pool = Pool(processes=core_number) 
    
    # loop to devide trajectories between core
    for i in range(1, len(time)):
        
        # func to multiple parameters
        func = partial(propagate_forward, state, up_jump, down_jump)
        
        # parallel map
        results = pool.map(func, core_traj)
        
        # extract results
        states = [results[i][0] for i in range(core_number)]
        up_jumps = [results[i][1] for i in range(core_number)]
        down_jumps = [results[i][2] for i in range(core_number)]
        
        # average each cores results CHECK THIS
        avg_state = np.mean(states, axis = 0)
        avg_up_jump = np.mean(up_jumps, axis = 0)
        avg_down_jump = np.mean(down_jumps, axis = 0)
        
        # store the data we just used as old
        old_state = state
        old_up_jump = up_jump
        old_down_jump = down_jump
        
        # set the new data as the input data for the next pass
        state = avg_state
        up_jump = avg_up_jump
        down_jump = avg_down_jump
              
        # generate old and current density matrix
        old_density_matrix = outer(old_state, old_state)
        density_matrix = outer(state, state)
        
        # calculate von Neumann entropy at t
        s1 = -trace(old_density_matrix.dot(logm(old_density_matrix)))
        s2 = -trace(density_matrix.dot(logm(density_matrix)))
        
        # calculate von Neumann rate
        S = (s2 -s1)/dt
        
        # plot the single point
        plt.scatter(time[i], S, color = 'b')
            
    # close and join the pool
    pool.close()
    pool.join()
    
    plt.show()
    

"""    
# calculate error of average dist
std_density_op = np.std(master_density_op, axis = 0) / np.sqrt(number_of_traj)

# calculate S
S = []
s=[]
time_S = []
for i in range(1, len(avg_density_op)):
    s1 = -trace(avg_density_op[i-1].dot(logm(avg_density_op[i-1])))
    s2 = -trace(avg_density_op[i].dot(logm(avg_density_op[i])))
    S.append((s2-s1)/dt) 
    s.append(-trace(avg_density_op[i].dot(logm(avg_density_op[i]))))
    time_S = np.delete(time,0)
    
# calculate J
J = []
for i in range(1, len(avg_density_op)):
    J.append((little_omega/T)*((jump_diff[i])-(jump_diff[i-1]))/dt)
    
# calculate total entropy production
sigma = []
for i in range(len(S)):
    sigma.append(np.real(S[i] + J[i]))
    
# calculate state purity
pure = []
for i in range(len(avg_density_op)):  
    pure.append(np.real(trace(avg_density_op[i].dot(avg_density_op[i]))))

# graph 
plt.figure()
plt.plot(time, pure)
plt.xlabel("$t$")
plt.ylim(0,1.2)
plt.ylabel("$Tr(\\rho^{2})$")
plt.title("Purity of state")
plt.show()

# von Neuman entropy
plt.figure()
plt.plot(time_S, s)
plt.xlabel('$t$')
plt.ylabel('$S$')
plt.title('von-Neuman entropy for %s trajectories' %number_of_traj)
plt.show()

# graph
plt.figure()
plt.plot(time_S, S)
plt.title('von-Neuman entropy change for %s trajectories' %number_of_traj)
plt.xlabel('$t$')
plt.ylabel('$\\frac{dS}{dt}$')
plt.show()

# graph
plt.figure()
plt.plot(time_S, J)
plt.title('Entropy flux for %s trajectories' %number_of_traj)
plt.xlabel('$t$')
plt.ylabel('$J$')
plt.show()

# graph
plt.figure()
plt.plot(time_S, sigma)
plt.title('Entropy production rate for %s trajectories' %number_of_traj)
plt.xlabel('$t$')
plt.ylabel('$\sigma$')
plt.show()
"""