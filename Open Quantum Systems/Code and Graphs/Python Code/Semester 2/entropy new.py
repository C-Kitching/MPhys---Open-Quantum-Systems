import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import expm, sinm, cosm, eig, logm
import sys
import timeit
from multiprocessing import Pool, cpu_count

# propogate forward in time
def propagate_forward(t, dt, state, down_jumps, up_jumps):
    
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
    

def single_MC(initial_state, t):
    
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
        
    return(reduced_density_op_matrices, record_time, record_down_jumps, record_up_jumps)
    

j = 1j  # imaginary unit

delta = 1  
epsilon = 0  # Detuning
gamma = delta / 6
little_omega = 1
eta = np.sqrt(epsilon**2+delta**2)
k = 8.617e-5 # boltzman constant eV/K
T = 4000
alpha = 1/(12*np.pi)


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

dt = 0.1  # stepsize
t = 0  # end time 
tmax = 20  # start time

# Hamiltonian
H_sys = (epsilon/2)*sigma_z + (delta/2)*sigma_x
H_eff = H_sys - j*gamma*N*sigma_minus.dot(sigma_plus) - j*gamma*(1 + N)*sigma_plus.dot(sigma_minus)
#H_eff = H_sys

# set the initial states initial state
initial_state = g 

# traj number
number_of_traj = 1000

# record result of each traj
master_density_op = []
master_down_jumps = []
master_up_jumps = []

start = timeit.default_timer() # start timer

# calculate on each traj
for i in range(number_of_traj):
    t = 0 # reset time 
 
    density_op, time, down_jumps, up_jumps = single_MC(initial_state, t) # calculate on single traj
    
    # append to master arrays 
    master_density_op.append(density_op)
    master_down_jumps.append(down_jumps)
    master_up_jumps.append(up_jumps)
    
    
# calculate averages
avg_density_op = np.mean(master_density_op, axis = 0)
avg_up_jumps = np.mean(master_up_jumps, axis = 0)
avg_down_jumps = np.mean(master_down_jumps, axis = 0)
jump_diff = avg_down_jumps - avg_up_jumps


stop = timeit.default_timer() # stope the timer

print('Time: ', stop - start) 

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














