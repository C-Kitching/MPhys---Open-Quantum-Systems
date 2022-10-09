import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.linalg import expm, sinm, cosm

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
        
        dp_plus = (state.conj().T).dot(sigma_plus(time).dot(sigma_minus(time).dot(state)))
        dp_minus = (state.conj().T).dot(sigma_minus(time).dot(sigma_plus(time).dot(state)))
        
        dp_plus_normalised = dp_plus / (dp_plus + dp_minus)
        dp_minus_normalised = dp_minus / (dp_plus + dp_minus)
        
        # jump up
        if(random_two >= dp_plus_normalised):
            jump_up_prefactor = sigma_plus(time)/np.sqrt(dp_minus/dt)
            new_state = jump_up_prefactor.dot(state)
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus(time)/np.sqrt(dp_plus/dt)
            new_state = jump_down_prefactor.dot(state)

    return(new_state, t+dt)

# calculate outer product
def outer(state1, state2):
  return(state1.dot(state2.conj().T))

# calculate inner product
def inner(state):
  return((state.conj().T).dot(state))

def single_MC(initial_state, t):
    
    record_states = []  # blank array to record states
    record_time = []  # blank array to record time
    
    # record initial values
    record_states.append(initial_state)
    record_time.append(t)
    
    state = initial_state
    
    # propgate forward in time
    for i in range(int(tmax/dt) -1):
                        
        # get new state and time
        (state, t) = propagate_forward(t, dt, state)
        
        # normalise the states
        normalised_state = (1/np.sqrt(inner(state)))*state
    
        # record states
        record_states.append(normalised_state)
        record_time.append(t*detuning(t))
        
        # set the state to the normalised state
        state = normalised_state
    
    reduced_density_op_matrices = []  # blank array to store density ops at each step
    
    # calculate the density ops at each step
    for i in range(len(record_states)):
        reduced_density_op_matrices.append(outer(record_states[i], record_states[i]))
    
    population_plus = []  # blank array to store population numbers
    population_minus = []
    
    # extract the population
    for i in range(len(reduced_density_op_matrices)):
        population_minus.append(((minus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(minus(record_time[i]))[0][0])
        population_plus.append(((plus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(plus(record_time[i]))[0][0])
    
    population_plus = np.real(population_plus)
    population_minus = np.real(population_minus)
    
    return(population_minus, population_plus, record_time)


# calculate effective hamitonian
def effective_hamiltonian(time):
    
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))
    
    # Effective Hamiltonian
    #H_eff = H_sys - j*(gamma_0/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time)/2)*(1+N)*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time)/2)*N*P_eta(time).dot(P_eta(time).conj().T)
    H_eff = H_sys

    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):   
    
    return 1
    
# calculate time dependent bias
def detuning(time):
    
    return (10*time-100)

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
    return np.arctan(tunnelling_coeff(time)/detuning(time))

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


#### BEGIN MAIN PROGRAM

# Define some constants
j = 1j  # imaginary unit
little_omega = 1.602e-19
k = 1.381e-23 # boltzman constant
T = 5000 # temperature 
alpha = 1/(12*np.pi) # coupling strength

# Time indep rate
gamma_0= 4*np.pi*alpha*k*T

# states in z basis
e = np.array([[1],[0]])
g = np.array([[0],[1]])

#states in +/- basis time indep
#plus = np.sin(theta/2)*g+np.cos(theta/2)*e
#minus = np.cos(theta/2)*g-np.sin(theta/2)*e

# bosonic occupation number
N = 1/(np.exp(little_omega/(k*T)) - 1)

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
I = np.array([[1,0], [0,1]])

dt = 0.1  # stepsize
t = 0  # end time 
tmax = 20  # start 

time = np.arange(t, tmax, dt)
number_of_traj = 10

trigger = 0
#initial_state = np.sin(theta(time[0])/2)*g+np.cos(theta(time[0])/2)*e # excited state
initial_state = np.cos(theta(time[0])/2)*g-np.sin(theta(time[0])/2)*e # ground state


# Test different time dependence
"""
master_population = [] # master array to hold 2D arrays for different time depedence

# loop to investigate time dependence
for i in range(3):
    
    trigger = i  # to change time dependence of tunnelling coeff in function
    
    multiple_traj_pop = []  # store population array for all trajectories

    for i in range(number_of_traj):
        multiple_traj_pop.append(single_MC(initial_state, t)[0])

    master_population.append(multiple_traj_pop)

labels = ['$~t$','$~t^{2}$', '~$\sqrt{t}$']

# plot graphs for all time dependence
for i in range(len(master_population)):

    # extract array for each time dependence
    multiple_traj_pop = master_population[i]

    # calculate average pop
    avg_pop = np.mean(multiple_traj_pop, axis = 0)
    
    # calculate error of average dist
    std = np.std(multiple_traj_pop, axis = 0) / np.sqrt(number_of_traj)
    
    # plot errorbar plot
    plt.errorbar(time, avg_pop, yerr = std, capsize = 2.5, errorevery = len(time)//20, ecolor = 'm',
                 label = labels[i])
"""
    
    
    
multiple_traj_excited_pop = []  
multiple_traj_ground_pop = []

# loop over all trajectories
for i in range(number_of_traj):
    multiple_traj_excited_pop.append(single_MC(initial_state, t)[0])
    multiple_traj_ground_pop.append(single_MC(initial_state, t)[1])

# calculate average pop
avg_excited_pop = np.mean(multiple_traj_excited_pop, axis = 0)
avg_ground_pop = np.mean(multiple_traj_ground_pop, axis = 0)

# calculate error of average dist
std_excited = np.std(multiple_traj_excited_pop, axis = 0) / np.sqrt(number_of_traj)
std_ground = np.std(multiple_traj_ground_pop, axis = 0) / np.sqrt(number_of_traj)

# plot errorbar plot
plt.errorbar(time, avg_excited_pop, yerr = std_excited, capsize = 2.5, errorevery = len(time)//20, ecolor = 'm', label = '$\\rho_{++}$')
plt.errorbar(time, avg_ground_pop, yerr = std_ground, capsize = 2.5, errorevery = len(time)//20, ecolor = 'm', label = '$\\rho_{--}$')

# Graph details
plt.title('Time-dependent bias %d trajectories' %number_of_traj)
plt.ylabel('$\overline{P}$')
plt.legend()
plt.xlabel('t')   
    
    
    
    
    
    
    
    
    
    

