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
        
        dp_plus = (state.conj().T).dot(sigma_plus.dot(sigma_minus.dot(state)))
        dp_minus = (state.conj().T).dot(sigma_minus.dot(sigma_plus.dot(state)))
        
        dp_plus_normalised = dp_plus / (dp_plus + dp_minus)
        dp_minus_normalised = dp_minus / (dp_plus + dp_minus)
        
        # jump up
        if(random_two >= dp_plus_normalised):
            jump_up_prefactor = sigma_plus/np.sqrt(dp_minus/dt)
            new_state = jump_up_prefactor.dot(state)
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus/np.sqrt(dp_plus/dt)
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
        record_time.append(t)
        
        # set the state to the normalised state
        state = normalised_state
    
    reduced_density_op_matrices = []  # blank array to store density ops at each step
    
    # calculate the density ops at each step
    for i in range(len(record_states)):
        reduced_density_op_matrices.append(outer(record_states[i], record_states[i]))
    
    population_e = []  # blank array to store population numbers
    population_g = []
    
    # extract the population
    for i in range(len(reduced_density_op_matrices)):
        population_g.append(((minus.conj().T).dot(reduced_density_op_matrices[i])).dot(minus)[0][0])
        population_e.append(((plus.conj().T).dot(reduced_density_op_matrices[i])).dot(plus)[0][0])
    
    population_e = np.real(population_e)
    population_g = np.real(population_g)
    
    return(population_e, population_g, record_time)


# calculate effective hamitonian
def effective_hamiltonian(time):
    
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus, plus) - outer(minus, minus))
    
    # Effective Hamiltonian
    H_eff = H_sys - j*(gamma_0/2)*P_0.dot(P_0) - j*(gamma_eta(time)/2)*(1+N)*P_eta.conj().T.dot(P_eta) - j*(gamma_eta(time)/2)*N*P_eta.dot(P_eta.conj().T)

    return(H_eff)


# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):   
    
    if(trigger == 0):
        return 1 + time
    if(trigger == 1):
        return 1 + time**2
    if(trigger == 2):
        return 1 + time**(1/2)


# calculate eta
def eta(time):
    return(np.sqrt(tunnelling_coeff(time)**2+epsilon**2))


# calculate time dep rate
def gamma_eta(time):
    return 2*np.pi*alpha*eta(time)


#### BEGIN MAIN PROGRAM

# Define some constants
j = 1j  # imaginary unit
epsilon = 0  # Detuning
little_omega = 1.602e-19
k = 1.381e-23 # boltzman constant
T = 5000 # temperature 
alpha = 1/(12*np.pi) # coupling strength

# Calculate theta
if(epsilon != 0):
    theta = np.arctan(delta/epsilon)
else:
    theta = np.pi/2

# Rates
gamma_0= 4*np.pi*alpha*k*T

# states in z basis
e = np.array([[1],[0]])
g = np.array([[0],[1]])

#states in +/- basis
plus = np.sin(theta/2)*g+np.cos(theta/2)*e
minus = np.cos(theta/2)*g-np.sin(theta/2)*e

# P ops
P_0 = np.cos(theta)*(outer(plus, plus) - outer(minus, minus))
P_eta = np.sin(theta)*(outer(minus, plus))

# bosonic occupation number
N = 1/(np.exp(little_omega/(k*T)) - 1)

# Pauli matricies
sigma_plus = outer(plus, minus)
sigma_minus = outer(minus, plus)
sigma_x = outer(plus, minus) + outer(minus, plus)
sigma_y = np.array([[0, -j], [-j, 0]])
sigma_z = outer(plus, plus) - outer(minus, minus)
I = np.array([[1,0], [0,1]])

dt = 0.01  # stepsize
t = 0  # end time 
tmax = 20  # start time

initial_state = plus # set the initial states initial state

time = np.arange(t, tmax, dt)
number_of_traj = 1000

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

# Graph details
plt.title('Time-dependent Tunnelling Coefficient %d trajectories' %number_of_traj)
plt.legend()
plt.ylabel('$\overline{P}_{e}$')
plt.xlabel('$\Delta$t')












