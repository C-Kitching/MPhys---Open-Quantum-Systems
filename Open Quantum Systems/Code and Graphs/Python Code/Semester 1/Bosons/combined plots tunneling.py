import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.linalg import expm, sinm, cosm

# propogate forward in time
def propagate_forward(t, dt, state):
    
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


j = 1j  # imaginary unit

delta = 1  # Rabi Frequency
epsilon = 0  # Detuning
little_omega = 1.602e-19
eta = np.sqrt(epsilon**2+delta**2)
k = 1.381e-23
T = 5000
alpha = 1/(12*np.pi)

if(epsilon != 0):
    theta = np.arctan(delta/epsilon)
else:
    theta = np.pi/2

# gammas
gamma_0= 4*np.pi*alpha*k*T
#gamma_0 = 0
gamma_eta = 2*np.pi*alpha*eta
#gamma_eta = 0


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
tmax = 100  # start time

# Hamiltonian
H_sys = (eta/2)*(outer(plus, plus) - outer(minus, minus))
H_eff = H_sys - j*(gamma_0/2)*P_0.dot(P_0) - j*(gamma_eta/2)*(1+N)*P_eta.conj().T.dot(P_eta) - j*(gamma_eta/2)*N*P_eta.dot(P_eta.conj().T)

initial_state = plus # set the initial states initial state

multiple_traj_pop = []
time = np.arange(t, tmax, dt)
number_of_traj = 20

for i in range(number_of_traj):
    multiple_traj_pop.append(single_MC(initial_state, t)[0])

# calculate average pop
avg_pop = np.mean(multiple_traj_pop, axis = 0)

# calculate error of average dist
std = np.std(multiple_traj_pop, axis = 0) / np.sqrt(number_of_traj)

# subplot 1
plt.subplot(1,2,1)

# plot errorbar plot
plt.errorbar(time, avg_pop, yerr = std, capsize = 2.5, errorevery = len(time)//20, ecolor = 'm',
             label = '%d trajectories' %number_of_traj, color = 'r')


# EXACT SOLUTION

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# basis
plus_ket = basis(2,0)
minus_ket = basis(2,1)
plus_bra = plus_ket.dag()
minus_bra = minus_ket.dag()

# parameters
nu = 0 # detuning
epsilon = 1.602e-19 # optical transition
delta = 1 # tunneling
eta = np.sqrt(epsilon**2+delta**2)
alpha = 1/12*np.pi # coupling strength
kB = 1.381e-23 #Bolzman constant
T = 5000 #temperature

# derived paramenters
N = 1/(np.exp(epsilon/(kB*T))-1) # occupation numbers
Gamma_0 = 4*np.pi*alpha*kB*T # rate 
Gamma_eta = 2*np.pi*alpha*eta # rate
c = nu/eta # cos(epsilon/eta)
s = delta/eta # sin(delta/eta)

# pauli matricies
sigma_z = c*(plus_ket*plus_bra - minus_ket*minus_bra) - s*(plus_ket*minus_bra + minus_bra*plus_ket)
sigma_x = s*(plus_ket*plus_bra - minus_ket*minus_bra) + c*(plus_ket*minus_bra + minus_ket*plus_bra)

# P matricies
P_0 = c*(plus_ket*plus_bra - minus_ket*minus_bra)
P_eta = s*(minus_ket*plus_bra)
P_eta_dag = P_eta.dag()

# system hamiltonian
Hs = eta/2*(plus_ket*plus_bra - minus_ket*minus_bra)

# lindblad operators
destruction = np.sqrt(Gamma_eta*(1+N))*P_eta
creation = np.sqrt(Gamma_eta*N)*P_eta_dag
neutral = np.sqrt(Gamma_0)*P_0

# initual state
psi0 = basis(2,0) # ground state

#times
t_start = 0
t_end = 100
dt = 0.01
times = np.linspace(t_start, t_end, int((t_end - t_start)/dt)) # time
times_delta = [t * delta for t in times] # scaled time


# solve exactly
result = mesolve(Hs, psi0, times, [neutral, creation, destruction], [sigmap() * sigmam()])

# plot exact result
plt.plot(times_delta, result.expect[0], label = 'Exact Solution')



# graph details
plt.legend()
plt.ylabel('$\overline{P}_{e}$')
plt.xlabel('$\Delta$t')























# propogate forward in time
def propagate_forward(t, dt, state):
    
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
    for i in range(int(tmax/dt)):
      
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
    
    new_times = []
    for i in range(len(record_time)):
        new_times.append(delta*record_time[i])
    
    return(population_e, population_g, new_times)


j = 1j  # imaginary unit

delta = 1  # tunnelling coefficient
epsilon = 0  # Detuning
little_omega = 1.602e-19
eta = np.sqrt(epsilon**2+delta**2)
k = 1.381e-23
T = 5000
alpha = 1/(12*np.pi)

if(epsilon != 0):
    theta = np.arctan(delta/epsilon)
else:
    theta = np.pi/2

# gammas
gamma_0= 4*np.pi*alpha*k*T
#gamma_0 = 0
gamma_eta = 2*np.pi*alpha*eta
#gamma_eta = 0


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
tmax = 100  # start time

# Hamiltonian
H_sys = (eta/2)*(outer(plus, plus) - outer(minus, minus))
H_eff = H_sys - j*(gamma_0/2)*P_0.dot(P_0) - j*(gamma_eta/2)*(1+N)*P_eta.conj().T.dot(P_eta) - j*(gamma_eta/2)*N*P_eta.dot(P_eta.conj().T)


initial_state_1 = plus # set the initial states initial state
initial_state_2 = minus

# Do the MC for a single tragetory
plt.subplot(1,2,2)
(pop_e, pop_g, time) = single_MC(initial_state_1, t)
(pop_e2, popg_2, time2) = single_MC(initial_state_2, t)

# graph details
plt.plot(time, pop_e, 'r--')
plt.plot(time2, pop_e2)
plt.ylabel('${P}_{e}$')
plt.xlabel('$\Delta$t')



# common title
plt.suptitle('Spin-boson model with tunnelling')
plt.tight_layout()
plt.subplots_adjust(top=0.88)



