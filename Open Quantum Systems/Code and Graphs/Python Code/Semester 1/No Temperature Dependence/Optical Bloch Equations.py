import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import fsolve

# propogate forward in time
def propagate_forward(t, dt, state):
    
    random_one = np.random.uniform(0,1)
    dp = dt*(state.conj().T.dot(j*(H_eff-H_eff.conj().T)).dot(state))

    # no jump
    if(random_one > dp):
        no_jump_prefactor = (I-j*dt*H_eff)/(np.sqrt(1-dp))
        new_state = no_jump_prefactor.dot(state)
    # jump
    elif(random_one < dp):
        jump_prefactor = sigma_minus/(np.sqrt(dp/dt))
        new_state = jump_prefactor.dot(state)

    return(new_state, t+dt)

# calculate outer product
def outer(state):
  return(state.dot(state.conj().T))

# calculate inner product
def inner(state):
  return((state.conj().T).dot(state))

j = 1j  # imaginary unit

omega = 1  # Rabi Frequency
delta = 0  # Detuning
gamma = omega / 6

# Pauli matricies
sigma_plus = np.array([[0,1],[0,0]])
sigma_minus = np.array([[0,0],[1,0]])
sigma_x = np.array([[0,1],[1,0]])
I = np.array([[1,0], [0,1]])

dt = 0.1  # stepsize
t = 0  # end time 
tmax = 20  # start time

initial_state = np.array([[0],[1]])  # initial state

record_states = []  # blank array to record states
record_time = []  # blank array to record time

# record initial values
record_states.append(initial_state)
record_time.append(t)

# Hamiltonians
H_opt = -(omega/2)*sigma_x - delta*sigma_plus.dot(sigma_minus)
H_eff = H_opt - j*(gamma/2)*sigma_plus.dot(sigma_minus)

state = initial_state

# propgate forward in time
for i in range(int(tmax/dt)):
  
    # get new state and time
    (state, t) = propagate_forward(t, dt, state)
    
    # normalise the states
    normalised_state = (1/np.linalg.norm(state))*state

    # record states
    record_states.append(normalised_state)
    record_time.append(t)

reduced_density_op_matrices = []  # blank array to store density ops at each step

# calculate the density ops at each step
for i in range(len(record_states)):
    reduced_density_op_matrices.append(outer(record_states[i]))

population = []  # blank array to store population numbers

# extract the population
for i in range(len(reduced_density_op_matrices)):
    population.append((reduced_density_op_matrices[i])[0][0])

record_time_omega = [dt * omega for dt in record_time]
population = np.real(population)

plt.plot(record_time_omega, population)
plt.xlabel('$t\Omega$')
plt.ylabel('$<P_{ee}>$')
plt.title('Single Quantum Trajectory, dt = %s' %dt)
plt.show()