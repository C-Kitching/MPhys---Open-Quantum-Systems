import numpy as np
import matplotlib.pyplot as plt

# propogate forward in time
def propagate_forward(dt, state, count):
    
    random_one = np.random.uniform(0,1)
    dp = dt*(state.conj().T.dot(j*(H_eff-H_eff.conj().T)).dot(state))

    # no jump
    if(random_one > dp):
        no_jump_prefactor = (I-j*dt*H_eff)/(np.sqrt(1-dp))
        new_state = no_jump_prefactor.dot(state)
    # jump
    elif(random_one < dp):
        count += 1
        jump_prefactor = sigma_minus/(np.sqrt(dp/dt))
        new_state = jump_prefactor.dot(state)

    return(new_state, count)

# calculate outer product
def outer(state):
  return(state.dot(state.conj().T))

# calculate inner product
def inner(state):
  return((state.conj().T).dot(state))

def count_jumps(initial_state):
     
    count = 0 # count the number of jumps
    
    state = initial_state
    
    # propgate forward in time
    for i in range(int(tmax/dt)):
      
        # get new state and time
        (state, count) = propagate_forward(dt, state, count)
    
    return(count)
    


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
tmax = 500  # end time

# Hamiltonians
H_opt = -(omega/2)*sigma_x - delta*sigma_plus.dot(sigma_minus)
H_eff = H_opt - j*(gamma/2)*sigma_plus.dot(sigma_minus)

initial_state = np.array([[0],[1]])  # initial state

number_of_traj = 1000

counts = [] # track counts for each trajectory

for i in range(number_of_traj):
    counts.append(count_jumps(initial_state))
    
plt.hist(counts, align = 'mid', rwidth = 0.9, bins = 20)
plt.xlabel('Number of Jumps')
plt.ylabel('Frequency')
plt.title('Quantum Jump Occurance in %d jumps, dt = %s, t = %s' %(number_of_traj, dt, tmax))
plt.show()













