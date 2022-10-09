import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# gaussian curve
def gaussian(x, mu, std):
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-(1/2)*((x - mu)/std)**2)

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
    

_, p_value = stats.jarque_bera(counts)

_, bin_borders, _ = plt.hist(counts, align = 'mid', rwidth = 0.9, density = True, label = 'p_value = %.2f' %p_value, bins = 20) # plot histogram

# get mean and std
mean_count = np.mean(counts)
std_count = np.std(counts)

# plot fitted curve
x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, mean_count, std_count))


plt.legend()
plt.xlabel('Number of Jumps')
plt.ylabel('Normalised Frequency')
plt.title('Quantum Jump Occurance in %d jumps, dt = %s, t = %s' %(number_of_traj, dt, tmax))
plt.show()













