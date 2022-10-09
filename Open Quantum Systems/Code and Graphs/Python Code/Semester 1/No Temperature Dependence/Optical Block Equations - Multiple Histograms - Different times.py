import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats

def gaussian(x, mu, std):
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-(1/2)*((x - mu)/std)**2)

# propogate forward in time
def propagate_forward(state, count):
    
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

def count_jumps(initial_state, tmax):
     
    count = 0 # count the number of jumps
    
    state = initial_state
    
    # propgate forward in time
    for i in range(int(tmax/dt)):
      
        # get new state and time
        (state, count) = propagate_forward(state, count)
    
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

tmax = np.arange(100, 500, 100)  # end time
dt = 0.1

# Hamiltonians
H_opt = -(omega/2)*sigma_x - delta*sigma_plus.dot(sigma_minus)
H_eff = H_opt - j*(gamma/2)*sigma_plus.dot(sigma_minus)

initial_state = np.array([[0],[1]])  # initial state

number_of_traj = 1000

master_counts = [] # track counts for each time

# loop over all times
for k in range(len(tmax)):
    
    counts = [] # store counts of each trajectory
    
    for i in range(number_of_traj):
        counts.append(count_jumps(initial_state, tmax[k]))

    master_counts.append(counts)    

    
jarque_bera = [] # record jarque bera p values

plt.figure() # multiple histogram figure
    
# plot histogram for each time
for l in range(len(master_counts)):
    
    # calulcate and record jarque bera p value
    _ , p_value = stats.jarque_bera(master_counts[l])
    jarque_bera.append(p_value)
    
    # plot histogram
    _, bin_borders, _ = plt.hist(master_counts[l], bins = 20, align = 'mid', rwidth = 0.9, density = True, alpha = 0.5, 
             label = 't = %d, p_value = %.2f' %(tmax[l], p_value))
    
    # get mean and std
    mean_count = np.mean(master_counts[l])
    std_count = np.std(master_counts[l])
    
    # plot fitted curve
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, mean_count, std_count))
     
    
    
plt.xlabel('Number of Jumps')
plt.ylabel('Normalised Frequency')
plt.legend()
plt.title('Quantum Jump Occurance in %d jumps for different times, dt = %s' %(number_of_traj, dt))
plt.show()


















