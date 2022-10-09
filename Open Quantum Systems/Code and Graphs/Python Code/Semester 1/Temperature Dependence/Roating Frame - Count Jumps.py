import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, std):
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-(1/2)*((x - mu)/std)**2)

# propogate forward in time
def propagate_forward(state, count_up, count_down):
    
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
            count_up += 1
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus/np.sqrt(dp_plus/dt)
            new_state = jump_down_prefactor.dot(state)
            count_down += 1

    return(new_state, count_up, count_down)

# calculate outer product
def outer(state):
  return(state.dot(state.conj().T))

# calculate inner product
def inner(state):
  return((state.conj().T).dot(state))

def count_jumps(initial_state):
     
    count_up = 0 # up jumps
    count_down = 0 # down jumps
    
    state = initial_state
    
    # propgate forward in time
    for i in range(int(tmax/dt)):
      
        # get new state and time
        (state, count_up, count_down) = propagate_forward(state, count_up, count_down)
    
    return(count_up, count_down)
    

j = 1j  # imaginary unit

omega = 1  # Rabi Frequency
nu = 0  # Detuning
gamma = omega / 6
little_omega = 1.602e-19
k = 1.381e-23
T = 2500000

N = 1/(np.exp(little_omega/(k*T)) - 1)

# Pauli matricies
sigma_plus = np.array([[0,1],[0,0]])
sigma_minus = np.array([[0,0],[1,0]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0, -j], [-j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
I = np.array([[1,0], [0,1]])

dt = 0.01  # stepsize
t = 0  # end time 
tmax = 20  # start time

# Hamiltonians
H_eff = (nu/2)*sigma_z + (omega/2)*sigma_x - j*(gamma*N*sigma_minus.dot(sigma_plus) + gamma*(N+1)*sigma_plus.dot(sigma_minus)) 


initial_state = np.array([[0],[1]])  # initial state

number_of_traj = 1000 # number of trajectories

# track counts for each trajectory
track_counts_up = [] # up jumps
track_counts_down = [] # down jumps

for i in range(number_of_traj):
    (count_up, count_down) = count_jumps(initial_state)
    track_counts_up.append(count_up)
    track_counts_down.append(count_down)
    
_, bin_borders, _ = plt.hist(track_counts_up, color = 'lightsteelblue', align = 'mid', alpha = 0.7, rwidth = 0.9, bins = 20, label = 'Up Jumps', density = True)
_, bin_borders, _ = plt.hist(track_counts_down, color = 'moccasin', align = 'mid', alpha = 0.7, rwidth = 0.9, bins = 20, label = 'Down Jumps', density = True)

# get mean and std
mean_count_up = np.mean(track_counts_up)
std_count_up = np.std(track_counts_up)
mean_count_down = np.mean(track_counts_down)
std_count_down = np.std(track_counts_down)

# plot fitted curve
x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, mean_count_up, std_count_up))
plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, mean_count_down, std_count_down))

# graph details
plt.legend()
plt.xlabel('Number of Jumps')
plt.ylabel('Normalised Frequency')
plt.title('Quantum Jump Occurance for %s trajectories - Driven System, dt = %s, t = %s' %(number_of_traj, dt, tmax))
plt.show()