import numpy as np
import matplotlib.pyplot as plt

# calculate outer product
def outer(state):
  return(np.outer(state, state.conj()))

# calculate inner product
def inner(state):
  return(np.vdot(state, state))

# propogate forward in time
def propagate_forward(state):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*H_eff*dt).dot(state) # phi(t+dt)
    dp = 1 - inner(state_1) # prob to jump

    # no jump
    if(random_one > dp):
        no_jump_prefactor = (I-j*dt*H_eff)/(np.sqrt(1-dp))
        new_state = no_jump_prefactor.dot(state)
    # jump
    elif(random_one < dp):
        jump_prefactor = sigma_minus/(np.sqrt(dp/dt))
        new_state = jump_prefactor.dot(state)

    return(new_state)

def single_MC(initial_state):
    
    record_states = []  # blank array to record states
    
    # record initial values
    record_states.append(initial_state)
    
    state = initial_state
    
    # propgate forward in time
    for i in range(int(tmax/dt) - 1):
      
        # get new state and time
        state = propagate_forward(state)
        
        # normalise the states
        normalised_state = (1/np.linalg.norm(state))*state
    
        # record states
        record_states.append(normalised_state)
    
    reduced_density_op_matrices = []  # blank array to store density ops at each step
    
    # calculate the density ops at each step
    for i in range(len(record_states)):
        reduced_density_op_matrices.append(outer(record_states[i]))
    
    population = []  # blank array to store population numbers
    
    # extract the population
    for i in range(len(reduced_density_op_matrices)):
        population.append((reduced_density_op_matrices[i])[0][0])
    population = np.real(population)
    
    return(population)


j = 1j  # imaginary unit

omega = 1  # Rabi Frequency
delta = 0  # Detuning
gamma = omega / 6

# Pauli matricies
sigma_plus = np.array([[0,1],[0,0]])
sigma_minus = np.array([[0,0],[1,0]])
sigma_x = np.array([[0,1],[1,0]])
I = np.array([[1,0], [0,1]])

dt = 0.01  # stepsize
t = 0  # end time 
tmax = 20  # start time

# Hamiltonians
H_opt = -(omega/2)*sigma_x - delta*sigma_plus.dot(sigma_minus)
H_eff = H_opt - j*(gamma/2)*sigma_plus.dot(sigma_minus)

initial_state = np.array([[0],[1]])  # initial state

number_of_traj = 1000  # number of trajectories 

time = np.arange(0, tmax, dt)  # time

multiple_traj_pop = []  # record populations of each trajectory - 2D array

# get populations for each trajectory
for i in range(number_of_traj):
    multiple_traj_pop.append(single_MC(initial_state))
    
# calculate average pop
avg_pop = np.mean(multiple_traj_pop, axis = 0)

# calculate error of average dist
std = np.std(multiple_traj_pop, axis = 0) / np.sqrt(number_of_traj)

time_omega = np.array([omega * t for t in time]) # adjusted time

# plot errorbar plot
plt.errorbar(time_omega, avg_pop, yerr = std, capsize = 2.5, errorevery = len(time_omega)//20, ecolor = 'm',
             label = '%d trajectories' %number_of_traj, color = 'r')

# Exact Solution
t = time_omega
p_ee = (1/8395)*(36*np.exp(-t/8)*(115*np.exp(t/8)-115*np.cos((5*np.sqrt(23)/24)*t)-3*np.sqrt(23)*np.sin((5*np.sqrt(23)/24)*t)))

plt.plot(time_omega, p_ee, 'b--', label = 'Exact Solution') # plot exact solution

# graph details
plt.title('Average of %s MC Quantum Trajectories, dt = %s' %(number_of_traj,dt))
plt.ylim(0,1)
plt.legend()
plt.xlabel('$t\Omega$')
plt.ylabel('$<P_{e}>$')
plt.show()





