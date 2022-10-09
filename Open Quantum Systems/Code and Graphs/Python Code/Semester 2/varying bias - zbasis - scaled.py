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
    
    excited_population = []  # blank array to store population numbers
    ground_population = []
    
    # extract the population
    for i in range(len(reduced_density_op_matrices)):
        ground_population.append(((g.conj().T).dot(reduced_density_op_matrices[i])).dot(g)[0][0])
        excited_population.append(((e.conj().T).dot(reduced_density_op_matrices[i])).dot(e)[0][0])
        
    excited_population = np.real(excited_population)
    ground_population = np.real(ground_population)
    
    return excited_population, ground_population, record_time


# calculate effective hamitonian
def effective_hamiltonian(time):
    
    # System Hamiltonian
    H_sys = (detuning(time)/2)*sigma_z + (tunnelling_coeff(time)/2)*sigma_x
    
    # Effective Hamiltonian
    #H_eff = H_sys - j*((gamma(time)/2)*N(time)*sigma_minus.dot(sigma_plus) + (gamma(time)/2)*(N(time)+1)*sigma_plus.dot(sigma_minus))  
    H_eff = H_sys

    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):
       
    return 0.01*detuning(tmax)
    
# calculate time dependent bias
def detuning(time):  
    
    return 20*(time/tmax-(1./2.))

# calculate time dep rate
def gamma(time):
    return tunnelling_coeff(time)/6

# calculate occupation number
def N(time):
    return 1/(np.exp(detuning(time)/(k*T)) - 1)



#### BEGIN MAIN PROGRAM

# Define some constants
j = 1j  # imaginary unit
k = 1 # boltzman constant eV/K
T = 0.1 # temperature 
alpha = 1/(12*np.pi) # coupling strength

# states in z basis
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# Pauli matrices
sigma_x = outer(e,g) + outer(g,e)
sigma_z = outer(e,e) - outer(g,g)

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
sigma_plus = outer(e,g)
sigma_minus = outer(g,e)
I = np.array([[1,0], [0,1]])

t = 0  # end time 
tmax = 20000 # start 
dt = 0.1  # stepsize

#time = np.arange(t, tmax, dt)
number_of_traj = 1

initial_state = g

multiple_traj_excited_pop = []
multiple_traj_ground_pop = []
time = []

# loop over all trajectories
for i in range(number_of_traj):
    
    # do the monte carlo for each traj
    temp_excited, temp_ground, temp_time = single_MC(initial_state, t)
    
    # append to mastere arrays
    multiple_traj_excited_pop.append(temp_excited)
    multiple_traj_ground_pop.append(temp_ground)
    
    # get the time array on first pass
    if(i == 0): time = temp_time
    
# get bias and tunnelling coeff data
detuning_data = []
tunnelling_data = []
for i in range(len(time)):
    detuning_data.append(detuning(time[i]))
    tunnelling_data.append(tunnelling_coeff(time[i]))

# scale the time to be between 0 and 1
for i in range(len(time)):
    time[i] = time[i]/tmax

# calculate average pop
avg_excited_pop = np.mean(multiple_traj_excited_pop, axis = 0)
avg_ground_pop = np.mean(multiple_traj_ground_pop, axis = 0)

# calculate error of average dist
std_excited = np.std(multiple_traj_excited_pop, axis = 0) / np.sqrt(number_of_traj)
std_ground = np.std(multiple_traj_ground_pop, axis = 0) / np.sqrt(number_of_traj)

# Population plots
plt.figure()
# plot errorbar plot
plt.plot(time, avg_excited_pop, label = '$\\rho_{ee}$')
plt.plot(time, avg_ground_pop,  label = '$\\rho_{gg}$')
plt.title('Slow time-dependent bias in z-basis \nfor %d trajectories with no environment' %number_of_traj)
plt.ylabel('$\overline{P}$', fontsize = 14)
plt.xlabel('$\\frac{t}{t_{max}}$', fontsize = 14)   
plt.legend()

# plot epsilon and delta data
plt.figure()
plt.plot(time, detuning_data, label = r'$\epsilon(t)$')
plt.plot(time, tunnelling_data, label = r'$\Delta$')
plt.title('Tunneling Coefficient and Detuning', fontsize = 16)
plt.ylabel('Energy scale', fontsize = 14)
plt.xlabel('$\\frac{t}{t_{max}}$', fontsize = 14)
plt.legend() 


time = time[90000:110000]
avg_excited_pop = avg_excited_pop[90000:110000]
avg_ground_pop = avg_ground_pop[90000:110000]




# Population plots
plt.figure(figsize = (9,5))
plt.subplot(1,2,2)
plt.plot(time, avg_excited_pop, label = r'$\rho_{\downarrow\hspace{-0.5}\downarrow}$')
plt.plot(time, avg_ground_pop, label = r'$\rho_{\uparrow\hspace{-0.5}\uparrow}$')
plt.title('Adiabatic regime', fontsize = 14)
plt.legend()
plt.ylabel('Probability', fontsize = 14)
plt.xlabel('$\\frac{t}{t_{max}}$', fontsize = 14)  

t = 0  # end time 
tmax = 0.2 # start 
dt = 0.01  # stepsize

number_of_traj = 1

initial_state = g
    
multiple_traj_excited_pop = []  
multiple_traj_ground_pop = []
time = []

# loop over all trajectories
for i in range(number_of_traj):
    
    # do the monte carlo for each traj
    temp_excited, temp_ground, temp_time = single_MC(initial_state, t)
    
    # append to mastere arrays
    multiple_traj_excited_pop.append(temp_excited)
    multiple_traj_ground_pop.append(temp_ground)
    
    # get the time array on first pass
    if(i == 0): time = temp_time
    
    
# get bias and tunnelling coeff data
detuning_data = []
tunnelling_data = []
for i in range(len(time)):
    detuning_data.append(detuning(time[i]))
    tunnelling_data.append(tunnelling_coeff(time[i]))
    
# scale the time to be between 0 and 1
for i in range(len(time)):
    time[i] = time[i]/tmax

# calculate average pop
avg_excited_pop = np.mean(multiple_traj_excited_pop, axis = 0)
avg_ground_pop = np.mean(multiple_traj_ground_pop, axis = 0) 

plt.subplot(1,2,1)
plt.plot(time, avg_excited_pop, label = '$\\rho_{++}$')
plt.plot(time, avg_ground_pop, label = '$\\rho_{--}$')
plt.title('Diabatic regime', fontsize = 14)
plt.ylabel('Probability', fontsize = 14)
plt.xlabel('$\\frac{t}{t_{max}}$', fontsize = 14) 

# common title
plt.suptitle('Diabatic state populations with no environement', fontsize = 16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)

























    