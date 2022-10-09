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
        population_g.append(((minus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(minus(record_time[i]))[0][0])
        population_e.append(((plus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(plus(record_time[i]))[0][0])
    
    population_e = np.real(population_e)
    population_g = np.real(population_g)
    
    return(population_e, population_g, record_time)


# calculate effective hamitonian
def effective_hamiltonian(time):
    
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))
    
    # Effective Hamiltonian
    H_eff = H_sys - j*(gamma_0/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time)/2)*(1+N)*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time)/2)*N*P_eta(time).dot(P_eta(time).conj().T)

    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time): 
    
    if(trigger == 0):
        return t-10
    if(trigger == 1):
        return 0.5*t-5
    if(trigger == 2):
        return 0.1*t-1
    
# calculate time depedent detuning
def detuning(time):
    
    return 0
    
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
    if(detuning(time) == 0):
        return np.pi/2
    else:
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
number_of_traj = 100

# Test different time dependence
master_population = [] # master array to hold 2D arrays for different time depedence

# loop to investigate time dependence
for i in range(3):
    
    trigger = i  # to change time dependence of tunnelling coeff in function
    initial_state = np.cos(theta(time[0])/2)*g-np.sin(theta(time[0])/2)*e # set the initial state
    
    multiple_traj_pop = []  # store population array for all trajectories

    for i in range(number_of_traj):
        multiple_traj_pop.append(single_MC(initial_state, t)[0])

    master_population.append(multiple_traj_pop)


labels = ['$t-10$','$0.1t-1$', '$0.01t-0.1$']

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
#### Exact Solution

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# time dependent coefficient of hamiltonian - eta(t)
def Hs_coeff(t, args):
    return t-10

# time dep destruction coefficient
def des_coeff(t, args):
    return np.sqrt(gamma_eta(t)*(1+N))*np.sin(theta(t))
    
# time indep creation coefficient
def cre_coeff(t, args):
    return np.sqrt(gamma_eta(t)*N)*np.sin(theta(t))

# time indep neutral coefficient
def neu_coeff(t, args):
    return np.cos(theta(t))
    
    
# basis
plus_ket = basis(2,0)
minus_ket = basis(2,1)
plus_bra = plus_ket.dag()
minus_bra = minus_ket.dag()

# parameters
epsilon = 1 # detuning
little_omega = 1.602e-19 # optical transition
alpha = 1/12*np.pi # coupling strength
kB = 1.381e-23 #Bolzman constant
T = 5000 #temperature

# derived paramenters
N = 1/(np.exp(little_omega/(kB*T))-1) # occupation numbers
Gamma_0 = 4*np.pi*alpha*kB*T # time indep rate

# system hamiltonian - time indep
Hs = (plus_ket*plus_bra - minus_ket*minus_bra)

# lindblad operators - time indep
des_op = (minus_ket*plus_bra)
cre_op = (plus_ket*minus_bra)
neu_op = (plus_ket*plus_bra - minus_ket*minus_bra)

# initual state
psi0 = basis(2,0) # ground state


#times
t_start = 0
t_end = 20
dt = 0.01
times = np.linspace(t_start, t_end, int((t_end - t_start)/dt)) # time

H = [[Hs, Hs_coeff]] # time dep system hamiltonian
c_ops = [[des_op, des_coeff],[cre_op, cre_coeff],[neu_op,neu_coeff]]

# solve exactly
result = mesolve(H, psi0, times, c_ops, [sigmap() * sigmam()])

# plot exact result
plt.plot(times, result.expect[0], label = 'Exact Solution')
"""




# Graph details
plt.title('Time-dependent Bias %d trajectories' %number_of_traj)
plt.legend()
plt.ylabel('$\overline{P}_{e}$')
plt.xlabel('$\Delta$t')





























