import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.linalg import expm, sinm, cosm
from math import log10, floor

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
        
        dp_plus = (state.conj().T).dot(sigma_plus(t).dot(sigma_minus(t).dot(state)))
        dp_minus = (state.conj().T).dot(sigma_minus(t).dot(sigma_plus(t).dot(state)))
        
        dp_plus_normalised = dp_plus / (dp_plus + dp_minus)
        dp_minus_normalised = dp_minus / (dp_plus + dp_minus)
        
        # jump up
        if(random_two >= dp_plus_normalised):
            jump_up_prefactor = sigma_plus(t)/np.sqrt(dp_minus/dt)
            new_state = jump_up_prefactor.dot(state)
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus(t)/np.sqrt(dp_plus/dt)
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
    
    population_plus = []  # blank array to store population numbers
    population_minus = []
    
    # extract the population
    for i in range(len(reduced_density_op_matrices)):
        population_minus.append(((minus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(minus(record_time[i]))[0][0])
        population_plus.append(((plus(record_time[i]).conj().T).dot(reduced_density_op_matrices[i])).dot(plus(record_time[i]))[0][0])
        
    population_plus = np.real(population_plus)
    population_minus = np.real(population_minus)
    
    return population_minus, population_plus, record_time


# calculate effective hamitonian
def effective_hamiltonian(time):
    
    # System Hamiltonian
    H_sys = (eta(time)/2)*(outer(plus(time), plus(time)) - outer(minus(time), minus(time)))
    
    # Effective Hamiltonian
    #H_eff = H_sys - j*(gamma_0/2)*P_0(time).dot(P_0(time)) - j*(gamma_eta(time)/2)*(1+N)*P_eta(time).conj().T.dot(P_eta(time)) - j*(gamma_eta(time)/2)*N*P_eta(time).dot(P_eta(time).conj().T)
    H_eff = H_sys

    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):
       
    return 0.01*detuning(tmax)
    
# calculate time dependent bias
def detuning(time):  
    
    return 20*(time/tmax-(1./2.))

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
        return np.arctan2(tunnelling_coeff(time),detuning(time))

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

# spin states
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# initial state
tmax = 1 # place holder
initial_state = minus(0)

# time data
tmax_data = 2* np.logspace(-2,5, 10000)
tmax_data = np.linspace(0.02, 20000, 10000)

# ramp gradient
ramp = []
for i in range(len(tmax_data)):
    tmax = tmax_data[i]
    ramp.append((detuning(tmax)-detuning(0))/tmax)

# LZ calculations
LZ_gamma = [tunnelling_coeff(tmax_data[i])**2/abs(ramp[i]) for i in range(len(tmax_data))]
LZ_prob = [1-np.exp(-np.pi*LZ_gamma[i]/2) for i in range(len(tmax_data))]

# final states
states = []
for i in range(len(LZ_prob)):
    state = [LZ_prob[i], 1-LZ_prob[i]]
    states.append(np.array(state))
    

# final hamiltonian
H_final = effective_hamiltonian(tmax_data[0])

# initial Hamiltoanian
H_initial = effective_hamiltonian(0)

# calculate work
work_data = []
for i in range(len(states)):
    work = (states[i].conj().T).dot(H_final.dot(states[i]))-(initial_state.conj().T).dot(H_initial.dot(initial_state))
    work_data.append(work[0][0])
    
plt.figure(figsize = (9,5)) 
    
plt.subplot(1,2,1)
plt.plot(np.log10(tmax_data), work_data, label = 'Work on system')
plt.xlabel(r'$log(t_{max})$', fontsize = 14)
plt.legend()
plt.ylabel('Energy scale', fontsize = 14)
plt.title('Work done', fontsize = 14)
    

# get spin states
LZ_prob_spin = [np.exp(-np.pi*LZ_gamma[i]/2) for i in range(len(tmax_data))]
up_spin = []
down_spin = []
for i in range(len(LZ_prob_spin)):
    up_spin.append(LZ_prob_spin[i])
    down_spin.append(1-LZ_prob_spin[i])
    
# spin figure
plt.subplot(1,2,2)
plt.plot(np.log10(tmax_data), up_spin, label = r'$\rho_{\uparrow \hspace{-0.5} \uparrow}$')
plt.plot(np.log10(tmax_data), down_spin, label = r'$\rho_{\downarrow \hspace{-0.5} \downarrow}$')
plt.xlabel('$log(t_{max})$', fontsize = 14)
plt.ylabel('Probability', fontsize = 14)
plt.title('Diabatic state populatins', fontsize = 14)
plt.legend()

# common title
plt.suptitle('Effects of ramp speed with no envrironment', fontsize = 16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)



















