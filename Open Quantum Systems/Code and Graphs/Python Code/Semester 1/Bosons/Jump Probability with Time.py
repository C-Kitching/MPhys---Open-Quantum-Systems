import numpy as np
import matplotlib.pyplot as plt

# propogate forward in time
def propagate_forward(t, dt, state):
    
    random_one = np.random.uniform(0,1)
    
    #dp = dt*(state.conj().T.dot(j*(H_eff-H_eff.conj().T)).dot(state))
    
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

    return(new_state, dp, t+dt)

# calculate outer product
def outer(state):
  return(np.outer(state, state.conj()))

# calculate inner product
def inner(state):
  return(np.vdot(state, state))

j = 1j  # imaginary unit

omega = 1  # Rabi Frequency
delta = 0  # Detuning
gamma = omega / 6

# Pauli matricies
sigma_plus = np.array([[0,1],[0,0]])
sigma_minus = np.array([[0,0],[1,0]])
sigma_x = np.array([[0,1],[1,0]])
I = np.array([[1,0], [0,1]])

dt = 0.001  # stepsize
t = 0  # end time 
tmax = 20  # start time

initial_state = np.array([[0],[1]])  # initial state

record_prob = [] # record jump probability
record_time = []  # blank array to record time

# record initial values

record_time.append(t)

# Hamiltonians
H_opt = -(omega/2)*sigma_x - delta*sigma_plus.dot(sigma_minus)
H_eff = H_opt - j*(gamma/2)*sigma_plus.dot(sigma_minus)

initial_prob = dt*(initial_state.conj().T.dot(j*(H_eff-H_eff.conj().T)).dot(initial_state)) # initial prob
record_prob.append(np.float(np.real(initial_prob)))

normalised_state = initial_state


# propgate forward in time
for i in range(int(tmax/dt)):
  
    # get new state and time
    (state, p, t) = propagate_forward(t, dt, normalised_state)
    
    
    # normalise the states
    normalised_state = (1/np.linalg.norm(state))*state

    # record time
    record_time.append(t)
    record_prob.append(np.float(np.real(p)))
    
record_time_omega = np.array([dt * omega for dt in record_time])

plt.plot(record_time_omega, record_prob)
plt.title("Single Trajectory - Probability to Quantum Jump, dt = %s" %dt)
plt.xlabel('$t\Omega$')
plt.ylabel('Jump Probability')
plt.show()












