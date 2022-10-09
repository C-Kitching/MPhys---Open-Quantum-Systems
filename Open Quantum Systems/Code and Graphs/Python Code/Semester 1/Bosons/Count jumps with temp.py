import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, sinm, cosm


# propogate forward in time
def propagate_forward(state, count_up, count_down, N):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective(N)*dt).dot(state) # phi(t+dt)
    dp = 1 - inner(state_1) # prob to jump

    # no jump
    if(random_one >= dp):
        no_jump_prefactor = (I-j*dt*effective(N))/(np.sqrt(1-dp))
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

def count_jumps(initial_state, N):
     
    count_up = 0 # up jumps
    count_down = 0 # down jumps
    
    state = initial_state
    
    # propgate forward in time
    for i in range(int(tmax/dt)):
      
        # get new state and time
        (state, count_up, count_down) = propagate_forward(state, count_up, count_down, N)
    
    return(count_up, count_down)


#calculate effective hamiltonian
def effective(N):
    return((nu/2)*sigma_z + (omega/2)*sigma_x - j*(gamma*N*sigma_minus.dot(sigma_plus) + gamma*(N+1)*sigma_plus.dot(sigma_minus)))




j = 1j  # imaginary unit

omega = 1  # Rabi Frequency
nu = 0  # Detuning
gamma = omega / 6
little_omega = 1.602e-19
k = 1.381e-23

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

initial_state = np.array([[0],[1]])  # initial state

up_jumps = []
down_jumps = []
temp = np.arange(25, 100, 1)

for i in range(len(temp)):
    T= temp[i]
    N = 1/(np.exp(little_omega/(k*T)) - 1)
    (up_jumps_traj, down_jumps_traj) = count_jumps(initial_state, N)
    
    up_jumps.append(up_jumps_traj)
    down_jumps.append(down_jumps_traj)

plt.figure()
plt.plot(temp, up_jumps, label = 'up jumps')
plt.plot(temp, down_jumps, label = 'down jumps')
plt.legend()
plt.title('Jump frequency with temperature - Optical Bloch Equations')
plt.xlabel('Temperature / K')
plt.ylabel('Number of jumps')
plt.show()

diff = [up_jumps[i] - down_jumps[i] for i in range(len(up_jumps))]
plt.figure()
plt.plot(temp, diff)
