import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import fsolve

j = 1j  # imaginary unit

omega = 1  # Rabi Frequency
delta = 1  # Detuning
gamma = omega / 6
epsilon = 1
omega_l = 1
N = 1

# Pauli matricies
sigma_plus = np.array([[0,1],[0,0]])
sigma_minus = np.array([[0,0],[1,0]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0, -j], [-j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
I = np.array([[1,0], [0,1]])

state = np.array([[0],[1]])  # initial state

dt = 0.1  # stepsize
t = 0  # end time 
tmax = 20  # start time

# Hamiltonians
H_eff = (epsilon/2)*sigma_z + (omega/2)*sigma_x - j*(gamma*N*sigma_minus.dot(sigma_plus) + gamma*(N+1)*sigma_plus.dot(sigma_minus)) 

random = np.random.uniform(0, 1)
func = lambda tau: random - np.float(np.real((state.conj().T).dot(expm(tau*j*(H_eff.conj().T - H_eff)).dot(state))[0]))
tau = np.arange(0, tmax, dt)
x = []
for i in range(len(tau)):
    x.append(func(tau[i]))
        
plt.plot(tau, x)
plt.hlines(0, t, tmax, linestyle = 'dashed', color = 'red')

tau_initial_guess = 1
tau_solution = fsolve(func, tau_initial_guess)

plt.plot(tau_initial_guess, func(tau_initial_guess), 'bo')
plt.plot(tau_solution, func(tau_solution), 'ro')

plt.xlabel('$t_{1}$')
plt.ylabel('Equation to Solve')
plt.title('Numerically Calculating Roots')













