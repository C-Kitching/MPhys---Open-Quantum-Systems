import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# basis
plus_ket = basis(2,0)
minus_ket = basis(2,1)
plus_bra = plus_ket.dag()
minus_bra = minus_ket.dag()

# parameters
nu = 0 # detuning
epsilon = 1.602e-19 # optical transition
delta = 1 # tunneling
eta = np.sqrt(epsilon**2+delta**2)
alpha = 1/(12*np.pi) # coupling strength
kB = 1.381e-23 #Bolzman constant
T = 5000 #temperature

# derived paramenters
N = 1/(np.exp(epsilon/(kB*T))-1) # occupation numbers
Gamma_0 = 4*np.pi*alpha*kB*T # rate 
Gamma_eta = 2*np.pi*alpha*eta # rate
c = nu/eta # cos(epsilon/eta)
s = delta/eta # sin(delta/eta)

# pauli matricies
sigma_z = c*(plus_ket*plus_bra - minus_ket*minus_bra) - s*(plus_ket*minus_bra + minus_bra*plus_ket)
sigma_x = s*(plus_ket*plus_bra - minus_ket*minus_bra) + c*(plus_ket*minus_bra + minus_ket*plus_bra)

# P matricies
P_0 = c*(plus_ket*plus_bra - minus_ket*minus_bra)
P_eta = s*(minus_ket*plus_bra)
P_eta_dag = P_eta.dag()

# system hamiltonian
Hs = eta/2*(plus_ket*plus_bra - minus_ket*minus_bra)

# lindblad operators
destruction = np.sqrt(Gamma_eta*(1+N))*P_eta
creation = np.sqrt(Gamma_eta*N)*P_eta_dag
neutral = np.sqrt(Gamma_0)*P_0

# initual state
psi0 = basis(2,0) # ground state

#times
t_start = 0
t_end = 100
dt = 0.01
times = np.linspace(t_start, t_end, int((t_end - t_start)/dt)) # time
times_delta = [t * delta for t in times] # scaled time

# solve exactly
result = mesolve(Hs, psi0, times, [neutral, creation, destruction], [sigmap() * sigmam()])

# plot exact result
plt.plot(times_delta, result.expect[0], label = 'Exact Solution')










