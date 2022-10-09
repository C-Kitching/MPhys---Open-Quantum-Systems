from qutip import *
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':

    t_start = 0
    t_end = 20
    dt = 0.01
    
    omega = 1  # Rabi Frequency
    nu = 0  # Detuning
    gamma = omega / 6 # rates
    little_omega = 1.602e-19 # rapid oscillations
    k = 1.381e-23 # boltzman constant
    T = 8000 # temperature
    
    traj = 1000 # number of trajectories
    
    N = 1/(np.exp(little_omega/(k*T)) - 1) # bosonic occupation number
    
    H = (nu/2) * sigmaz() + (omega/2) * sigmax() # Hamiltonian
    psi_mc = tensor(fock(2, 1))
    psi_exact = basis(2, 1)
    
    times = np.linspace(t_start, t_end, int((t_end - t_start)/dt))
    time_omega = [t * omega for t in times]
    
    mc = mcsolve(H, psi_mc, time_omega, [np.sqrt(gamma * N) * sigmap(), np.sqrt(gamma * (N+1)) * sigmam()], [sigmap() * sigmam()], ntraj = traj)
    exact = mesolve(H, psi_exact, time_omega, [np.sqrt(gamma * N) * sigmap(), np.sqrt(gamma * (N+1)) * sigmam()], [sigmap() * sigmam()])
    
    plt.figure()
    
    plt.plot(time_omega, exact.expect[0], label = 'Exact Solution')
    plt.plot(time_omega, mc.expect[0], label = '%s trajectories' %traj)
    
    plt.xlabel('$t\Omega$');
    plt.ylabel('$P_{ee}$');
    plt.title('Rotating Frame - Monte Carlo vs Exact Solution')
    plt.legend()
    plt.ylim(0, 1)
    plt.xlim(0,t_end)
    plt.show()




