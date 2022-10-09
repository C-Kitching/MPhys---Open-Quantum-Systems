import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import expm, sinm, cosm, eig, logm
import sys
import multiprocessing
import timeit
from multiprocessing import Pool, cpu_count
from functools import partial
from numba import jit, cuda
import itertools

# propogate forward in time
def propagate_forward(t, dt, state, jump, tmax, T):
    
    random_one = np.random.uniform(0,1)
    
    state_1 = (I - j*effective_hamiltonian(t, tmax, T)*dt).dot(state) # phi(t+dt)
    
    dp = 1 - inner(state_1) # prob to jump
    
    # no jump
    if(random_one >= dp):
        
        jump = False
        
        no_jump_prefactor = (I-j*dt*effective_hamiltonian(t, tmax, T))/(np.sqrt(1-dp))
        new_state = no_jump_prefactor.dot(state)
        
    # jump
    elif(random_one < dp):
        
        jump = True
        
        random_two = np.random.uniform(0, 1)
        
        dp_plus = (state.conj().T).dot(sigma_plus(t, tmax).dot(sigma_minus(t, tmax).dot(state)))
        dp_minus = (state.conj().T).dot(sigma_minus(t, tmax).dot(sigma_plus(t, tmax).dot(state)))
        
        dp_plus_normalised = dp_plus / (dp_plus + dp_minus)
        dp_minus_normalised = dp_minus / (dp_plus + dp_minus)
        
        # jump up
        if(random_two >= dp_plus_normalised):
            jump_up_prefactor = sigma_plus(t, tmax)/np.sqrt(dp_minus/dt)
            new_state = jump_up_prefactor.dot(state)
            
        # jump down
        else:
            jump_down_prefactor = sigma_minus(t, tmax)/np.sqrt(dp_plus/dt)
            new_state = jump_down_prefactor.dot(state)

    return(new_state, t+dt, jump)

# calculate outer product
def outer(state1, state2):
  return(state1.dot(state2.conj().T))

# calculate inner product
def inner(state):
  return((state.conj().T).dot(state))

# calculate trace of a 2d array
def trace(matrix):
    
    trace = 0
    
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
                if(i == j):
                    trace += matrix[i][j]
                    
    return trace
    

def single_MC(core_traj, tmax, T):
    
    # create master arrays
    master_work = []
    master_heat = []
    master_pop_plus = []
    master_pop_minus = []
    master_spin_up = []
    master_spin_down = []
    
    # set dt
    if(tmax < 0.2):
        dt = tmax/10
    else:
        dt = 0.1
    
    # loop over all trajectories
    for i in range(core_traj):
    
        # reset time and initial state 
        t = 0  # start time    
        initial_state = minus(0, tmax) # set the initial states initial state
                
        old_state = initial_state
        
        # reset heat to zero
        heat = 0
        
        # initialise jump counts to 0
        jump = False
        
        # propgate forward in time
        for i in range(int(tmax/dt)):
          
            # get new state and time
            (new_state, t, jump) = propagate_forward(t, dt, old_state, jump, tmax, T)

            # normalise the states
            normalised_state = (1/np.linalg.norm(new_state))*new_state
            
            # if jump then heat exchanged
            if jump:
                heat += expectation(system_hamiltonian(t, tmax), normalised_state)- expectation(system_hamiltonian(t, tmax), old_state)
            # if no jump then work done
            
            # set the state to the normalised state
            old_state = normalised_state
            
        # calculate energy change over trajectory
        total_energy_change = expectation(system_hamiltonian(tmax, tmax), old_state)- expectation(system_hamiltonian(0, tmax), initial_state)
          
        # calculate work done on traj
        work = total_energy_change - heat
        
        # get final density operator
        density_op = outer(old_state, old_state)
        
        # get populations
        minus_pop = np.real(((minus(tmax, tmax).conj().T).dot(density_op)).dot(minus(tmax, tmax))[0][0])
        plus_pop = np.real(((plus(tmax, tmax).conj().T).dot(density_op)).dot(plus(tmax, tmax))[0][0])
        up_pop = np.real(((e.conj().T).dot(density_op)).dot(e)[0][0])
        down_pop = np.real(((g.conj().T).dot(density_op)).dot(g)[0][0])
                                    
        # add each traj variables to master array
        master_work.append(work)
        master_heat.append(heat)
        master_pop_plus.append(plus_pop)
        master_pop_minus.append(minus_pop)
        master_spin_up.append(up_pop)
        master_spin_down.append(down_pop)
        
    # average over trajectories
    avg_pop_plus = np.mean(master_pop_plus)
    avg_pop_minus = np.mean(master_pop_minus)
    avg_spin_up = np.mean(master_spin_up)
    avg_spin_down = np.mean(master_spin_down)
    avg_work = np.mean(master_work)
    avg_heat = np.mean(master_heat)
        
    return(avg_work, avg_heat, avg_pop_plus, avg_pop_minus, avg_spin_up, avg_spin_down)

# calculate expectation of matrix and state
def expectation(matrix, state):
    return np.real((state.conj().T).dot(matrix.dot(state)))[0][0]


# calculate system hamiltonian
def system_hamiltonian(time, tmax):   
    # System Hamiltonian
    H_sys = (eta(time, tmax)/2)*(outer(plus(time, tmax), plus(time, tmax)) - outer(minus(time, tmax), minus(time, tmax)))
    
    return H_sys
        
# calculate effective hamitonian
def effective_hamiltonian(time, tmax, T):   
    H_eff = system_hamiltonian(time, tmax) - j*(gamma_0(T)/2)*P_0(time, tmax).dot(P_0(time, tmax)) - j*(gamma_eta(time, tmax)/2)*(1+N(time, tmax, T))*P_eta(time, tmax).conj().T.dot(P_eta(time, tmax)) - j*(gamma_eta(time, tmax)/2)*N(time, tmax, T)*P_eta(time, tmax).dot(P_eta(time, tmax).conj().T)
    #H_eff = system_hamiltonian(time, tmax)
    return(H_eff)

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time, tmax):
    return 0.01*detuning(tmax, tmax)
    
# calculate time dependent bias
def detuning(time, tmax):  
    
    return 10*(time/tmax - 0.5)

# calculate time depedent P_0 operator
def P_0(time, tmax):
    return np.cos(theta(time, tmax))*(outer(plus(time, tmax), plus(time, tmax)) - outer(minus(time, tmax), minus(time, tmax)))

# calculate time depedent P_eta operator
def P_eta(time, tmax):
    return np.sin(theta(time, tmax))*(outer(minus(time, tmax), plus(time, tmax)))
    
#calculate time depedent minus state
def minus(time, tmax):
    return np.cos(theta(time, tmax)/2)*g-np.sin(theta(time, tmax)/2)*e
    
# calculate time depedent plus state
def plus(time, tmax):
    return np.sin(theta(time, tmax)/2)*g+np.cos(theta(time, tmax)/2)*e
    
# calculate time dependent theta
def theta(time, tmax):
    if(detuning(time, tmax) == 0):
        return np.pi/2
    else:
        return np.arctan2(tunnelling_coeff(time, tmax),detuning(time, tmax))

# calculate eta
def eta(time, tmax):
    return(np.sqrt(tunnelling_coeff(time, tmax)**2+detuning(time, tmax)**2))

# calculate time dep rate
def gamma_eta(time, tmax):
    return 2*np.pi*alpha*eta(time, tmax)

# calculate time dep sigma+
def sigma_plus(time, tmax):
    return outer(plus(time, tmax),minus(time, tmax))

# calculate time dep sigma-
def sigma_minus(time, tmax):
    return outer(minus(time, tmax),plus(time, tmax))

# calculate time dep sigma_x
def sigma_x(time, tmax):
    return outer(plus(time, tmax), minus(time, tmax)) + outer(minus(time, tmax), plus(time, tmax))

# calculate time dep sigma_z
def sigma_z(time, tmax):
    return outer(plus(time, tmax), plus(time, tmax)) - outer(minus(time, tmax), minus(time, tmax))

# calculate occupation number
def N(time, tmax, T):
    return 1/(np.exp(eta(time, tmax)/(k*T)) - 1)

# time indep rate
def gamma_0(T):
    return 4*np.pi*alpha*k*T



j = 1j  # imaginary unit
little_omega = 1 # oprtical transition
k = 1 # boltzman constant 
alpha = 1/(12*np.pi) # coupling strength

# excited and ground states
e = np.array([[1],[0]])
g = np.array([[0],[1]])

# Pauli matricies
sigma_y = np.array([[0, -j], [-j, 0]])
I = np.array([[1,0], [0,1]])

# time parameters
dt = 0.1  # stepsize


# being main program
if __name__ == '__main__':
    
    start = timeit.default_timer() # start timer
    
    # check how many cores we have
    core_number = multiprocessing.cpu_count()
    print('You have {0:1d} CPU cores'.format(core_number))

    # traj number
    number_of_traj = 4
    
    core_traj = [int(number_of_traj/core_number) for i in range(core_number)]

    #Create the worker pool
    pool = Pool(processes=core_number) 
    
    # different regimes
    regime = [0.02*pow(10, i) for i in range(3)]
    Temps = [0.01*pow(10,i) for i in range(3)]
    ramp = [10/regime[i] for i in range(len(regime))]
    norm_regime = [regime[i]/max(regime) for i in range(len(regime))]
    
    # 2D data array
    master_data = []
    
    # loop over different temps
    for Temp in Temps:
        
        print("T = ", Temp)
        
        # containers for each regime
        master_minus = []
        master_plus = []
        master_up = []
        master_down = []
        master_work= []
        master_heat = []
        
    
        # loop over different regimes
        for t in regime:
            
            print("tmax = ",t)
            
            # set the regime
            tmax = t
            
            MC = partial(single_MC, tmax = t, T = Temp)
            
            # parallel map
            results = pool.map(MC, core_traj)
        
            # combine results of each cores calculations
            core_work = [results[i][0] for i in range(len(results))]
            core_heat = [results[i][1] for i in range(len(results))]  
            plus_pop = [results[i][2] for i in range(len(results))]
            minus_pop = [results[i][3] for i in range(len(results))]
            spin_up = [results[i][4] for i in range(len(results))]
            spin_down = [results[i][5] for i in range(len(results))]
     
            # average each cores results
            minus_avg = np.mean(minus_pop, axis = 0)
            plus_avg = np.mean(plus_pop, axis = 0)
            up_avg = np.mean(spin_up, axis = 0)
            down_avg = np.mean(spin_down, axis = 0)
            work = np.mean(core_work, axis = 0)
            heat = np.mean(core_heat, axis = 0)
    
            # store results on each regime
            master_plus.append(plus_avg)
            master_minus.append(minus_avg)
            master_up.append(up_avg)
            master_down.append(down_avg)
            master_work.append(work)
            master_heat.append(heat)
            
        # append to master data
        master_data.append([master_minus, master_plus, master_up, master_down, master_work, master_heat])
        
       
       
    # close the pool
    pool.close()
        
    # output calculation time
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    
    #combined figure
    fig = plt.figure(figsize = (9,5))
    
    # cycle colours
    colors = ['b', 'r', 'g']
    cc = itertools.cycle(colors)
    plot_lines = []
    
    # first subplot
    ax1 = fig.add_subplot(1,2,1)
    for i in range(len(master_data)):
               
  
        c = next(cc)
        
        # plot data
        up, = ax1.plot(np.log(regime), master_data[i][2], '--', color=c)
        down, = ax1.plot(np.log(regime), master_data[i][3], '-', color=c)
        plot_lines.append([up, down])
        
    
    # subplot 1 legend
    legend1 = ax1.legend(plot_lines[0], [r'$\rho_{\uparrow\hspace{-0.5}\uparrow}$', r'$\rho_{\downarrow\hspace{-0.5}\downarrow}$'], loc=1)
    ax1.legend([l[0] for l in plot_lines], Temps, loc=4)
    
    # subplot 1 titles
    ax1.set_xlabel(r'$log(t_{max})$')
    ax1.set_ylabel('$\overline{P}$')
    ax1.set_title('Final spin-state population')
        
        
    # cycle colours
    colors = ['b', 'r', 'g']
    cc = itertools.cycle(colors)
    plot_lines = []
        
    # subplot 2
    ax2=fig.add_subplot(1,2,2)
    for i in range(len(master_data)):
        
 
        c = next(cc)
        
        # plot data
        work = ax1.plot(np.log(regime), master_data[i][4], '--', color=c)
        heat = ax1.plot(np.log(regime), master_data[i][5], '-', color=c)
        plot_lines.append([work, heat])
        
    # subplot 1 legend
    legend1 = ax1.legend(plot_lines[0], [r'$\rho_{\uparrow\hspace{-0.5}\uparrow}$', r'$\rho_{\downarrow\hspace{-0.5}\downarrow}$'], loc=1)
    ax1.legend([l[0] for l in plot_lines], Temps, loc=4)
    
    # subplot 1 titles
    ax1.set_xlabel(r'$log(t_{max})$')
    ax1.set_ylabel('Energy scale')
    ax1.set_title('Heat and work')
        
    # common title
    plt.suptitle('Effects of ramp speed with environment T = 0.1')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    
    
    
    
    