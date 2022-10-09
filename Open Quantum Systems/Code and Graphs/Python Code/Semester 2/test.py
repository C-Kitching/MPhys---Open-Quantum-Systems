import numpy as np
import matplotlib.pyplot as plt

# calculate eta
def eta(time):
    return(np.sqrt((tunnelling_coeff(time))**2+detuning(time)**2))

# calculate time dependent tunneling coefficient
def tunnelling_coeff(time):
    return 0.5*detuning(tmax)
    
# calculate time dependent bias
def detuning(time):  
    
    return 10*(time/tmax - 0.5)

tmin = 0
tmax = 2 # start 

x = np.linspace(tmin,tmax,100)

c = -2
y = [eta(x[i]) for i in range(len(x))]
z = [detuning(x[i]) for i in range(len(x))]
#z = [-np.sqrt((x[i]-c)**2+1**2) for i in range(len(x))]

#plt.plot(x, z)
plt.plot(z, y)