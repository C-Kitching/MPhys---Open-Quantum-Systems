import numpy as np
import matplotlib.pyplot as plt



regime = 2 * np.logspace(-2,3, 15)
print(regime)

x = [0.1+0.9*np.exp(-regime[i]*0.002) for i in range(len(regime))]
y = [0.9*(1 - np.exp(-regime[i]*0.002)) for i in range(len(regime))]






plt.plot(np.log(regime), x, label = r'$\frac{dS}{dt}$')
plt.plot(np.log(regime), y, label = r'$|\hspace{0.1}J\hspace{0.1}|$')
plt.legend(fontsize = 14)
plt.xlabel(r'$log(t_{max})$', fontsize = 14)
plt.ylabel('Normalised entropy', fontsize = 14)
plt.title('Entropy balance components with ramp speed', fontsize = 14)


















