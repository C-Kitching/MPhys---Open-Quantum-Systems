import numpy as np
import matplotlib.pyplot as plt

epsilon = np.linspace(-10, 10, 1000)

plt.figure(figsize = (9,5))

plt.subplot(1,2,2)
delta = 0.2 * max(epsilon)
plus = [1/2*np.sqrt(epsilon**2+delta**2) for epsilon in epsilon]
minus = [-1/2*np.sqrt(epsilon**2+delta**2) for epsilon in epsilon]
plt.plot(epsilon, epsilon/2, label = r'$\left| \uparrow \right>$', color = 'red', linestyle = '--')
plt.plot(epsilon, -epsilon/2, label = r'$\left| \downarrow \right>$', color = 'blue', linestyle = '--')
plt.plot(epsilon, plus, label = '$\left| + \\right>$', color = 'blue', linestyle = '-')
plt.plot(epsilon, minus, label = '$\left| - \\right>$', color = 'red', linestyle = '-')
plt.xlabel('$\epsilon(t)$', fontsize = 14)
plt.title(r'$\Delta \neq 0$', fontsize = 14)
plt.legend()
plt.ylabel('Energy scale', fontsize = 14)

plt.subplot(1,2,1)
delta = 0
plus = [1/2*np.sqrt(epsilon**2+delta**2) for epsilon in epsilon]
minus = [-1/2*np.sqrt(epsilon**2+delta**2) for epsilon in epsilon]
plt.plot(epsilon, epsilon/2, label = r'$\left| \uparrow \right>$', color = 'red', linestyle = '--')
plt.plot(epsilon, -epsilon/2, label = r'$\left| \downarrow \right>$', color = 'blue', linestyle = '--')
plt.plot(epsilon, plus, label = '$\left| + \\right>$', color = 'blue', linestyle = '-')
plt.plot(epsilon, minus, label = '$\left| - \\right>$', color = 'red', linestyle = '-')
plt.xlabel('$\epsilon(t)$', fontsize = 14)
plt.title('$\Delta = 0$', fontsize = 14)
plt.ylabel('Energy scale', fontsize = 14)

# common title
plt.suptitle('Avoided crossing', fontsize = 16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)