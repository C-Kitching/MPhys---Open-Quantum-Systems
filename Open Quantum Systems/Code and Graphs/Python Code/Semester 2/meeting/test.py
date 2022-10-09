import numpy as np
import matplotlib.pyplot as plt


x = [0.02*pow(10, i) for i in range(3)]

print(x)

T = [0.01*pow(10,i) for i in range(3)]
print(T)

master_work2 = [4.999749536842715,4.9996976512726565,4.99640369192344,4.382367994466725,2.804768590859619,2.2541260122165205,1.773300547141912]

master_heat2= [0.0
,0.0
,0.0
,-0.020536021856255065
,0.11276233969311418
,0.37107232173805227
,0.47681194918412634]

master_up2 = [0.9999748299407288
,0.9999666132563328
,0.9989292131885806
,0.872924334570232
,0.5830059191681896
,0.5250168842601872
,0.4500025081475848]

master_down2 = [2.5170059271473686e-05
,3.338674366738327e-05
,0.0010707868114194952
,0.1270756654297684
,0.41699408083181067
,0.474983115739813
,0.5499974918524153]

#combined figure
plt.figure(figsize = (9,5))

plt.subplot(1,2,2)
plt.plot(np.log(x), master_up2, label = r'$\rho_{\uparrow\hspace{-0.5}\uparrow}$')
plt.plot(np.log(x), master_down2, label = r'$\rho_{\downarrow\hspace{-0.5}\downarrow}$')
plt.xlabel(r'$log(t_{max})$')
plt.ylabel('$\overline{P}$')
plt.legend()
plt.title('Final spin-state population')

plt.subplot(1,2,1)
plt.plot(np.log(x), master_work2, label = r'Work on system')
plt.plot(np.log(x), master_heat2, label = r'Heat to system')
plt.xlabel(r'$log(t_{max})$')
plt.ylabel('Energy scale')
plt.legend()
plt.title('Heat and work')

# common title
plt.suptitle('Effects of ramp speed with environment T = 0.1')
plt.tight_layout()
plt.subplots_adjust(top=0.88)