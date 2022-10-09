import matplotlib.pyplot as plt
import numpy as np


regime = [0.02
,0.2
,2.0
,20.0
,200.0
,2000.0
,20000.0]


master_work2 = [4.999749536842715
,4.9996976512726565
,4.893957788661118
,4.470007724840696
,2.437277027170789
,1.9279654330005587
,1.7115662145544484]


master_heat2 = [0.0
,0.0
,-0.022730944446173004
,-0.0018289205226500937
,0.16527209026162748
,0.3953894141362292
,0.4885437817102113]


master_up2 = [0.9999748299407288
,0.9999666132563328
,0.9738703293235809
,0.8943329925455783
,0.5201603789542134
,0.46463667850644036
,0.44000299189522807]


master_down2 = [2.5170059271473686e-05
,3.338674366738328e-05
,0.02612967067641937
,0.10566700745442213
,0.4798396210457868
,0.5353633214935599
,0.5599970081047722]



#combined figure
plt.figure(figsize = (9,5))

plt.subplot(1,2,2)
plt.plot(np.log10(regime), master_up2, label = r'$\rho_{\uparrow\hspace{-0.5}\uparrow}$')
plt.plot(np.log10(regime), master_down2, label = r'$\rho_{\downarrow\hspace{-0.5}\downarrow}$')
plt.xlabel(r'$log(t_{max})$', fontsize = 14)
plt.ylabel('$\overline{P}$', fontsize = 14)
plt.legend()
plt.title('Diabatic state population', fontsize = 14)

plt.subplot(1,2,1)
plt.plot(np.log10(regime), master_work2, label = r'Work on system')
plt.plot(np.log10(regime), master_heat2, label = r'Heat to system')
plt.xlabel(r'$log(t_{max})$', fontsize = 14)
plt.ylabel('Energy scale', fontsize = 14)
plt.legend()
plt.title('Heat and work', fontsize = 14)

# common title
plt.suptitle('Effects of ramp speed with environment', fontsize = 16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)


















