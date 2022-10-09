import numpy as np
import matplotlib.pyplot as plt



regime = [0.02
,0.2
,2.0
,20.0
,200.0
,2000.0
,20000.0]

master_heat = [0.0
,0.0
,-0.007695582345739264
,-0.005747350131259704
,0.023340779734624395
,0.4514810588165682
,0.5174170175232756]

master_work = [4.999749536842715
,4.9996976512726565
,4.829116395238685
,4.326879207363059
,2.3977783527253624
,1.8486333119853546
,1.8827029781929516]

master_up = [0.9999748299407288
,0.9999666132563328
,0.9639222819198194
,0.8648877715284169
,0.48378433894653394
,0.4599988738215432
,0.4800009872759029]

master_down = [2.5170059271473686e-05
,3.338674366738328e-05
,0.03607771808018072
,0.13511222847158347
,0.5162156610534664
,0.540001126178457
,0.5199990127240972]

#combined figure
plt.figure(figsize = (9,5))

plt.subplot(1,2,2)
plt.plot(np.log10(regime), master_up, label = r'$\rho_{\uparrow\hspace{-0.5}\uparrow}$')
plt.plot(np.log10(regime), master_down, label = r'$\rho_{\downarrow\hspace{-0.5}\downarrow}$')
plt.xlabel(r'$log(t_{max})$', fontsize = 14)
plt.ylabel('$\overline{P}$', fontsize = 14)
plt.legend()
plt.title('Diabatic state population', fontsize = 14)

plt.subplot(1,2,1)
plt.plot(np.log10(regime), master_work, label = r'Work on system')
plt.plot(np.log10(regime), master_heat, label = r'Heat to system')
plt.xlabel(r'$log(t_{max})$', fontsize = 14)
plt.ylabel('Energy scale', fontsize = 14)
plt.legend()
plt.title('Heat and work', fontsize = 14)

# common title
plt.suptitle('Effects of ramp speed with environment', fontsize = 16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)