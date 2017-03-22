import numpy as np
import matplotlib.pyplot as plt

fig,axs = plt.subplots(2,1)

###########################################################
ax = axs[0]
dat = np.loadtxt('ml_estimates.txt',skiprows=1)
periods = dat[:,0]
means = dat[:,1]
percs = dat[:,2:]
p10 = percs[:,2]
p30 = percs[:,6]
p50 = percs[:,10]
p70 = percs[:,14]
p90 = percs[:,18]

#for i,period in enumerate(periods):
ax.plot(periods,means,'b--',zorder=2)
ax.plot(periods,medians,'b-',zorder=2)
ax.fill_between(periods,p10,p90,color='b',alpha=0.2,zorder=2) #,edgecolor='none')
ax.fill_between(periods,p30,p70,color='b',alpha=0.2,zorder=2) #,edgecolor='none')

ax.set_xlabel('time series length [years]',fontsize=10)
ax.set_ylabel('random walk scale [mm/yr$\mathregular{^{0.5}}$]',fontsize=10)

ax.grid(True,ls=':',color='0.5',zorder=0)
ax.plot([periods[0],periods[-1]],[1.3,1.3],'k-',lw=2,zorder=1)
ax.set_xlim((periods[0],periods[-1]))
ax.set_ylim((0.0,2.5))
ax.set_title('Langbein',fontsize=10)
###########################################################

###########################################################
ax = axs[1]

dat = np.loadtxt('reml_estimates.txt',skiprows=1)
periods = dat[:,0]
means = dat[:,1]
percs = dat[:,2:]
p10 = percs[:,2]
p30 = percs[:,6]
p50 = percs[:,10]
p70 = percs[:,14]
p90 = percs[:,18]

#for i,period in enumerate(periods):
ax.plot(periods,means,'b--',zorder=2)
ax.plot(periods,medians,'b-',zorder=2)
ax.fill_between(periods,p10,p90,color='b',alpha=0.2,edgecolor='none',zorder=2)
ax.fill_between(periods,p30,p70,color='b',alpha=0.2,edgecolor='none',zorder=2)

ax.set_xlabel('time series length [years]',fontsize=10)
ax.set_ylabel('random walk scale [mm/yr$\mathregular{^{0.5}}$]',fontsize=10)

ax.grid(True,ls=':',color='0.5',zorder=0)
ax.plot([periods[0],periods[-1]],[1.3,1.3],'k-',lw=2,zorder=1)
ax.set_xlim((periods[0],periods[-1]))
ax.set_ylim((0.0,2.5))
ax.set_title('Me',fontsize=10)
###########################################################

plt.tight_layout()
plt.show()

