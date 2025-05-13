import numpy as np
import matplotlib.pyplot as plt



bandlabels={'theta':'$\Theta$','lgamma':'low $\gamma$','mgamma': 'mid $\gamma$','hgamma': 'high $\gamma$'}
bandstr=list(bandlabels.keys())

col={'all_SOM':'green','local':'blue','proj':'orange'}
popstr=list(col.keys())

data={}
for band in bandstr:
    for pop in popstr:
        
        ending='_'+band+'_'+pop+'.txt'
        print(ending)
        circ=np.loadtxt('circ'+ending)
        mrl =np.loadtxt('mrl'+ending)
        data[(band,pop)]=np.append(circ.reshape(-1,1),mrl.reshape(-1,1),axis=1)



fig=plt.figure( )
fig.set_size_inches((8,6),forward=True)

for nb,band in enumerate(bandstr):
    ax=plt.subplot(2,2,nb+1, projection='polar')
    for pop in popstr:
        mat=data[(band,pop)]
        phases=np.mod(mat[:,0]+2*np.pi,2*np.pi)
        Rs=mat[:,1]
        mv=np.angle(np.nansum(np.exp(1j*phases)*Rs))
        ax.plot(phases,Rs,'.', alpha=0.5,  color=col[pop], label=pop)
        ax.plot([mv, mv],[0,1], '-', color=col[pop])    
        ax.set_title(bandlabels[band],fontsize=10)
        ax.set_ylim([0,1])
        ax.set_yticks([1])
        ax.set_yticklabels(['R=1'])
        ax.set_aspect('equal')
        if nb==0:
            ax.legend(frameon=False,loc='upper right', bbox_to_anchor=(1.75, 1.))
        
fig.subplots_adjust(wspace=.4,hspace=.5)
#fig.tight_layout() 
fig.savefig('locking.pdf')
#!gv locking.pdf
plt.show(block=0)
