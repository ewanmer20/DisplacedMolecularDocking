import matplotlib.pyplot as plt
import numpy as np
ngraphs=200
nsqz_tot=2
target_nsqz=np.linspace(0.1,1,nsqz_tot)
prob_edges=[0.3,0.8]
enhancement_array=np.loadtxt('Enhancement_array.txt')
enhancement_array=enhancement_array.reshape((len(prob_edges),nsqz_tot,ngraphs))

fig = plt.figure(figsize=plt.figaspect(0.4))
ax = fig.add_subplot(221)
plt.hist(enhancement_array[0,1,:])
ax.set_title('edge probability: {:.2f} squeezing photon number{:.2f}'.format(prob_edges[0],target_nsqz[1]))
ax = fig.add_subplot(222)
plt.hist(enhancement_array[1,1,:])
ax.set_title('edge probability: {:.2f} squeezing photon number{:.2f}'.format(prob_edges[1],target_nsqz[1]))
ax = fig.add_subplot(223)
plt.hist(enhancement_array[0,0,:])
ax.set_title('edge probability: {:.2f} squeezing photon number{:.2f}'.format(prob_edges[0],target_nsqz[0]))
ax = fig.add_subplot(224)
plt.hist(enhancement_array[1,0,:])
ax.set_title('edge probability: {:.2f} squeezing photon number{:.2f}'.format(prob_edges[1],target_nsqz[0]))

ax.set_ylabel('Enhancement')
fig.suptitle(r'Enhancement for Erdos-Renyi graphs of {:.1f} vertices and displacement of {:.1f} with {:.1f} graphs'.format(nvertices, target_ncoh, ngraphs), wrap=True)
plt.savefig('Enhancement for Erdos-Renyi graphs of {:.1f} vertices and displacement of {:.1f} with {:.1f} graphs.pdf'.format(nvertices, target_ncoh, ngraphs),format='pdf')
fig.show()