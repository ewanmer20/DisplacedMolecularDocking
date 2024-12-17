import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams.update({'font.size': 22})
succ_gbs=[0.0, 5.058365758754864, 21.40077821011673, 38.13229571984436, 56.03112840466926, 59.92217898832685, 59.92217898832685]
succ_uni= [ 0.,1.8957346,10.42654028,19.43127962,31.7535545,34.5971564,34.5971564 ]
succ_oh=[ 0.,5.55555556,8.33333333,13.88888889,36.11111111,41.66666667,41.66666667] 
succ_dgbs=[ 0.,8.,38.8,67.4,80.8,86.2,86.2]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16))
ax.plot(np.array(succ_gbs), label='GBS sampler',color='k',linestyle='--')
ax.plot(np.array(succ_dgbs), label='DGBS sampler', color='g')
ax.plot(np.array(succ_uni), label='Uniform samples', color='r')
ax.plot(np.array(succ_oh), label='Oh samples', color='b')
print('Oh sampler',np.array(succ_oh))
print('DGBS sampler',np.array(succ_dgbs))
print('Uniform sampler',np.array(succ_uni))
print('GBS sampler',np.array(succ_gbs))
ax.set_xlabel('Iteration step of local search algorithm')
ax.set_ylabel('Success rate (%)')
ax.grid()
plt.legend()
current_dir = os.path.dirname(__file__)
os.chdir(current_dir)
# plt.savefig('SuccessRate.png',dpi=300)
plt.savefig('SuccessRateTest.svg')
plt.show()