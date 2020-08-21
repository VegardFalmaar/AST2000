import numpy as np
from project_module import *
from plots_module import *

inst = Star()

### By-eye estimates for the center of the spectral lines
# lmdas = 656 + np.array([0.331, 0.334, 0.337, 0.334, 0.331, 0.328, 0.33, 0.333, 0.337, 0.335])


### Least squares for the center of the spectral lines

lmdas = np.zeros(10)
for day in range(10):
  lims = inst.find_lims(day)
  params, delta = inst.least_squares(day, lims, steps=[30, 30, 30])
  lmdas[day] = params[2]

vels = inst.doppler(lmdas)
for i, vel in enumerate(vels):
  print('Day {:2}, velocity: {:.1f}km/s'.format(inst.days[i], vel/1000))

times = np.array([0,2,3,5,6,8,9,11,13,14])
plot_vel(times, vels)
plt.savefig('../output/plots/radial_vels.pdf')


### Least squares for the center of the spectral lines
"""
values = np.zeros((2, 10, 4))     # indexed by lims, day, param
for i, steps in enumerate([[30, 30, 30], [40, 40, 40]]):
  print('lims:', steps)
  print(('{:^10}'*5).format('Day index', 'F_min', 'sigma', 'lmda', 'delta'))
  print('-'*50)
  for day in range(10):
    lims = inst.find_lims(day)
    params, delta = inst.least_squares(day, lims, steps=steps)
    print(('{:^10.0f}' + '{:^10.5f}'*4).format(day, params[0], params[1], params[2], delta))
    values[i, day] = params + [delta]
  print()

relative_change = (values[1] - values[0])/values[0]*100
print('Relative change')
print(('{:^10}'*5).format('Day index', 'F_min', 'sigma', 'lmda', 'delta'))
print('-'*50)
for day, data in enumerate(relative_change):
  print(('{:^10.0f}' + '{:^10.5f}'*4).format(day, data[0], data[1], data[2], data[3]))
"""
