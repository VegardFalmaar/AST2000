import numpy as np
from ast2000tools.solar_system import SolarSystem
import ast2000tools.constants as constants
from project_module import *
from plots import *

seed = 31526
system = SolarSystem(seed)

inst = Motion(system)
r, v, rcm, vcm, t = inst.solve(2E-5, 1_000_000)
labels = ['Star'] + ['Planet {}'.format(i) for i in inst.indexes]
orbits(rcm, labels)
vcm *= constants.AU/constants.yr # Convert from AU/yr to m/s
vels(t, vcm[:, 0])

### Radial velocity
# Peculiar velocity of the star, [m/s]
v_pec = 316_527 
# Inclination of orbit is 60 degrees
i = np.pi/3
# Radial velocity is measured i y-direction
v_y = vcm[:, 0, 1]
sigma = np.amax(v_y)/5
noise = np.random.normal(loc=0, scale=sigma, size=len(v_y))
v_r = v_pec + v_y*np.sin(i)
radial_vel(t, v_r+noise)
radial_vel(t, v_r)
plt.savefig('../output/plots/star_radial_vel.pdf')
plt.clf()

### Flux
t *= constants.yr/3600     # Convert from years to hours
times, planet = inst.eclipse(rcm)
eclipse_plot(rcm, times, planet, labels)
flux = inst.flux(t, times, planet)
flux_plot(t, flux)
