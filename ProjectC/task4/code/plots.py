import numpy as np
from plots_module import *
from project_module import *

inst = Star()

v_pec = 100
v_max = 10
v = 10
P = 30
t0 = 2

t = np.linspace(0, 3*P, 10000)

# plot_theoretical_vel(t, v_pec, v_max, P, t0, noise=True)
# plt.savefig('../output/plots/theoretical_star_vel.jpg')

# plot_theoretical_vel(t, v_pec, v_max, P, t0, noise=False)
# plt.savefig('../output/plots/theoretical_star_vel_nonoise.jpg')

# plot_orbital_vels(inst)
# plt.show()

# plot_raw_wavelength(inst)
# plt.show()

# plot_light_flux(inst)
# plt.show()

parameters(inst)
plt.show()
