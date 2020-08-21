#### Egen kode ####

import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
import matplotlib.pyplot as plt
from project_module import *
import latex

seed = 31526
system = SolarSystem(seed)

"""
orbits = SystemMotion(system, steps=1_300_000)
# orbits.solve()
# orbits.plot_calc()
orbits.plot_exact()
plt.tight_layout()
# plt.savefig('../output/plots/orbits_calc.pdf')
plt.savefig('../output/plots/orbits_exact.pdf')
"""
"""
### Make a table of the periods
a = system.semi_major_axes
P = np.power(a, 3/2)
data = [['Planet', 'Period, [yr]']]
for i , period in enumerate(P):
  data.append([str(i), '{:.2f}'.format(period)])
latex.make_table(data)   # Own function for creating LaTeX table
"""
"""
threebody = ThreeBody(system, steps=1_200_000)
threebody.solve()
threebody.plot_calc()
plt.tight_layout()
plt.savefig('../output/plots/threebody.pdf')
threebody.gen_vid()
"""
"""
lander = Lander(system)
lander.land()
"""
