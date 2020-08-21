#### Egen kode ####

import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='Computer Modern', size=15)
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

class SystemMotion:
  def __init__(self, system, dt=2E-5, steps=1_000_000):
    """Set values for the masses, number of bodies and their initial positions and velocities."""
    self.M = system.star_mass
    self.N = system.number_of_planets
    self.G = constants.G_sol
    self.dt, self.steps = dt, steps
    self.r = np.zeros((self.N, steps, 2))
    self.v = np.zeros_like(self.r)
    self.t = np.zeros(steps)
    self.r[:, 0, :] = system.initial_positions.T
    self.v[:, 0, :] = system.initial_velocities.T
    self.system = system

  def _a(self, positions, _):
    """Return an array with the acceleration of every object."""
    r_norm = np.linalg.norm(positions, axis=1).reshape(-1, 1)
    return -self.G*self.M*positions/r_norm**3

  def solve(self):
    """Solve the motion of the bodies numerically.
    Use 'steps' number of time steps of length 'dt'.
    """
    i = 0
    while self._condition(i):
      self.v[:, i+1, :] = self.v[:, i, :] + self._a(self.r[:, i, :], self.v[:, i, :])*self.dt
      self.r[:, i+1, :] = self.r[:, i, :] + self.v[:, i+1, :]*self.dt
      self.t[i+1] = self.t[i] + self.dt
      i += 1
      self.i = i
    self.i = i

  def _condition(self, i):
    return True if i < self.steps-1 else False

  def plot_calc(self):
    """Plot the calculated orbits of the planets."""
    plt.title('Landing')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    # plt.plot(0,0, 'ro', label='Star')
    # names = ['Planet', 'Star 1', 'Star 2']
    for i in range(self.N):
      plt.plot(self.r[i, :self.i:100, 0], self.r[i, :self.i:100, 1], label='Lander', color=colors[i])
    plt.axis('equal')
    plt.legend()
    # plt.grid()

  def plot_exact(self):
    """Plot the calculated orbits vs. the analytical solutions for the orbits."""
    system = self.system
    omega = system.aphelion_angles.reshape(-1, 1)
    a = system.semi_major_axes.reshape(-1, 1)
    e = system.eccentricities.reshape(-1, 1)
    theta = np.linspace(0, 2*np.pi, 1000)
    f = theta-omega+np.pi
    r = a*(1 - e**2)/(1 + e*np.cos(f))

    plt.title('Analytical orbits')
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.plot(0,0, 'ro', label='Star')
    for i in range(self.N):
      plt.plot(r[i]*np.cos(theta), r[i]*np.sin(theta), label='{}'.format(i), color=colors[i])
    plt.axis('equal')
    plt.legend()
    plt.grid()



class ThreeBody(SystemMotion):
  def __init__(self, system, steps=1_000_000):
    """Set values for the masses, number of bodies and their initial positions and velocities."""
    m_mars = 6.4185E23/constants.m_sun
    self.masses = np.array([[m_mars], [1], [4]])
    self.G = constants.G_sol
    self.dt = 400/constants.yr
    self.steps = steps
    self.N = 3
    self.r = np.zeros((self.N, steps, 2))
    self.v = np.zeros_like(self.r)
    self.t = np.zeros(steps)
    self.r[:, 0, :] = np.array([[-1.5, 0], [0, 0], [3, 0]])
    self.v[:, 0, :] = np.array([[0, -1], [0, 30], [0, -7.5]])*1000*constants.yr/constants.AU
    self.system = system

  def _F_G(self, positions, i1, i2):
    G = constants.G_sol
    r = positions[i2] - positions[i1]
    r_norm = np.linalg.norm(r)
    F = G*self.masses[i1, 0]*self.masses[i2, 0]*r/r_norm**3
    return F

  def _a(self, positions, velocities):
    F_ps1 = self._F_G(positions, 0, 1)
    F_ps2 = self._F_G(positions, 0, 2)
    F_s1s2 = self._F_G(positions, 1, 2)
    return np.array([F_ps1 + F_ps2, -F_ps1 + F_s1s2, -F_ps2 - F_s1s2])/self.masses

  def gen_vid(self):
    pos_planet, pos_star1, pos_star2 = self.r
    self.system.generate_binary_star_orbit_video(self.t, pos_planet.T, pos_star1.T, pos_star2.T)

class Lander(SystemMotion):
  def __init__(self, system, planet_index=1, dt=0.1,steps=100_000):
    """Set values for the masses, number of bodies and their initial positions and velocities."""
    self.system = system
    self.mass = 100
    self.N = 1
    self._planetary_properties(planet_index)
    self.dt, self.steps = dt, steps
    self.r = np.zeros((self.N, steps, 2))
    self.v = np.zeros_like(self.r)
    self.t = np.zeros(steps)

    orbital_radius = self.planet_radius + 40_000_000
    self.r[:, 0, :] = np.array([[orbital_radius, 0]])
    v = np.sqrt(constants.G*self.M/orbital_radius)
    self.v[:, 0, :] = np.array([[-500, 0.574*v]])
    self.A = 100
    self.G = constants.G

    self.landed = False
    self.dt_status = 'large'

  def _planetary_properties(self, index):
    system = self.system
    self.M = system.masses[index]*constants.m_sun
    self.planet_radius = system.radii[index]*1000
    self.rho0 = system.atmospheric_densities[index]
    g = constants.G*self.M/self.planet_radius**2
    self.h_scale = 75_200/g

  def _a(self, positions, velocities):
    r_norm = np.linalg.norm(positions, axis=1).reshape(-1, 1)
    a_G = -self.G*self.M*positions/r_norm**3

    h = r_norm - self.planet_radius
    rho = self.rho0*np.exp(-h/self.h_scale)

    drag = -0.5*rho*self.A*np.linalg.norm(velocities)*velocities
    if np.linalg.norm(drag) > 25_000:
      print('Position: ', positions)
      print('          ', np.linalg.norm(positions))
      print('Velocity: ', velocities)
      print('          ', np.linalg.norm(velocities))
      raise ValueError('Drag force was too large: {:.0f}N.'.format(np.linalg.norm(drag)))
    return a_G + drag/self.mass

  def _condition(self, i):
    r_norm = np.linalg.norm(self.r[:, i, :])
    v_norm = np.linalg.norm(self.v[:, i, :])
    E = 0.5*self.mass*v_norm**2 - constants.G*self.mass*self.M/r_norm
    assert E < 0, 'Wrong energy, i = {}'.format(i)

    if self.dt_status == 'small' and r_norm-self.planet_radius > 100*self.h_scale:
      self.dt *= 1000
      print('dt changed to {:.0E} in time step: {}'.format(i))
      self.dt_status = 'large'
    elif self.dt_status == 'large' and r_norm-self.planet_radius < 100*self.h_scale:
      self.dt /= 1000
      print('dt changed to {:.0E} in time step: {}'.format(self.dt, i))
      self.dt_status = 'small'

    if r_norm > self.planet_radius and i < self.steps-1:
      return True
    elif i >= self.steps-1:
      print('Finished running {} time steps.'.format(i+1))
      return False
    else:
      print('The eagle has landed')
      print('With radial velocity: {}'.format(self.v[0, i, :].dot(self.r[0, i, :]/r_norm)))
      self.landed = True
      return False

  def plot_planet(self):
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(self.planet_radius*np.cos(theta), self.planet_radius*np.sin(theta), 'b--', label='Planet')

  def land(self):
    self.solve()
    self.plot_calc()
    self.plot_planet()
    plt.grid()
    plt.tight_layout()
    plt.savefig('../output/plots/lander_precise.pdf')
    while not self.landed:
      print()
      print('Position: ({:.2g}, {:.2g})m'.format(self.r[0, -1, 0], self.r[0, -1, 1]))
      print('Height:   {:.0f}m'.format(np.linalg.norm(self.r[:, -1, :]) - self.planet_radius))
      print('Velocity: ({:.2g}, {:.2g})m/s'.format(self.v[0, -1, 0], self.v[0, -1, 1]))
      print('          {:.2f}m/s'.format(np.linalg.norm(self.v[:, -1, :])))
      self.r[0, 0, :] = np.array([self.r[0, -1, :]])
      self.v[0, 0, :] = np.array([self.v[0, -1, :]])
      self.solve()
      self.plot_calc()
      plt.tight_layout()
      plt.savefig('../output/plots/lander_precise.pdf')

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
import latex
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
