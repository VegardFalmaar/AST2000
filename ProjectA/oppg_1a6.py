#### Egen kode ####

import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
import random
import time
import matplotlib.pyplot as plt


def timeit(func):
  """Decorator to time other functions, used to predict the run-time of the program."""
  def new(*args, **kwargs):
    start = time.time()
    rv = func(*args, **kwargs)
    stop = time.time()
    print('\nFunction {} runtime:'.format(func.__name__), stop - start)
    return rv
  return new

seed = 31526
system = SolarSystem(seed)
random.seed(seed)
np.random.seed(seed)

def escape_velocity(mass, radius):
  """Calculate the escape velocity [m/s] from the surface of a planet with mass and radius given as input."""
  return np.sqrt(2*constants.G*mass/radius)

class Planet:
  def __init__(self, index):
    self.planet_index = index
    self.planetary_properties()

  def planetary_properties(self):
    """Calculate properties of the home planet in SI units."""
    self.mass = system.masses[self.planet_index]*constants.m_sun
    self.radius = system.radii[self.planet_index]*1000

  def calc_v_esc(self):
    """Return the escape velocity of the planet."""
    self.v_esc = escape_velocity(self.mass, self.radius)
    return self.v_esc


class GasSim:
  # @timeit
  def __init__(self, no_of_particles, temperature, L):
    """Store values as class variables and give the particles initial positions and velocities according to the probability distributions."""
    self.T = temperature
    self.L = L
    self.N = no_of_particles
    self.m = constants.m_H2
    self.pi = constants.pi
    self.k = constants.k_B
    self.mass = 1000   # total mass of the satellite

    mean = 0
    sigma = np.sqrt(self.k*self.T/self.m)
    self.positions = np.random.uniform(low=0, high=self.L, size=(self.N, 3))
    self.velocities = np.random.normal(loc=mean, scale=sigma, size=(self.N, 3))

  def kinetic(self, v):
    """ Calculate the kinetic energy of a hydrogen gas particle with velocity vector v."""
    m = self.m
    v_squared = 0
    for v_i in v:
      v_squared += v_i**2
    return 0.5*m*v_squared

  # @timeit
  def test_E_K(self, show=False):
    """Test if the mean kinetic energy of the particles of the gas matches the analytical expression."""
    analytical_mean = 1.5*self.k*self.T
    computed_mean = 0
    for v in self.velocities:
      computed_mean += self.kinetic(v)
    computed_mean = computed_mean/self.N
    relative_error = abs(analytical_mean - computed_mean)/analytical_mean
    if show:
      print('\nKinetic energy:')
      print('{:<20}{:g}'.format('Computed mean:', computed_mean))
      print('{:<20}{:g}'.format('Analytical mean:', analytical_mean))
      print('{:<20}{:.2f}%'.format('Relative error:', relative_error*100))
    assert relative_error < 0.02, 'The mean kinetic energy is off'

    analytical_mean = np.sqrt(8*self.k*self.T/(self.pi*self.m))
    self.analytical_mean_vel = analytical_mean
    computed_mean = 0
    for v in self.velocities:
      add = 0
      for v_i in v:
        add += v_i**2
      computed_mean += np.sqrt(add)/self.N
    relative_error = abs(analytical_mean - computed_mean)/analytical_mean
    if show:
      print('\nMean velocity:')
      print('{:<20}{:.2f}'.format('Computed mean:', computed_mean))
      print('{:<20}{:.2f}'.format('Analytical mean:', analytical_mean))
      print('{:<20}{:.2f}%'.format('Relative error:', relative_error*100))
    assert relative_error < 0.01, 'The mean velocity is off'

  # @timeit
  def box_sim(self, time=1E-9, dt=1E-12, rocket=False):
    """Run a simulation of gas particles moving around the box and return.

    Keyword arguments:
    time   -- how long the simulation lasts (default 1E-9 seconds)
    dt     -- size of the time steps in the simulation (default 1E-12 seconds)
    rocket -- if this is set to true, the function will run a rocket simulation (see below, default False)

    Returns:
    escaped  -- the number of particles that escaped during rocket simulation or
                if rocket is set to False, the number of particles that were outside the box due to an error
    momentum -- the total momentum of the particles that escape during rocket simulation or
                if rocket is set to False, the total momentum of the particles that hit the wall along x=0
    hits     -- if rocket is set to False, the number of particles that hit the wall along x=0, else 0

    rocket simulation: During rocket simulation, the box will have a square hole with sides of length L/2
                        in one of the sides of the box. The function will return the number of particles
                        that escape through the hole and the total momentum of these particles.
    """
    steps = int(np.ceil(time/dt))
    buffer_zone = 0.0125*self.L
    escaped = 0
    hits = 0
    momentum = 0
    for _ in range(steps):
      for n in range(self.N):
        for j in range(3):
          if abs(self.positions[n, j] - 0.5*self.L) > 0.5*self.L + buffer_zone and not rocket:
            escaped += 1
          if self.positions[n, j] > self.L - buffer_zone and self.velocities[n, j] > 0:
            self.velocities[n, j] *= -1
          elif self.positions[n, j] < buffer_zone and self.velocities[n, j] < 0:
            if j == 0 and rocket and abs(self.positions[n, 1]-self.L/2) < self.L/4 and abs(self.positions[n, 2]-self.L/2) < self.L/4:
              escaped += 1
              momentum += self.m*abs(self.velocities[n, j])
              self.positions[n, j] += self.L
            elif j == 0:
              hits += 1
              if not rocket:
                momentum += self.m*abs(self.velocities[n, j])
              self.velocities[n, j] *= -1
            else:
              self.velocities[n, j] *= -1
      self.positions += self.velocities*dt
    return escaped, momentum, hits

  def test_pressure(self, time=1E-9):
    escaped, momentum, hits =  self.box_sim(time=time, rocket=False)
    F = 2*momentum/time
    A = self.L**2
    V = self.L**3
    computed_P = F/A
    analytical_P = self.N*self.k*self.T/V
    relative_error = abs(analytical_P - computed_P)/analytical_P
    print('\nPressure test (closed box):')
    print('{:<20}{}'.format('Hits:', hits))
    print('{:<20}{}'.format('Escaped:', escaped))
    print('{:<20}{:.2f}'.format('Computed P:', computed_P))
    print('{:<20}{:.2f}'.format('Analytical P:', analytical_P))
    print('{:<20}{:.2f}%'.format('Relative error:', relative_error*100))

  def rocket_sim(self, time=1E-9, show=False):
    escaped, momentum, hits = self.box_sim(time=time, rocket=True)
    mass_lost = escaped*self.m
    delta_v = momentum/self.mass
    if show:
      print('\nRocket simulation:')
      print('{:<20}{}'.format('Hits:', hits))
      print('{:<20}{}'.format('Escaped:', escaped))
      print('{:<20}{:g}'.format('Momentum:', momentum))
      print('{:<20}{:g}'.format('Speed gain:', delta_v))
    return delta_v, mass_lost

  def fuel_calc(self, t, planet):
    delta_v, fuel_mass = gas.rocket_sim(time=t, show=True)       # the speed gained during time interval time
    delta_v = delta_v/t                # the speed gained during a second
    acceleration_time = 20*60          # 20 minutes
    delta_v = delta_v*acceleration_time
    no_of_boxes = planet.v_esc/delta_v
    fuel_lost = fuel_mass*no_of_boxes*acceleration_time/t
    analytical_fuel_loss = self.mass*np.sqrt(3)*planet.v_esc/self.analytical_mean_vel
    print('{:<20}{:.3e}'.format('Number of boxes:', no_of_boxes))
    print('{:<20}{:g}{}'.format('Fuel loss:', fuel_lost, 'kg'))
    print('{:<20}{:g}{}'.format('Analytical fuel loss:', analytical_fuel_loss, 'kg'))

  def P_vel(self, x):
    sigma = np.sqrt(self.k*self.T/self.m)
    exponent = -x**2/(2*sigma**2)
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(exponent)


L = 1E-6          # Length of the sides of the box, [m]
T = 10_000        # Temperature of the gas, [K]
N = int(1E5)      # Number of particles in the box
t = 1E-9
gas = GasSim(N, T, L)

# 1)
home_planet = Planet(0)
esc_velocity = home_planet.calc_v_esc()
# print(home_planet.mass/5.97237E24, home_planet.radius/6371000)
print('Home planet mass:        {:.3e} kg'.format(home_planet.mass))
print('Home planet radius:      {:.0f} m/s'.format(home_planet.radius))
print('Home escape velocity:    {:.0f} m/s'.format(home_planet.v_esc))
print("Earth's escape velocity: {:.0f} m/s".format(escape_velocity(5.97237E24, 6_371_000)))
print('Number of planets:       {}'.format(system.number_of_planets))
# print('Star color:              {}'.format(system.star_color))
# system.print_info()
print('\nNumber of particles:', N)

x = np.linspace(-25000, 25000, 51)
x_labels = ['v_x', 'v_y', 'v_z']
for i, label in enumerate(x_labels):
    plt.hist(gas.velocities[:, i], bins=31, density=True, histtype='step')
    plt.plot(x, gas.P_vel(x), 'r-')
    plt.xlabel(label)
    plt.ylabel('Probability density')
    plt.show()


# 3)
gas.test_E_K(show=True)

# 4 og 5)
gas.test_pressure()

# 6, 7, 8 og 9)
gas.fuel_calc(t, home_planet)

"""
Terminal> python3 oppg_1a6.py
Home planet mass:        1.389e+25 kg
Home planet radius:      8375940 m/s
Home escape velocity:    14876 m/s
Earth's escape velocity: 11186 m/s
Number of planets:       7

Number of particles: 100000

Kinetic energy:
Computed mean:      2.07298e-19
Analytical mean:    2.0715e-19
Relative error:     0.07%

Mean velocity:
Computed mean:      10258.07
Analytical mean:    10249.67
Relative error:     0.08%

Pressure test (closed box):
Hits:               260799
Escaped:            82
Computed P:         13995.02
Analytical P:       13810.00
Relative error:     1.34%

Rocket simulation:
Hits:               217706
Escaped:            76666
Momentum:           2.07807e-18
Speed gain:         2.07807e-21
Number of boxes:    5.966e+12
Fuel loss:          1837.16kg
Analytical fuel loss:2513.86kg
"""
