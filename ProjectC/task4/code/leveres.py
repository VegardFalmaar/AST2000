### Egen kode ###

import numpy as np
import ast2000tools.constants as constants
import matplotlib.pyplot as plt

def read_file(filename):
  data = []
  with open(filename, 'r') as infile:
    for line in infile:
      data.append([float(value) for value in line.split()])
  data = np.array(data)
  return data[:, 0], data[:, 1], data[:, 2]

class Star:
  def __init__(self):
    self.read_data()

  def read_data(self):
    path = ''# '../star_data/'
    self.masses = np.array([1.36, 2.68, 6.62, 1.02, 1.42])
    data = []
    for i, mass in enumerate(self.masses):
      filename = path + 'star{}_{}.txt'.format(i, mass)
      data.append(np.array(read_file(filename)))
    self.data = data

  def v_model(self, t, t_0, P, v_max, noise=False):
    err = np.random.normal(loc=0, scale=0.2*v_max, size=len(t)) if noise else 0
    return v_max*np.cos(2*np.pi/P*(t - t_0)) + err

  def doppler(self):
    lmda_0 = 656.28
    velocities = []
    for data in self.data:
      lmda = data[1]
      velocities.append((lmda - lmda_0)*constants.c/lmda_0)
    return velocities

  def peculiars(self):
    peculiars = np.zeros(len(self.data))
    velocities = self.doppler()
    for i, velocity in enumerate(velocities):
      peculiars[i] = np.mean(velocity)
    return velocities, peculiars

  def orbital_vel(self):
    orbital_velocities = []
    velocities, peculiars = self.peculiars()
    for i, velocity in enumerate(velocities):
      orbital_velocities.append(velocity - peculiars[i])
    self.orbital_velocities = orbital_velocities

  def delta(self, star, t_0, P, v_r):
    t = self.data[star][0]
    v_data = self.orbital_velocities[star]
    dev = v_data - self.v_model(t, t_0, P, v_r, noise=False)
    return np.sum(np.power(dev, 2))

  def least_squares(self, star, lims, steps=[20, 20, 20]):
    t_0_min, t_0_max, P_min, P_max, v_r_min, v_r_max = lims
    t_0_list = np.linspace(t_0_min, t_0_max, steps[0])
    P_list = np.linspace(P_min, P_max, steps[1])
    v_r_list = np.linspace(v_r_min, v_r_max, steps[2])
    optimized = np.array([t_0_min, P_min, v_r_min])
    old_min = self.delta(star, t_0_min, P_min, v_r_min)
    for t_0 in t_0_list:
      for P in P_list:
        for v_r in v_r_list:
          new = self.delta(star, t_0, P, v_r)
          if new < old_min:
            optimized = [t_0, P, v_r]
            old_min = new
    return old_min, optimized

  def find_lims(self, star):
    t = self.data[star][0]
    vel = self.orbital_velocities[star]
    t_max = float(t[np.where(vel == np.amax(vel))])
    ind_max = np.argpartition(vel, -10)[-10:]
    mean_max = np.mean(vel[ind_max])
    period = (t[-1] - t[0])/2
    return [t_max-400, t_max+400, period-300, period+300, 0.7*mean_max, mean_max]

  def planet_mass(self, star, P, v_r):
    star_mass = self.masses[star]*constants.m_sun
    P *= 24*3600
    numerator = star_mass**(2/3)*v_r*P**(1/3)
    denominator = (2*np.pi*constants.G)**(1/3)
    return numerator/denominator

def plot_theoretical_vel(t, v_pec, v_max, P, t0, noise=False):
  vel = v_pec + Star().v_model(t, t0, P, v_max, noise=noise)
  plt.plot(t, vel)
  xlocs = [t0 + i*P for i in range(3)]
  xticks = ['t0', 't0 + P', 't0 + 2P']
  plt.xticks(xlocs, xticks)
  ylocs = [v_pec - v_max, v_pec, v_pec + v_max]
  yticks = ['v_min', 'v_pec', 'v_max']
  plt.yticks(ylocs, yticks)
  plt.grid()
  plt.title('Velocity of the star relative to earth')
  plt.xlabel('time')
  plt.ylabel('velocity')

def plot_orbital_vels(inst):
  inst.orbital_vel()
  for i, vel in enumerate(inst.orbital_velocities):
    plt.subplot(3, 2, i+1)
    plt.title('Star {}, {} solar masses'.format(i+1, inst.masses[i]))
    plt.plot(inst.data[i][0], vel)
    plt.xlabel('time [days]')
    plt.ylabel('v [m/s]')
    plt.grid()
  plt.suptitle('Observed orbital velocities of the five stars')

def plot_raw_wavelength(inst):
  for i, data in enumerate(inst.data):
    t, lmda, flux = data
    plt.subplot(2, 3, i+1)
    plt.title('Star {}, {} solar masses'.format(i+1, inst.masses[i]))
    plt.plot(t, lmda)
    plt.xlabel('time [days]')
    plt.ylabel('wavelength [nm]')
    min_, max_ = np.amin(lmda), np.amax(lmda)
    ylocs = np.linspace(min_, max_, 10)
    yticks = ['{:.6f}'.format(val) for val in ylocs]
    plt.yticks(ylocs, yticks)
    plt.grid()
  plt.suptitle('Observed wavelength of a spectral line')

def plot_light_flux(inst):
  for i, data in enumerate(inst.data):
    t, lmda, flux = data
    plt.subplot(3, 2, i+1)
    plt.title('Star {}, {} solar masses'.format(i+1, inst.masses[i]))
    plt.plot(t, flux)
    plt.xlabel('time [days]')
    plt.ylabel('relative flux')
    plt.grid()
  plt.suptitle('Measured flux of light relative to maximum flux')

def parameters(inst):
  inst.orbital_vel()
  star = 3
  print('By-eye limits for the parameters')
  print('Star 4:')

  lims = [1000, 2000, 3000, 4000, 25, 45]
  min_val, params = inst.least_squares(star, lims, steps=[40, 40, 40])
  t_0, P, v_r = params
  print('Minimum squared error: {:.0f}'.format(min_val))
  print('{:4}: {:.2f} days'.format('t_0', t_0))
  print('{:4}: {:.2f} days'.format('P', P))
  print('{:4}: {:.2f}m/s'.format('v_r', v_r))
  mass = inst.planet_mass(star, P, v_r)
  print('mass: {:.2E}kg = {:.3f} Jupiter masses'.format(mass, mass/1.898E27))

  print('\nAutomatically selected limits for the parameters')
  for i, star in enumerate([3, 4]):
    print('Star {}:'.format(star+1))
    lims = inst.find_lims(star)
    min_val, params = inst.least_squares(star, lims, steps=[40, 40, 40])
    t_0, P, v_r = params
    print('Minimum squared error: {:.0f}'.format(min_val))
    print('{:4}: {:.2f} days'.format('t_0', t_0))
    print('{:4}: {:.2f} days'.format('P', P))
    print('{:4}: {:.2f}m/s'.format('v_r', v_r))
    mass = inst.planet_mass(star, P, v_r)
    print('mass: {:.2E}kg = {:.3f} Jupiter masses'.format(mass, mass/1.898E27))
    print()

    plt.subplot(2, 1, i+1)
    plt.title('Star {}, {} solar masses'.format(star+1, inst.masses[star]))
    t = inst.data[star][0]
    plt.plot(t, inst.orbital_velocities[star])
    plt.plot(t, inst.v_model(t, t_0, P, v_r), 'r-', label='Modelled solution')
    plt.xlabel('time [days]')
    plt.ylabel('v [m/s]')
    plt.legend()
    plt.grid()
  plt.suptitle('Observed orbital velocities of the five stars')


inst = Star()

### Peculiar velocities
# print('Peculiar velocities:')
# for i, vel in enumerate(inst.peculiars()[1]):
#   print('Star {}: {:4.0f}km/s'.format(i+1, vel/1000))

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
