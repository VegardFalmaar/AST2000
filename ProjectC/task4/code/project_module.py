import numpy as np
import ast2000tools.constants as constants

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
    path = '../star_data/'
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
