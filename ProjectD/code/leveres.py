#### Egen kode ####

import numpy as np
import ast2000tools.constants as constants

def read_file(filename):
  """Read the data in one file."""
  data = []
  with open(filename, 'r') as infile:
    for line in infile:
      data.append([float(value) for value in line.split()])
  data = np.array(data)
  return data.T

class Star:
  def __init__(self):
    self.data = self.read_data()

  def read_data(self):
    """Call the read_file function to obtain the data from the files.
    Return the data as a list.
    """
    self.days = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14]
    path = '../data/'
    data = []
    for day in self.days:
      filename = path + 'spectrum_day{}.txt'.format(day)
      data.append(read_file(filename))
    return data

  def doppler(self, lmdas):
    """Calculate the relative velocity of the star from the observed wavelength."""
    lmda_0 = 656.3
    velocities = []
    for lmda in lmdas:
      velocities.append((lmda - lmda_0)*constants.c/lmda_0)
    return np.array(velocities)

  def flux_model(self, lmda, F_min, sigma, lmda_center):
    """Calculate the flux."""
    F_max = 1
    exponent = -np.power((lmda - lmda_center), 2)/(2*sigma**2)
    return F_max + (F_min - F_max)*np.exp(exponent)

  def delta(self, day, F_min, sigma, lmda_center):
    """"Return the sum of all squared deviances."""
    lmdas, flux = self.data[day]
    dev = flux - self.flux_model(lmdas, F_min, sigma, lmda_center)
    return np.sum(np.power(dev, 2))

  def find_lims(self, day):
    """Return intervals for F_min, sigma and lmda_center to use with least_squares."""
    lmda, flux = self.data[day]
    ind_min = np.argpartition(flux, 3)[:3]
    mean_min = np.mean(flux[ind_min])
    lmda_min = float(lmda[np.where(flux == np.amin(flux))])
    return [mean_min, 0.4 + 0.6*mean_min, 0.001, 0.05, lmda_min - 0.02, lmda_min + 0.02]

  def least_squares(self, day, lims, steps=[20, 20, 20]):
    """Return the optimized values found within the intervals given as input."""
    min_flux = np.linspace(lims[0], lims[1], steps[0])
    sigmas = np.linspace(lims[2], lims[3], steps[1])
    lmda_mins = np.linspace(lims[4], lims[5], steps[2])
    old = 1E8    # random big number
    params = np.zeros(3)
    for F_min in min_flux:
      for sigma in sigmas:
        for lmda_center in lmda_mins:
          new = self.delta(day, F_min, sigma, lmda_center)
          if new < old:
            params = [F_min, sigma, lmda_center]
            old = new
    return params, old

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='Computer Modern', size=15)

def plot_spectra(inst):
  """Plot the raw data."""
  for i in range(3):
    for j in range(4):
      if (i < 2 or j < 2):
        index = i*4 + j
      else:
        break
      lmda, flux = inst.data[index]
      plt.subplot(4, 1, j+1)
      plt.title('Day {}'.format(inst.days[index]))
      plt.plot(lmda, flux)
      plt.xlabel('Wavelength, [nm]')
      plt.ylabel('Relative flux')
    plt.suptitle('Flux relative to the continuum flux around spectral line')
    plt.show()

def plot_analytical_flux(inst, inc_noise=False):
  F_min = 0.8
  sigma = 0.01
  lmda_center = 656.4
  lmda = np.linspace(656.2, 656.6, 800)
  noise = np.random.normal(loc=0, scale=0.05, size=len(lmda)) if inc_noise else 0
  flux = inst.flux_model(lmda, F_min, sigma, lmda_center) + noise
  plt.plot(lmda, flux)
  plt.title('Received flux relative to the continuum flux')
  plt.xlabel('Wavelength, [nm]')
  plt.ylabel('Relative flux')
  plt.xticks([lmda_center], [r'$\lambda_{center}$'])
  plt.yticks([F_min, 1], [r'$F_{min}$', r'$F_{max}$'])
  plt.grid()
  plt.tight_layout()

def plot_flux(inst):
  day = 0
  lims = inst.find_lims(day)
  params, delta = inst.least_squares(day, lims, steps=[30, 30, 30])
  F_min, sigma, lmda_center = params
  lmdas, flux = inst.data[day]
  plt.plot(lmdas, flux, 'b', label='observed')
  plt.plot(lmdas, inst.flux_model(lmdas, F_min, sigma, lmda_center), 'r', label='modelled')
  plt.title('Observed vs. modelled flux for day 0')
  plt.xlabel(r'$\lambda$, [nm]')
  plt.ylabel('Relative flux')
  plt.legend()

def plot_vel(times, vels):
  plt.plot(times, vels)
  plt.title('Radial velocity of the star')
  plt.xlabel('Time, [days]')
  plt.ylabel(r'$v_r$, [m/s]')
  plt.tight_layout()

inst = Star()
path = '../output/plots/'

#plot_spectra(inst)

### Plot theoretical flux
#plot_analytical_flux(inst)#, inc_noise=True)
#plt.savefig(path + 'theoretical_flux_noise.pdf')
#plt.savefig(path + 'theoretical_flux_nonoise.pdf')

### Plot modelled flux
# plot_flux(inst)
# plt.savefig(path + 'modelled_flux.pdf')

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
