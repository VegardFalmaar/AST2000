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
