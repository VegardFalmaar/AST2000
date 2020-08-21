import matplotlib.pyplot as plt
import numpy as np

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
