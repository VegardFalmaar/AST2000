import matplotlib.pyplot as plt
import numpy as np
from project_module import Star

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
