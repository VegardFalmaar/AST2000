import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='Computer Modern', size=15)
path = '../output/plots/'

def orbits(r, labels):
  for i in range(1):
    plt.plot(r[:, i, 0], r[:, i, 1], label=labels[i])
  plt.title('Orbit relative to the center of mass')
  plt.xlabel('x, [AU]')
  plt.ylabel('y, [AU]')
  plt.axis('equal')
  plt.legend()
  plt.tight_layout()
  plt.savefig(path + 'orbits_star_cm.pdf')
  plt.clf()

def vels(t, v):
  plt.plot(t, v[:, 0], label=r'$v_x$')
  plt.plot(t, v[:, 1], label=r'$v_y$')
  plt.title('Orb. vel. of the star relative to center of mass')
  plt.xlabel('Time, [yr]')
  plt.ylabel('v, [m/s]')
  plt.legend()
  plt.tight_layout()
  plt.savefig(path + 'orbital_star_vel_cm.pdf')
  plt.clf()

def radial_vel(t, v):
  plt.plot(t, v)
  plt.title('Radial velocity of the star with noise')
  plt.xlabel('Time, [yr]')
  plt.ylabel('v, [m/s]')
  plt.tight_layout()

def eclipse_plot(rcm, times, planet, labels):
  planet = np.copy(planet) + 1
  for i in range(5):
    plt.plot(rcm[:, i, 0], rcm[:, i, 1], label=labels[i])
  plt.plot(rcm[(times, planet, 0)], rcm[(times, planet, 1)], 'rx')
  plt.title('Orbits around the center of mass, eclipses marked')
  plt.xlabel('x, [AU]')
  plt.ylabel('y, [AU]')
  plt.legend()
  plt.axis('equal')
  plt.tight_layout()
  plt.savefig(path + 'orbits_eclipse.pdf')
  plt.clf()

def flux_plot(t, flux):
  plt.plot(t, flux)
  plt.title('Flux received from the star\nrelative to normal value')
  plt.xlabel('Time, [hours]')
  plt.ylabel('Relative flux')
  plt.tight_layout()
  plt.savefig(path + 'flux.pdf')
  plt.clf()
  noise = np.random.normal(loc=0, scale=0.2, size=len(flux))
  plt.plot(t, flux+noise)
  plt.title('Flux received from the star\nrelative to normal value, with noise')
  plt.xlabel('Time, [hours]')
  plt.ylabel('Relative flux')
  plt.tight_layout()
  plt.savefig(path + 'flux_noise.pdf')
  plt.clf()
