#### Egen kode ####

import numpy as np
import ast2000tools.constants as constants

class Motion:
  def __init__(self, system):
    self.masses = np.zeros(5)
    self.masses[0] = system.star_mass

    # Find the four planets with the highest mass
    self.indexes = np.sort(np.argpartition(system.masses, -4)[-4:])
    self.masses[1:] = system.masses[self.indexes]
    self.r0 = system.initial_positions[:, self.indexes].T
    self.v0 = system.initial_velocities[:, self.indexes].T
    self.radii = system.radii[self.indexes]
    self.star_radius = system.star_radius

  def solve(self, dt, N):
    """Calculate the orbits of the star and the planets for
    N time steps of length dt.

    Return the positions, velocities, positions and velocities
    with regard to the center of mass of the system and an
    array t containing all the time steps.
    """
    r, v, rcm, vcm  = np.zeros((4, N, 5, 2))
    t = np.linspace(0, dt*(N-1), N)
    masses = self.masses.reshape((5, 1))
    r[0, 1:] = self.r0
    v[0, 1:] = self.v0
    cm_pos = np.sum(r[0]*masses, axis=0)/np.sum(masses)
    rcm[0] = r[0] - cm_pos
    cm_vel = np.sum(v[0]*masses, axis=0)/np.sum(masses)
    vcm[0] = v[0] - cm_vel
    for i in range(N-1):
      v[i+1] = v[i] + self.acceleration(r[i])*dt
      r[i+1] = r[i] + v[i+1]*dt
      cm_pos = np.sum(r[i+1]*masses, axis=0)/np.sum(masses)
      rcm[i+1] = r[i+1] - cm_pos
      cm_vel = np.sum(v[i+1]*masses, axis=0)/np.sum(masses)
      vcm[i+1] = v[i+1] - cm_vel
    return r, v, rcm, vcm, t

  def acceleration(self, r):
    """Calculate the acceleration of all the bodies in the
    system given their positions.
    """
    G = constants.G_sol
    a = np.zeros((5, 2))
    r_vec = r[1:] - r[0]
    r_norm = np.linalg.norm(r_vec, axis=1).reshape((4, 1))
    frac = r_vec/np.power(r_norm, 3)
    a[0] = G*np.sum(self.masses[1:].reshape((4, 1))*frac, axis=0)
    a[1:] = -G*self.masses[0]*frac
    return a

  def eclipse(self, rcm):
    """Return the indexes of the positions where the planets
    are in front of the sun star given the positions
    of the star and the planets with regard to the center
    of mass of the system.
    """
    rcm = np.copy(rcm)*constants.AU/1000      # Convert from AU to km
    s_radius = self.star_radius
    radii = self.radii
    eclipses = np.where((abs(rcm[:, 1:, 0]-rcm[:, 0, 0].reshape(-1, 1)) < s_radius + radii) & (rcm[:, 1:, 1] < 0))
    return eclipses

  def flux(self, t, times, planet):
    s_radius = self.star_radius
    radii = self.radii
    flux = np.ones(len(t))
    if 4 in planet:
      print('halp')
    flux[times] -= np.power(radii[planet]/s_radius, 2)
    return flux

import matplotlib.pyplot as plt

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

from ast2000tools.solar_system import SolarSystem

seed = 31526
system = SolarSystem(seed)

inst = Motion(system)
r, v, rcm, vcm, t = inst.solve(2E-5, 1_000_000)
labels = ['Star'] + ['Planet {}'.format(i) for i in inst.indexes]
orbits(rcm, labels)
vcm *= constants.AU/constants.yr # Convert from AU/yr to m/s
vels(t, vcm[:, 0])

### Radial velocity
# Peculiar velocity of the star, [m/s]
v_pec = 316_527
# Inclination of orbit is 60 degrees
i = np.pi/3
# Radial velocity is measured i y-direction
v_y = vcm[:, 0, 1]
sigma = np.amax(v_y)/5
noise = np.random.normal(loc=0, scale=sigma, size=len(v_y))
v_r = v_pec + v_y*np.sin(i)
radial_vel(t, v_r+noise)
radial_vel(t, v_r)
plt.savefig('../output/plots/star_radial_vel.pdf')
plt.clf()

### Flux
t *= constants.yr/3600     # Convert from years to hours
times, planet = inst.eclipse(rcm)
eclipse_plot(rcm, times, planet, labels)
flux = inst.flux(t, times, planet)
flux_plot(t, flux)
