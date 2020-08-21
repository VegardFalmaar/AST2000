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
