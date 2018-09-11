import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
import astropy.units as u
from astropy.coordinates import Galactic, ICRS, BarycentricTrueEcliptic
from collections import namedtuple

celestial_coords = namedtuple('celestial_coords', ('ra', 'dec'))
ecliptic_coords = namedtuple('ecliptic_coords', ('lon', 'lat'))
galactic_coords = namedtuple('galactic_coords', ('l', 'b'))

celestial_coords_w_pm = namedtuple('celestial_coords_w_pm',
                                   ('ra', 'dec', 'pm_ra', 'pm_dec'))
ecliptic_coords_w_pm = namedtuple('ecliptic_coords_w_pm',
                                   ('lon', 'lat', 'pm_lon', 'pm_lat'))
galactic_coords_w_pm = namedtuple('galactic_coords_w_pm',
                                   ('l', 'b', 'pm_l', 'pm_b'))

def celestial_rad(coords):
    '''
    Return a named tuple containing celestial coordinates in radians.
    
    Input
    -----
    coords: An astropy `SkyCoord` or coordinate frame object.
    
    Outputs
    -------
    ra: The right ascension, in radians.
    dec: The declination, in radians.
    '''
    ra = coords.icrs.ra.to(u.radian).value
    dec = coords.icrs.dec.to(u.radian).value
    return celestial_coords(ra, dec)

def ecliptic_rad(coords):
    '''
    Return a named tuple containing ecliptic coordinates in radians.
    
    Input
    -----
    coords: An astropy `SkyCoord` or coordinate frame object.
    
    Outputs
    -------
    lon: The ecliptic longitude, in radians.
    lat: The ecliptic latitude, in radians.
    '''
    lon = coords(psr).barycentrictrueecliptic.lon.to(u.radian).value
    lat = coords(psr).barycentrictrueecliptic.lat.to(u.radian).value
    return ecliptic_coords(lon, lat)

def galactic_rad(coords):
    '''
    Return a named tuple containing galactic coordinates in radians.
    
    Input
    -----
    coords: An astropy `SkyCoord` or coordinate frame object.
    
    Outputs
    -------
    l: The galactic longitude, in radians.
    b: The galactic latitude, in radians.
    '''
    l = coords(psr).galactic.l.to(u.radian).value
    b = coords(psr).galactic.b.to(u.radian).value
    return galactic_coords(l, b)

def from_celestial_pm(coords, pm_ra, pm_dec):
    return ICRS(ra = coords.icrs.ra, dec = coords.icrs.dec,
                pm_ra_cosdec = pm_ra, pm_dec = pm_dec)

def from_ecliptic_pm(coords, pm_lon, pm_lat):
    return BarycentricTrueEcliptic(lon = coords.barycentrictrueecliptic.lon,
                                   lat = coords.barycentrictrueecliptic.lat,
                                   pm_lon_coslat = pm_lon, pm_lat = pm_lat)

def from_galactic_pm(coords, pm_l, pm_b):
    galactic_coords
    return Galactic(l = coords.galactic.l, b = coords.galactic.b,
                    pm_l_cosb = pm_l, pm_b = pm_b)

def celestial_pm(coords_pm):
    return coords_pm.

def ecliptic_pm(coords_pm):
    

def galactic_pm(coords_pm):
    

