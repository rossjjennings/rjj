import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    Angle, SkyCoord, EarthLocation,
    ICRS, PrecessedGeocentric,
)
import pint.observatory

def read_catalog(filename):
    """
    Read a Cleo-style coordinate catalog.
    """
    pulsar_coords = {}
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            elif line.split('=')[0].strip() in ['format', 'coordmode', 'head']:
                continue
            parts = line.split()
            psr = parts[0]
            ra = parts[3]
            dec = parts[4]
            coords = SkyCoord(ra, dec, unit=(u.hourangle, u.degree))
            pulsar_coords[psr] = coords
    return pulsar_coords

def calc_rise_set(source_coords, observatory, horizon, equinox=None):
    """
    Calculate rise and set times for a source.
    This is based on a simple spherical geometry calculation,
    including precession of the equinoxes but ignoring subtler effects.

    Parameters
    ----------
    source_coords: SkyCoord giving the source location.
    observatory: PINT observatory code or EarthLocation.
    horizon: Altitude of effective horizon (Quantity or Angle).
    equinox: Equinox for coordinate precession (as Time). If left at `None`,
             will use the current time. The special value 'ICRS' will make it
             use ICRS J2000 coordinates directly (this is only reasonable near
             for observing times near J2000.0).
    """
    if not hasattr(observatory, 'to_geodetic'):
        observatory = pint.observatory.get_observatory(observatory)
        observatory = observatory.earth_location_itrf()
    obs_coords = observatory.to_geodetic()
    if equinox is None:
        frame = PrecessedGeocentric(equinox=Time.now())
    elif hasattr(equinox, 'upper') and equinox.upper() == 'ICRS':
        frame = ICRS
    else:
        frame = PrecessedGeocentric(equinox=Time(equinox, format='jyear'))
    source_coords = source_coords.transform_to(frame)
    ratio = (
        (np.sin(horizon) - np.sin(obs_coords.lat)*np.sin(source_coords.dec)) /
        (np.cos(obs_coords.lat)*np.cos(source_coords.dec))
    )
    if ratio < -1: # source is always above horizon
        rise_time = source_coords.ra - Angle(180, u.deg)
        set_time = source_coords.ra + Angle(180, u.deg)
    elif ratio > 1: # source is never above horizon
        rise_time = source_coords.ra
        set_time = source_coords.ra
    else:
        rise_time = source_coords.ra - np.arccos(ratio)
        set_time = source_coords.ra + np.arccos(ratio)
    return rise_time, set_time

def rise_set_table(source_coords, observatory, horizon, equinox=None):
    """
    Given a dictionary mapping source names to coordinates, make a markdown
    table of rise and set times for the sources.
    """
    table = (
        "| Pulsar     | Rise time| Set time | Up for   |\n"
        "| ---------- | -------- | -------- |----------|\n"
    )
    for source, coords in source_coords.items():
        rise_time, set_time = calc_rise_set(coords, observatory, horizon, equinox)
        diff = set_time - rise_time
        rise_time = rise_time % Angle(360, u.deg)
        set_time = set_time % Angle(360, u.deg)
        rise_time = rise_time.to_string(u.hourangle, sep=':', pad=True, precision=0)
        set_time = set_time.to_string(u.hourangle, sep=':', pad=True, precision=0)
        diff = diff.to_string(u.hourangle, sep=':', pad=True, precision=0)

        table += f"| {source} | {rise_time} | {set_time} | {diff} |\n"
    return table

def calc_equivalent_coords(start_lst, end_lst, observatory, horizon, equinox=None):
    """
    Inverse of calc_rise_set(): Calculate the sky coordinates that are above
    the horizon between the specified LSTs.

    Parameters
    ----------
    start_lst: Initial LST, as a Quantity or Angle.
    end_lst: Final LST, as a Quantity or Angle.
    observatory: PINT observatory code or EarthLocation.
    horizon: Altitude of effective horizon (Quantity or Angle).
    equinox: Equinox for coordinate precession (as Time). If left at `None`,
             will use the current time. The special value 'ICRS' will make it
             use ICRS J2000 coordinates directly (this is only reasonable near
             for observing times near J2000.0).
    """
    if not hasattr(observatory, 'to_geodetic'):
        observatory = pint.observatory.get_observatory(observatory)
        observatory = observatory.earth_location_itrf()
    obs_coords = observatory.to_geodetic()
    if equinox is None:
        frame = PrecessedGeocentric(equinox=Time.now())
    elif hasattr(equinox, 'upper') and equinox.upper() == 'ICRS':
        frame = ICRS
    else:
        frame = PrecessedGeocentric(equinox=Time(equinox, format='jyear'))
    span = (end_lst - start_lst) % (360*u.deg)
    ra = (start_lst + span/2) % (360*u.deg)
    sign_lat = np.sign(obs_coords.lat)
    discriminant = (
        np.cos(span/2)**2*np.cos(obs_coords.lat)**2
        + np.sin(obs_coords.lat)**2 - np.sin(horizon)**2
    )
    dec = 2*np.arctan(
        (np.sin(obs_coords.lat) - sign_lat*discriminant)
        /(np.cos(span/2)*np.cos(obs_coords.lat) + np.sin(horizon))
    )
    dec = Angle(dec.to(u.deg))
    return SkyCoord(ra=ra, dec=dec, frame=frame)
