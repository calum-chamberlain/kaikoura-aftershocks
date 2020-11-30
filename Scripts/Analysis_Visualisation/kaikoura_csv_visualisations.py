import pandas as pd
import numpy as np
import os
import copy
import datetime as dt

from collections import namedtuple
from eqcorrscan.utils.plotting import freq_mag
from progressbar import ProgressBar
from eqcorrscan.utils.mag_calc import calc_b_value, calc_max_curv

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, date2num

from gps_data_play import GPSStation

home = os.path.expanduser("~")

FAULT_FILE = f"{home}/.gmt_data/faults_NZ_WGS84.gmt"
MAINSHOCK = (-42.626466, 172.990578, 12.390625)  # My NLL location.

DEFAULT_XSECTIONS = {"1": [(-42.76, 172.824), (-41.572, 174.637)]}

def _get_grid_xsections(
    strike: float = 318,
    start: tuple = (-43.0, 173.127),
    spacing: float = 10.,
    length: float = 60.,
    n: int = 20
):
    import string
    from cjc_utilities.coordinates.coordinates import Location, Geographic

    origin = Geographic(latitude=start[0], longitude=start[1], depth=0)

    xsections = dict()
    for i in range(n):
        name = string.ascii_lowercase[i]
        _start = Location(x=spacing * i, y=0, z=0, 
                          origin=origin, strike=strike, dip=90.).to_geographic()
        _end = Location(x=spacing * i, y=length, z=0, 
                        origin=origin, strike=strike, dip=90.).to_geographic()
        xsections.update({name: [(_start.latitude, _start.longitude),
                                 (_end.latitude, _end.longitude)]})
    return xsections


DEFAULT_XSECTIONS.update(_get_grid_xsections())


def get_williams_contours():
    import netCDF4 as nc

    ds = nc.Dataset(f"{home}/.gmt_data/Williams_Hikurangi.grd")
    lats, lons, depths = ds["y"][:], ds["x"][:], ds["z"][:]
    return lats, lons, depths


# Scale bar functions from https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot/50674451#50674451
def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    import cartopy.crs as ccrs

    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment  from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(f"End is closer to start ({initial_distance}) than "
                         f"given distance ({distance}).")

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    import cartopy.geodesic as cgeo
    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys).base[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(ax, location, length, metres_per_unit=1000, unit_name='km',
              tol=0.01, angle=0, color='black', linewidth=3, text_offset=0.005,
              ha='center', va='bottom', plot_kwargs=None, text_kwargs=None,
              **kwargs):
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Args:
        ax:              CartoPy axes.
        location:        Position of left-side of bar in axes coordinates.
        length:          Geodesic length of the scale bar.
        metres_per_unit: Number of metres in the given unit. Default: 1000
        unit_name:       Name of the given unit. Default: 'km'
        tol:             Allowed relative error in length of bar. Default: 0.01
        angle:           Anti-clockwise rotation of the bar.
        color:           Color of the bar and text. Default: 'black'
        linewidth:       Same argument as for plot.
        text_offset:     Perpendicular offset for text in axes coordinates.
                         Default: 0.005
        ha:              Horizontal alignment. Default: 'center'
        va:              Vertical alignment. Default: 'bottom'
        **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
        **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
        **kwargs:        Keyword arguments for both plot and text.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {'linewidth': linewidth, 'color': color, **plot_kwargs,
                   **kwargs}
    text_kwargs = {'ha': ha, 'va': va, 'rotation': angle, 'color': color,
                   **text_kwargs, **kwargs}

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad,
                            tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(*text_location, f"{length} {unit_name}", rotation_mode='anchor',
            transform=ax.transAxes, **text_kwargs)


def read_faults(
    min_lat: float = -90, 
    max_lat: float = 90,
    min_lon: float = -360,
    max_lon: float = 360):
    with open(FAULT_FILE, "r") as f:
        lines = f.read().splitlines()
    faults, fault = [], []
    for line in lines[4:]:
        if line == ">":
            if len(fault) > 0:
                faults.append(fault)
                fault = []
        else:
            line = line.split()
            lon, lat = float(line[0]), float(line[1])
            if lon < max_lon and lon > min_lon and lat < max_lat and lat > min_lat:
                fault.append((lon, lat))
    if len(fault) > 0:
        faults.append(fault)
    return faults


def get_kaikoura_faults():
    print("Faults downloaded from NZ Active Fault DB. Reference: \n\t"
          "Langridge, R.M., Ries, W.F., Litchfield, N.J., Villamor, P., "
          "Van Dissen, R.J., Barrell, D.J.A., Rattenbury, M.S., Heron, D.W.,"
          " Haubrock, S., Townsend, D.B., Lee, J.M., Berryman, K.R., "
          "Nicol, A., Cox, S.C., Stirling, M.W. (2016). The New Zealand "
          "Active Faults Database. New Zealand Journal of Geology and "
          "Geophysics 59: 86-96. doi: 0.1080/00288306.2015.1112818.")
    with open(f"{home}/.gmt_data/NZAFD_Kaikoura_250K_traces_WGS84_Aug_2020.txt", "r") as f:
        lines = f.read().splitlines()
    faults, fault, fault_name = dict(), [], None
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) > 3:
            # New fault
            if len(fault) > 0 and fault_name:
                faults.update({fault_name: fault})
                fault, fault_name = [], None
            fault_name = f"{parts[1]}_{parts[0]}"
        elif parts[0] == "END":
            continue
        else:
            fault.append((float(parts[0]), float(parts[1])))
    if len(fault) > 0 and fault_name:
        faults.update({fault_name: fault})
    return faults


def filter_earthquakes(
    earthquakes: pd.DataFrame,
    min_latitude: float = None,
    min_longitude: float = None,
    max_latitude: float = None,
    max_longitude: float = None,
    min_depth: float = None,
    max_depth: float = None,
    starttime: dt.datetime = None,
    endtime: dt.datetime = None,
    min_mag: float = None,
    max_mag: float = None,
):
    quakes = earthquakes.copy()
    if max_depth:
        quakes = quakes[quakes.depth <= max_depth * 1000]
    if min_depth:
        quakes = quakes[quakes.depths >= min_depth * 1000]
    if min_longitude:
        quakes = quakes[quakes.longitude >= min_longitude]
    if max_longitude:
        quakes = quakes[quakes.longitude <= max_longitude]
    if min_latitude:
        quakes = quakes[quakes.latitude >= min_latitude]
    if max_latitude:
        quakes = quakes[quakes.latitude <= max_latitude]
    if starttime:
        quakes = quakes[quakes.origintime >= starttime]
    if endtime:
        quakes = quakes[quakes.origintime <= endtime]
    if min_mag:
        quakes = quakes[quakes.magnitude >= min_mag]
    if max_mag:
        quakes = quakes[quakes.magnitude <= max_mag]
    return quakes



def calculate_moving_b(
    earthquakes: pd.DataFrame,
    window_size: int = 1000, 
    mc: float = None,
    min_residual: float = 98.0, 
    min_events: int = 300,
):
    # Make sure indexing makes sense
    earthquakes = earthquakes.sort_values(by="origintime", ignore_index=True)
    # Get a minimal dataframe as required 
    magnitude_times = pd.concat( 
        [earthquakes["origintime"], earthquakes["magnitude"]], axis=1)
    magnitude_times.set_index(np.arange(len(magnitude_times)), inplace=True)
    # Work out the window length and mid-point 
    seconds_offset = ( 
        magnitude_times.origintime -  
        magnitude_times.origintime[0]).dt.total_seconds().astype(np.int32)
    magnitude_times = magnitude_times.merge( 
        seconds_offset.rename( 
            "seconds_offset"), left_index=True, right_index=True) 
    window_median = (
        magnitude_times.seconds_offset.rolling(window_size).median(
            ).fillna(0).astype(np.int32))
    window_median = pd.to_timedelta(window_median, unit="S") # Unit is seconds 
    window_median += magnitude_times.origintime[0] 
    magnitude_times = magnitude_times.merge( 
        window_median.rename("window_median"), left_index=True, right_index=True) 

    mc_max_curv = np.ones(len(magnitude_times.magnitude)) * np.nan
    bvalues = np.ones_like(mc_max_curv) * np.nan


    bar = ProgressBar(max_value=len(magnitude_times.magnitude) - window_size)
    for i in range(len(magnitude_times.magnitude) - window_size): 
        mags = [m for m in magnitude_times.magnitude[i: i + window_size] 
                if not np.isnan(m)]
        if mc is None:
            _mc = calc_max_curv(mags, bin_size=0.1)
        else:
            _mc = mc
        bvals = calc_b_value(
            mags, completeness=np.arange(_mc - 0.75, _mc + 3.5, 0.05), 
            plotvar=False)
        best_index = np.argmax(list(zip(*bvals))[2]) 
        if bvals[best_index][2] >= min_residual:
            if int(bvals[best_index][-1]) >= min_events:
                # Only report a bvalue for enough events
                bvalues[i + window_size] = bvals[best_index][1]
            # Record the completeness whatever.
            mc_max_curv[i + window_size] = bvals[best_index][0]
        bar.update(i)
    bar.finish()

    mc_max_curv = pd.Series(mc_max_curv) 
    bvalues = pd.Series(bvalues)

    magnitude_times = magnitude_times.merge( 
        mc_max_curv.rename("mc_max_curv"), left_index=True, right_index=True) 
    magnitude_times = magnitude_times.merge( 
        bvalues.rename("bvalues"), left_index=True, right_index=True)
    return magnitude_times


def shade(located_elevations):
    """
    Given an array of elevations in a LocatedImage, add a relief (shadows) to
    give a realistic 3d appearance.

    """
    from cartopy.io import srtm, LocatedImage
    new_img = srtm.add_shading(located_elevations.image,
                               azimuth=135, altitude=15)
    return LocatedImage(new_img, located_elevations.extent)


def access_srtm():
    """ Set up account for SRTM. """
    from http.cookiejar import CookieJar
    from urllib.parse import urlencode
    import urllib.request
    import getpass

    # The user credentials that will be used to authenticate access to the data

    # The url of the file we wish to retrieve

    # url = "http://e4ftl01.cr.usgs.gov/MOLA/MYD17A3H.006/2009.01.01/MYD17A3H.A2009001.h12v05.006.2015198130546.hdf.xml"

    # Create a password manager to deal with the 401 reponse that is returned from
    # Earthdata Login
    print("Enter your username and password for earthdata.nasa")
    username = input("Username: ")
    passwd = getpass.getpass()

    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(
        None, "https://urs.earthdata.nasa.gov", username, passwd)

    # Create a cookie jar for storing cookies. This is used to store and return
    # the session cookie given to use by the data server (otherwise it will just
    # keep sending us back to Earthdata Login to authenticate).  Ideally, we
    # should use a file based cookie jar to preserve cookies between runs. This
    # will make it much more efficient.

    cookie_jar = CookieJar()

    # Install all the handlers.

    opener = urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(password_manager),
        #urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
        #urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
        urllib.request.HTTPCookieProcessor(cookie_jar))
    urllib.request.install_opener(opener)


def plot_fm(
    strike: float,
    dip: float,
    rake: float,
    longitude: float,
    latitude: float,
    axes,
    color: str = "k",
    width: float = 30,
    edgecolor: str = "k",
    alpha: float = 1.0,
    linewidth: float = 0.3,
    zorder: int = 10,
    rasterize: bool = True,
):
    import cartopy.crs as ccrs
    from matplotlib import collections, transforms
    from obspy.imaging.beachball import plot_dc, NodalPlane

    np1 = NodalPlane(strike=strike, dip=dip, rake=rake)
    xy = (longitude, latitude)
    # xy = ccrs.PlateCarree().transform_point(
    #     x=longitude, y=latitude, src_crs=axes.projection)
    colors, p = plot_dc(np1, size=100, xy=xy, width=width)

    col = collections.PatchCollection(p, match_original=False)
    fc = [color if c == 'b' else "w" for c in colors]
    col.set_facecolors(fc)
    col.set_transform(transforms.IdentityTransform())
    # Next is a dirty hack to fix the positioning:
    # 1. Need to bring the all patches to the origin (0, 0).
    for p in col._paths:
        p.vertices -= xy
    # # 2. Then use the offset property of the collection to position the
    # #    patches
    col.set_offsets((longitude, latitude))
    # # col.set_offsets(xy)
    col._transOffset = ccrs.PlateCarree()

    col.set_edgecolor(edgecolor)
    col.set_alpha(alpha)
    col.set_linewidth(linewidth)
    col.set_zorder(zorder)

    # Rasterize
    if rasterize:
        col.set_rasterized(True)

    # Add collection
    axes.add_collection(col)
    return


def rake_cmap(alpha=1):
    """ A color map for rake. """
    from matplotlib.colors import ListedColormap

    N = 256
    vals = np.zeros((N, 4))
    vals[:, 3] = alpha
    vals[:, 2] = np.linspace(0, 1, N) ** 2
    blues = vals
    blues_r = blues[::-1]

    vals = np.zeros((N, 4))
    vals[:, 3] = alpha
    vals[:, 0] = np.linspace(0, 1, N) ** 2
    reds = vals
    reds_r = reds[::-1]

    newcolors = np.vstack((blues, blues_r, reds, reds_r))
    return ListedColormap(newcolors)



# TODO: size by m (optional), fault-focus zooms
def plot_locations(
    earthquakes: pd.DataFrame, 
    color_by: str = "depth",
    rotated: bool = True,
    max_depth: float = 40.0,
    min_depth: float = 0.0,
    min_latitude: float = -43.2,
    min_longitude: float = 172.5,
    max_latitude: float = -41.5,
    max_longitude: float = 174.7,
    starttime: dt.datetime = None,
    endtime: dt.datetime = None,
    logarithmic_color: bool = False,
    size: float = 0.2,
    scale_bar_length: float = 20,
    colormap: str = "plasma_r",
    cross_sections: dict = DEFAULT_XSECTIONS,
    cross_section_kwargs: dict = {"dip": 90.0, "swath_half_width": 10.0},
    relief: bool = False,  # Doesn't work at the moment
    focal_mechanisms: bool = False,
    plot_stations: bool = True,
    rupture_color: str = "cyan",
):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.io.srtm import SRTM3Source
    from cartopy.io import PostprocessedRasterSource
    from cjc_utilities.animator.animator import _blank_map

    assert {"magnitude", "latitude", "longitude", 
            "depth"}.issubset(set(earthquakes.columns))
    if "time" in earthquakes.columns:
        earthquakes = earthquakes.rename(columns={"time": "origintime"})
    assert "origintime" in earthquakes.columns

    if focal_mechanisms:
        assert {"strike", "dip", "rake"}.issubset(set(earthquakes.columns))
        if rotated:
            print("Cannot plot focal mechanisms on rotated map at the moment")

    assert color_by.lower() in ("depth", "time", "timestamp", "rake"), \
        "color_by must be depth, time or timestamp"

    if relief:
        access_srtm()

    if rotated:
        projection = ccrs.RotatedPole
        proj_kwargs = dict(
            pole_longitude=60.0, pole_latitude=60.0, # 60, 60
            central_rotated_longitude=170.0)
        figsize = (15, 5)
    else:
        if relief:
            projection = ccrs.PlateCarree
        else:
            projection = "local"
        proj_kwargs = dict()
        figsize = (10, 10.5)

    quakes = filter_earthquakes(
        earthquakes, min_longitude=min_longitude, max_longitude=max_longitude,
        min_latitude=min_latitude, max_latitude=max_latitude,
        min_depth=min_depth, max_depth=max_depth, starttime=starttime,
        endtime=endtime)

    quakes = quakes.sort_values(
        by="origintime", ignore_index=True, ascending=False)
    
    lats, lons, depths, times = (
        quakes.latitude.to_numpy(), quakes.longitude.to_numpy(),
        quakes.depth.to_numpy() / 1000., quakes.origintime.to_list())

    if size is None:
        size = quakes.magnitude ** 2
        in_size = None
    else:
        in_size = size # Used for x-section

    starttime = starttime or min(times)
    if color_by == "timestamp":
        times = [(t - starttime).total_seconds() for t in times]
    else:
        times = [date2num(t.to_pydatetime()) for t in times]

    if color_by.lower() == "depth":
        colors = depths
    elif color_by.lower() in ("time", "timestamp"):
        colors = times
    elif color_by.lower() == "rake" and focal_mechanisms:
        colors = quakes.rake.to_list()
    
    if logarithmic_color:
        norm = LogNorm(vmin=min(colors), vmax=max(colors))
    elif color_by == "depth":
        norm = Normalize(vmin=min(colors), vmax=20.0)
        colormap = copy.copy(plt.get_cmap(colormap))
        colormap.set_over(color="k")
    elif color_by == "rake":
        norm = Normalize(vmin=-180, vmax=180)
        colormap = rake_cmap()

    else:
        norm = Normalize(vmin=min(colors), vmax=max(colors))

    fig, map_ax, cbar_ax, cb = _blank_map(
        lats=lats, lons=lons, color=colors, projection=projection, 
        resolution="full", colormap=colormap, figsize=figsize, 
        proj_kwargs=proj_kwargs, norm=norm, continent_fill_color="0.65",
        water_fill_color="0.9")

    if relief:
        shaded_srtm = PostprocessedRasterSource(
            SRTM3Source(max_nx=10, max_ny=10), shade)
        map_ax.add_raster(shaded_srtm, cmap='Greys')
    
    min_latitude = min_latitude or min(lats) - 1.0
    max_latitude = max_latitude or max(lats) + 1.0
    min_longitude = min_longitude or min(lons) - 1.0
    max_longitude = max_longitude or max(lons) + 1.0
    
    # Draw gridlines
    gl = map_ax.gridlines(
        draw_labels=True, dms=False, x_inline=False, y_inline=False)

    # Plot earthquakes
    if not color_by.lower() == "rake":
        map_ax.scatter(
            lons, lats, marker="o", s=size, c=colors, zorder=10, alpha=0.9,
            cmap=colormap, transform=ccrs.PlateCarree(), rasterized=True,
            norm=norm)
    else:
        # Just plot in black
        map_ax.scatter(
            lons, lats, marker="o", s=size, c="k", zorder=10, alpha=0.9,
            cmap=colormap, transform=ccrs.PlateCarree(), rasterized=True,
            norm=norm)
    
    # Plot mainshock
    mainshock = map_ax.scatter(
        MAINSHOCK[1], MAINSHOCK[0], marker="*", facecolor="gold", 
        edgecolor="k", s=200.0, zorder=11, transform=ccrs.PlateCarree())

    # Plot Focal mechanisms
    if focal_mechanisms:
        s, d, r, lat, lon, _depth = (
                quakes.strike.to_list(), 
                quakes.dip.to_list(),
                quakes.rake.to_list(),
                quakes.latitude.to_list(),
                quakes.longitude.to_list(),
                quakes.depth.to_list())
        for i in range(len(quakes)):
            if np.any(np.isnan((s[i], d[i], r[i]))):
                continue
            if color_by == "rake":
                color = r[i]
            else:
                color = _depth[i] / 1000.
            red, green, blue, _alpha = colormap(norm(color))
            plot_fm(strike=s[i], dip=d[i], rake=r[i],
                    latitude=lat[i], longitude=lon[i], zorder=20, axes=map_ax,
                    width=15, color=(red, green, blue), alpha=0.6)
    
    # Plot Faults
    faults = read_faults(
        min_lat=min_latitude - 2, max_lat=max_latitude + 2,
        min_lon=min_longitude - 2, max_lon=max_longitude + 2)
    for fault in faults:
        flons, flats = zip(*fault)
        f_line, = map_ax.plot(
            flons, flats, color="k", linewidth=1.5, zorder=8,
            transform=ccrs.PlateCarree(), rasterized=True)

    # Plot Kaikoura ruptures
    try:
        kaikoura_faults = get_kaikoura_faults()
    except FileNotFoundError:
        print("Could not find Kaikoura faults, skipping")
        kaikoura_faults = None
    if kaikoura_faults:
        for fault in kaikoura_faults.values():
            flons, flats = zip(*fault)
            kaik_f_line, = map_ax.plot(
                flons, flats, color=rupture_color, linewidth=2.0, zorder=9,
                transform=ccrs.PlateCarree(), rasterized=True)

    if plot_stations:
        # Plot GPS stations
        gps_stations = pd.read_csv(
            "gps_station_locations.csv", parse_dates=["Start", "End"])
        gps_markers = map_ax.scatter(
            gps_stations.Longitude, gps_stations.Latitude, facecolor="lime",
            edgecolor="k", marker="^", zorder=10, transform=ccrs.PlateCarree(), 
            s=150)
        for lat, lon, code in zip(gps_stations.Latitude, gps_stations.Longitude, gps_stations["Station code"]):
            map_ax.text(lon + 0.01, lat - 0.01, code, transform=ccrs.PlateCarree(), 
                        zorder=100, clip_on=True)
        
        # Plot Seismographs
        seismographs = pd.read_csv(
            "seismograph_locations.csv", parse_dates=["Start", "End"])
        seismograph_markers = map_ax.scatter(
            seismographs.Longitude, seismographs.Latitude, facecolor="orange",
            edgecolor="k", marker="v", zorder=10, transform=ccrs.PlateCarree(),
            s=200)
        for lat, lon, code in zip(seismographs.Latitude, seismographs.Longitude, seismographs["Station code"]):
            map_ax.text(lon + 0.01, lat - 0.01, code, transform=ccrs.PlateCarree(),
                        zorder=100, clip_on=True)


    if not rotated:
        map_ax.set_extent(
            [min_longitude, max_longitude, min_latitude, max_latitude],
            crs=ccrs.PlateCarree())
    else:
        rp = projection(**proj_kwargs)
        xs, ys, zs = rp.transform_points(
            ccrs.PlateCarree(), 
            np.array([min_longitude, max_longitude]),
            np.array([min_latitude, max_latitude])).T
        map_ax.set_xlim((xs[0] + .2, xs[1]))
        map_ax.set_ylim((ys[0] - 0.2, ys[1] + 0.2))
        map_ax.set_aspect(1.4)  # Stretch - there should be a clever way to do this.
        # map_ax.set_ylim(ys)

    if color_by == "timestamp":
        cbar_ax.set_xlabel(f"Time from {starttime}")
        cb.set_ticks([10, 3600, 86400, 864000, 8640000, 365 * 86400])
        cbar_ax.set_xticklabels(
            ["10 s", "1 hr", "1 day", "10 days", "100 days", "1 year"])
    elif color_by == "rake":
        cbar_ax.set_xlabel('Rake$^\circ$')
        cb.set_ticks([-180, -90, 0, 90, 180])

    # Plot x-section lines
    if cross_sections:
        x_sections = dict()
        for x_section_name, x_section_points in cross_sections.items():
            x_start, x_end = x_section_points
            x_lats, x_lons = [x_start[0], x_end[0]], [x_start[1], x_end[1]]
            map_ax.plot(x_lons, x_lats, color="red", linewidth=1.0, zorder=10,
                        transform=ccrs.PlateCarree(), linestyle="--")
            # Label
            map_ax.text(x_start[1], x_start[0], s=x_section_name, 
                        transform=ccrs.PlateCarree())
            map_ax.text(x_end[1], x_end[0], s=f"{x_section_name}'",
                        transform=ccrs.PlateCarree())
            x_section = plot_x_section(
                earthquakes=quakes, start_latitude=x_start[0], 
                start_longitude=x_start[1], end_latitude=x_end[0], 
                end_longitude=x_end[1], max_depth=max_depth, starttime=starttime,
                endtime=endtime, logarithmic_color=logarithmic_color, 
                size=in_size, colormap=colormap, color_by=color_by,
                **cross_section_kwargs)
            if x_section is None:
                continue
            x_sections.update({x_section_name: x_section})
            x_sections[x_section_name].suptitle(
                f"{x_section_name} to {x_section_name}'")

    # Plot subduction contours
    subd_lats, subd_lons, subd_depths = get_williams_contours()
    lat_mask, lon_mask = (np.ones_like(subd_lats, dtype=bool), 
                          np.ones_like(subd_lons, dtype=bool))
    if min_latitude:
        lat_mask = np.logical_and(lat_mask, subd_lats >= min_latitude - 2)
    if max_latitude:
        lat_mask = np.logical_and(lat_mask, subd_lats <= max_latitude + 2)
    if min_longitude:
        lon_mask = np.logical_and(lon_mask, subd_lons >= min_longitude - 2)
    if max_longitude:
        lon_mask = np.logical_and(lon_mask, subd_lons <= max_longitude + 2)

    subd_lats = subd_lats[lat_mask]
    subd_lons = subd_lons[lon_mask]
    subd_depths = subd_depths[lat_mask][:, lon_mask] * -1
    contours = map_ax.contour(
        subd_lons, subd_lats, subd_depths, colors="k", linestyles="dashed",
        transform=ccrs.PlateCarree(), levels=[0, 5, 10, 15, 20, 25, 30, 40, 50])
    map_ax.clabel(contours, inline=1, fontsize=10, fmt="%i km")

    # Plot scale bar
    scale_bar(map_ax, (0.05, 0.1), scale_bar_length, angle=0)
    scale_bar(map_ax, (0.05, 0.1), scale_bar_length, angle=90)

    handles = [f_line, mainshock, contours.collections[0]]
    labels = ["Active Faults", "Mainshock", "Williams et al. Interface"]
    if kaik_f_line:
        handles.append(kaik_f_line)
        labels.append("Surface Rupture")
    if plot_stations:
        handles.extend([gps_markers, seismograph_markers])
        labels.extend(["cGPS", "Seismograph"])

    fig.legend(handles=handles, labels=labels, framealpha=1.0)

    if cross_sections:
        return fig, x_sections
    return fig


def get_origin_strike_length(
    start_latitude: float,
    start_longitude: float,
    end_latitude: float,
    end_longitude: float,
):
    from math import atan, degrees
    from cjc_utilities.coordinates.coordinates import Location, Geographic
    
    origin = Geographic(latitude=start_latitude, longitude=start_longitude,
                        depth=0.0)
    start = origin.to_xyz(origin=origin, strike=0, dip=90)
    assert start.x == 0.0 and start.y == 0.0 and start.z == 0.0
    end = Geographic(latitude=end_latitude, longitude=end_longitude, 
                     depth=0.0).to_xyz(origin=origin, strike=0, dip=90)
    length = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** .5
    assert length > 0, "Start and end are the same"
    
    if end.x > 0 and end.y > 0:
        strike = degrees(atan(end.x / end.y))
    elif end.x > 0 and end.y < 0:
        strike = 90 + degrees(atan(abs(end.y) / end.x))
    elif end.x < 0 and end.y < 0:
        strike = 180 + degrees(atan(abs(end.x) / abs(end.y)))
    elif end.x < 0 and end.y > 0:
        strike = 270 + degrees(atan(end.y / abs(end.x)))
    # Cases at compass points
    elif end.x > 0 and end.y == 0:
        strike = 90.0
    elif end.x == 0 and end.y > 0:
        strike = 0.0
    elif end.x < 0 and end.y == 0:
        strike = 270.0
    elif end.x == 0 and end.y < 0:
        strike = 180.0
    else:
        raise NotImplementedError("Could not calculate strike")

    return origin, strike, length


def project_onto_section(
    earthquakes: pd.DataFrame,
    start_latitude: float,
    start_longitude: float,
    end_latitude: float,
    end_longitude: float,
    max_depth: float = 50.0,
    swath_half_width: float = 30.0,
    dip: float = 90.0,
):
    from cjc_utilities.coordinates.coordinates import Location, Geographic
    from cjc_utilities.coordinates.extract_cross_section import get_plane

    origin, strike, length = get_origin_strike_length(
        start_latitude=start_latitude, start_longitude=start_longitude,
        end_latitude=end_latitude, end_longitude=end_longitude)

    # Get the x-section plane
    x_section = get_plane(origin=origin, strike=strike, dip=dip, length=length,
                          height=-1 * max_depth)

    # Project   
    lats, lons, depths, times, magnitudes = (
        earthquakes.latitude.to_numpy(), earthquakes.longitude.to_numpy(),
        -1 * (earthquakes.depth.to_numpy() / 1000.), earthquakes.origintime.to_list(), 
        earthquakes.magnitude.to_numpy())
    # Keep focal mechanism if there
    s1, d1, r1 = None, None, None  # Set for later comparison
    if {"strike", "dip", "rake"}.issubset(set(earthquakes.columns)):
        s1, d1, r1 = (
            earthquakes.strike.to_numpy(), earthquakes.dip.to_numpy(),
            earthquakes.rake.to_numpy())
    projected = [
        Geographic(latitude=lat, longitude=lon, depth=depth, 
                   time=time, magnitude=mag).to_xyz(
            origin=origin, strike=strike, dip=dip) 
        for lat, lon, depth, time, mag in zip(lats, lons, depths, times, magnitudes)]
    # x is strike perpendicular, y strike parallel
    # projected = [
    #     loc for loc in projected 
    #     if abs(loc.x) <= swath_half_width and loc.y >= 0.0 and loc.y <= length]
    
    # Add strike, dip and rake in if available
    NodalPlane = namedtuple("NodalPlane", ("strike", "dip", "rake"))
    _projected = []
    for i, loc in enumerate(projected):
        # x is strike perpendicular, y strike parallel
        if abs(loc.x) <= swath_half_width and loc.y >= 0.0 and loc.y <= length:
            if s1 is not None:
                loc.nodal_plane = NodalPlane(s1[i], d1[i], r1[i])
            _projected.append(loc)
    projected = _projected
    if len(projected) == 0:
        return None
    # Sort so oldest plot on top
    projected.sort(key=lambda loc: loc.time, reverse=True)
    return projected


def plot_x_section(
    earthquakes: pd.DataFrame,
    start_latitude: float,
    start_longitude: float,
    end_latitude: float,
    end_longitude: float,
    max_depth: float = 50.0,
    swath_half_width: float = 30.0,
    dip: float = 90.0,
    starttime: dt.datetime = None,
    endtime: dt.datetime = None,
    logarithmic_color: bool = False,
    size: float = 0.2,
    colormap: str = "plasma_r",
    color_by: str = "time",
    plot_mainshock: bool = False,
    focal_mechanisms: bool = False,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
):
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp2d
    from cjc_utilities.coordinates.coordinates import Location, Geographic
    from rotate_nodal_planes import rot_meca, Mechanism, NodalPlane
    from obspy.imaging.beachball import beach, aux_plane
    
    assert {"magnitude", "latitude", "longitude", 
            "depth"}.issubset(set(earthquakes.columns))
    if "time" in earthquakes.columns:
        earthquakes = earthquakes.rename(columns={"time": "origintime"})
    assert "origintime" in earthquakes.columns

    # sort by time
    earthquakes = earthquakes.sort_values(by="origintime", ignore_index=True)

    if color_by not in ["time", "timestamp"]:
        print("Coloring by distance from x-section")
        color_by = "distance"
    
    quakes = filter_earthquakes(
        earthquakes, max_depth=max_depth, starttime=starttime,
        endtime=endtime)
    
    origin, strike, length = get_origin_strike_length(
        start_latitude=start_latitude, start_longitude=start_longitude,
        end_latitude=end_latitude, end_longitude=end_longitude)
    
    projected = project_onto_section(
        earthquakes=quakes, start_latitude=start_latitude, 
        start_longitude=start_longitude, end_latitude=end_latitude,
        end_longitude=end_longitude, swath_half_width=swath_half_width,
        dip=dip)
    x, y, z, times, mags = zip(*[(loc.x, loc.y, loc.z * -1, 
                                  loc.time, loc.magnitude) 
                           for loc in projected])
    
    if focal_mechanisms:
        assert hasattr(projected[0], "nodal_plane"), "Nodal Plane not found"
        s1, d1, r1 = zip(*[(loc.nodal_plane.strike, loc.nodal_plane.dip, 
                            loc.nodal_plane.rake) for loc in projected])
    
    # Plot!
    if fig and not ax:
        ax = fig.gca()
    elif not fig and ax:
        fig = ax.get_figure()
    elif not fig and not ax:
        fig, ax = plt.subplots()

    starttime = starttime or min(times)
    if color_by == "timestamp":
        times = [(t - starttime).total_seconds() for t in times]
        min_color, max_color = 1e-10, max(times)
    else:
        times = [date2num(t.to_pydatetime()) for t in times]
    
    if color_by.lower() == "distance":
        colors = x
        min_color, max_color = -1 * swath_half_width, swath_half_width
    elif color_by.lower() in ("time", "timestamp"):
        colors = times
        if color_by.lower() == "time":
            min_color, max_color = min(times), max(times)

    if logarithmic_color:
        print(f"Min color is {min_color}")
        norm = LogNorm(vmin=20, vmax=max_color)
    else:
        norm = Normalize(vmin=min_color, vmax=max_color)

    # Plot earthquakes
    size = size or np.array(mags) ** 2
    alpha = 0.9
    if focal_mechanisms:
        alpha = 0.8
    scatters = ax.scatter(y, z, s=size, c=colors, alpha=alpha, cmap=colormap, 
                          rasterized=True, norm=norm)

    # Plot mainshock
    if plot_mainshock:
        mainshock = Geographic(
            latitude=MAINSHOCK[0], longitude=MAINSHOCK[1], depth=MAINSHOCK[2],
            time=dt.datetime(2016, 11, 13, 11, 2, 56), magnitude=7.8).to_xyz(
                origin=origin, strike=strike, dip=dip)
        ax.scatter(
            mainshock.y, mainshock.z, 
            marker="*", facecolor="gold", edgecolor="k", s=200.0, zorder=100)

    # Plot interface
    subd_lats, subd_lons, subd_depths = get_williams_contours()
    subd_depths *= -1

    # extract values near line
    num_points = 100
    xvalues = np.linspace(start_longitude, end_longitude, num_points)
    yvalues = np.linspace(start_latitude, end_latitude, num_points)
    interface_points = []
    for _x, _y in zip(xvalues, yvalues):
        lat_loc = np.argmin(np.abs(subd_lats - _y))
        lon_loc = np.argmin(np.abs(subd_lons - _x))
        try:
            interface_points.append(
                Geographic(latitude=subd_lats[lat_loc], 
                           longitude=subd_lons[lon_loc], 
                           depth=subd_depths[lat_loc][lon_loc]).to_xyz(
                               origin=origin, strike=strike, dip=dip))
        except TypeError:
            # No interface here
            continue
    
    if len(interface_points) > 0:
        interface_x, interface_y, interface_z = zip(*[
            (loc.x, loc.y, loc.z) for loc in interface_points])
        ax.plot(interface_y, interface_z, color="k", linestyle="dashed")

    if focal_mechanisms:
        _cmap = plt.get_cmap(colormap)
        for _y, _z, _s, _d, _r, _size, _color in zip(y, z, s1, d1, r1, size, colors):
            if not np.all(~np.isnan([_s, _d, _r])):
                # No mechanism:
                continue
            # Compute aux plane
            s2, d2, r2 = aux_plane(_s, _d, _r)
            # Project
            meca = Mechanism(np1=NodalPlane(_s, _d, _r), 
                             np2=NodalPlane(s2, d2, r2))
            meca_rotated = rot_meca(meca=meca, p_ref=NodalPlane(strike, 90, 0))
            sr, dr, rr = (meca_rotated.np1.strike, meca_rotated.np1.dip,
                          meca_rotated.np1.rake)
            red, green, blue, _alpha = _cmap(norm(_color))
            # Fix size
            _size, _alpha = 10, 0.4
            beach_col = beach(
                fm=(sr, dr, rr), xy=(_y, _z), width=_size, 
                facecolor=(red, green, blue, _alpha), edgecolor="None",
                axes=ax)
            # Rasterize to allow plotting
            beach_col.set_rasterized(True)
            ax.add_collection(beach_col)

    ax.grid()
    ax.set_facecolor('0.8')
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(0, length)
    ax.set_ylim(-2.0, max_depth)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    if color_by.lower() == "time":
        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        # Compat with old matplotlib versions.
        if hasattr(formatter, "scaled"):
            formatter.scaled[1 / (24. * 60.)] = '%H:%M:%S'
    else:
        locator = None
        formatter = None

    cbar = fig.colorbar(scatters, ax=ax, format=formatter, ticks=locator,
                        orientation="horizontal", )

    if color_by.lower() == "distance":
        cbar.set_label("Perpendicular distance from section (km)")
    elif color_by.lower() == "time":
        cbar.set_label("Origin time (UTC)")
    elif color_by.lower() == "timestamp":
        cbar.set_label(f"Time from {starttime}")
        cbar.set_ticks([10, 100, 3600, 86400, 864000, 8640000])
        cbar.ax.set_xticklabels(
            ["10 s", "100 s", "1 hr", "1 day", "10 days", "100 days"])

    return fig


def interactive_x_section(
    earthquakes: pd.DataFrame, 
    swath_half_width: float=5.0,
    *args, **kwargs
):
    """ 
    Hopefully a map that you can click on to draw cross-sections of your choosing
    """
    import cartopy.crs as ccrs
 
    if "rotated" in kwargs.keys():
        print("Rotated not allowed")
        kwargs.pop("rotated")
    fig = plot_locations(earthquakes, cross_sections=None, rotated=False,
                         *args, **kwargs)
    projection = fig.gca().projection

    if "focal_mechanisms" in kwargs.keys():
        kwargs.pop("focal_mechanisms")

    lons, lats, line, x_section = [], [], None, plt.figure()

    def onclick(event):
        nonlocal lons, lats, earthquakes, args, kwargs, line, x_section, swath_half_width
        x, y = event.xdata, event.ydata
        lon, lat = ccrs.Geodetic().transform_point(x, y, projection)
        print(f"You clicked {lat:.4f}, {lon:.4f}")
        lons.append(lon)
        lats.append(lat)
        if len(lats) == 2:
            if line is not None:
                line.remove()
            print("Making cross-section")
            line, = fig.gca().plot(
                lons, lats, color="green", linewidth=5, zorder=1000,
                transform=ccrs.PlateCarree())
            fig.canvas.draw()
            x_section.clear()
            try:
                x_section = plot_x_section(
                    earthquakes=earthquakes, start_latitude=lats[0], 
                    start_longitude=lons[0], end_latitude=lats[-1], 
                    end_longitude=lons[-1], swath_half_width=swath_half_width, 
                    color_by="timestamp", logarithmic_color=True, 
                    fig=x_section, colormap="viridis", size=None,
                    *args, **kwargs)
            except TypeError:
                pass
            x_section.suptitle(
                f"({lats[0]:.4f}, {lons[0]:.4f}) - ({lats[-1]:.4f}, {lons[-1]:.4f})")
            x_section.canvas.draw()

            # Freshen up for the next one
            lons, lats = [], []

    
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


def distance_time_plot(
    earthquakes: pd.DataFrame,
    start_latitude: float = DEFAULT_XSECTIONS["1"][0][0],
    start_longitude: float = DEFAULT_XSECTIONS["1"][0][1],
    end_latitude: float = DEFAULT_XSECTIONS["1"][1][0],
    end_longitude: float = DEFAULT_XSECTIONS["1"][1][1],
    max_depth: float = 50.0,
    swath_half_width: float = 200.0,
    dip: float = 90.0,
    starttime: dt.datetime = None,
    endtime: dt.datetime = None,
    logx: bool = False,
    colormap: str = "plasma_r",
    size: float = 1.0,
    dip_plot: bool = False,
):
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp2d
    from math import atan, degrees
    from cjc_utilities.coordinates.coordinates import Location, Geographic
    from cjc_utilities.coordinates.extract_cross_section import get_plane

    assert {"magnitude", "latitude", "longitude", 
            "depth", "origintime"}.issubset(set(earthquakes.columns))

    origin = Geographic(latitude=start_latitude, longitude=start_longitude,
                        depth=0.0)
    start = origin.to_xyz(origin=origin, strike=0, dip=90)
    assert start.x == 0.0 and start.y == 0.0 and start.z == 0.0
    end = Geographic(latitude=end_latitude, longitude=end_longitude, 
                     depth=0.0).to_xyz(origin=origin, strike=0, dip=90)
    length = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** .5
    assert length > 0, "Start and end are the same"
    
    if end.x > 0 and end.y > 0:
        strike = degrees(atan(end.x / end.y))
    elif end.x > 0 and end.y < 0:
        strike = 90 + degrees(atan(abs(end.y) / end.x))
    elif end.x < 0 and end.y < 0:
        strike = 180 + degrees(atan(abs(end.x) / abs(end.y)))
    elif end.x < 0 and end.y > 0:
        strike = 270 + degrees(atan(end.y / abs(end.x)))
    # Cases at compass points
    elif end.x > 0 and end.y == 0:
        strike = 90.0
    elif end.x == 0 and end.y > 0:
        strike = 0.0
    elif end.x < 0 and end.y == 0:
        strike = 270.0
    elif end.x == 0 and end.y < 0:
        strike = 180.0
    else:
        raise NotImplementedError("Could not calculate strike")

    # Get the x-section plane
    x_section = get_plane(origin=origin, strike=strike, dip=dip, length=length,
                          height=-1 * max_depth)

    # Project
    quakes = filter_earthquakes(
        earthquakes, max_depth=max_depth, starttime=starttime,
        endtime=endtime)
    
    lats, lons, depths, times, magnitudes, event_ids = (
        quakes.latitude.to_numpy(), quakes.longitude.to_numpy(),
        -1 * (quakes.depth.to_numpy() / 1000.), quakes.origintime.to_list(), 
        quakes.magnitude.to_numpy(), quakes.event_id.to_list())
    projected = [
        Geographic(latitude=lat, longitude=lon, depth=depth, 
                   time=time, magnitude=mag, event_id=ev_id).to_xyz(
            origin=origin, strike=strike, dip=dip) 
        for lat, lon, depth, time, mag, ev_id in zip(
            lats, lons, depths, times, magnitudes, event_ids)]
    # Sort by time so youngest events plot last (e.g. on top)
    projected.sort(key=lambda loc: loc.time, reverse=True)

    mainshock = Geographic(
        latitude=MAINSHOCK[0], longitude=MAINSHOCK[1], depth=MAINSHOCK[2],
        time=dt.datetime(2016, 11, 13, 11, 2, 56), magnitude=7.8).to_xyz(
            origin=origin, strike=strike, dip=dip)
    # x is strike perpendicular, y strike parallel
    projected = [
        loc for loc in projected 
        if abs(loc.x) <= swath_half_width and loc.y >= 0.0 and loc.y <= length]
    if len(projected) == 0:
        return None
    
    # Plot!
    fig, ax = plt.subplots()

    x, y, z, times, mags, event_ids = zip(
        *[(loc.x, loc.y, loc.z * -1, loc.time, loc.magnitude, loc.event_id)
        for loc in projected])
    cbar_label = "Depth (km)"
    ylabel = "Distance along section (km)"
    if dip_plot:
        y, z = z, y  # flip
        ylabel, cbar_label = cbar_label, ylabel
        mainshock.y, mainshock.z = mainshock.z, mainshock.y

    starttime = starttime or min(times)
    norm = Normalize(vmin=min(z), vmax=max(z))
    fig, ax = plt.subplots()
    size = size or np.array(mags) ** 2

    if logx:
        times = [(t - starttime).total_seconds() for t in times]
        mainshock.time = (mainshock.time - starttime).total_seconds()
    mappable = ax.scatter(times, y, c=z, s=size, 
                          cmap=colormap, norm=norm)
    ax.scatter(
        mainshock.time, mainshock.y, 
        marker="*", facecolor="gold", edgecolor="k", s=200.0)
    if logx:
        ax.set_xscale('log')
        ax.set_xlabel(f"Seconds from {starttime}")
    else:
        ax.set_xlabel("UTC time")   
    
    ax.set_ylabel(ylabel)
    if dip_plot:
        ax.invert_yaxis()

    cbar = fig.colorbar(mappable, ax=ax,
                        orientation="horizontal", norm=norm)
    cbar.set_label(cbar_label)

    return fig


def plot_cumulative(
    earthquakes: pd.DataFrame,
    moment: bool = False,
    min_latitude: float = -43.2,
    min_longitude: float = 172.5,
    max_latitude: float = -41.5,
    max_longitude: float = 174.7,
    min_depth: float = None,
    max_depth: float = None,
    starttime: dt.datetime = None,
    endtime: dt.datetime = None,
    min_mag: float = None,
    max_mag: float = None,
    gps_station: str = None,
    component: str = "total",
    plot_legend: bool = True,
    plot_errors: bool = True,
    detrend_gps: bool = True,
    ax = None,
    normalize: bool = True,
    split_on_fm: bool = False,
):
    import matplotlib.pyplot as plt
    starttime = starttime or min(earthquakes.origintime)
    endtime = endtime or max(earthquakes.origintime)
    if gps_station:
        gps_data = GPSStation.from_geonet(gps_station)
        if detrend_gps:
            try:
                gps_data = gps_data.detrend(
                    dt.datetime(2015, 1, 1), dt.datetime(2016, 1, 1))
            except Exception as e:
                print(f"Could not detrend {gps_station} due to {e}")
        gps_data = gps_data.trim(starttime, endtime).zero_start()
        if component not in ("total", "all", "horizontal"):
            gps_displacement = gps_data.select(component)[0].observations
            gps_errors = gps_data.select(component)[0].errors
        if component == "horizontal":
            gps_displacement = gps_data.rotate(
                gps_data.bearing).select("r")[0].observations
            gps_errors = gps_data.select("r")[0].errors
        gps_times = gps_data[0].times
    quakes = filter_earthquakes(
        earthquakes, min_longitude=min_longitude, max_longitude=max_longitude,
        min_latitude=min_latitude, max_latitude=max_latitude,
        min_depth=min_depth, max_depth=max_depth, starttime=starttime,
        endtime=endtime, min_mag=min_mag, max_mag=max_mag)

    # Plot cumulative moment and gps displacement.
    # plt.style.use("ggplot")
    if not ax:
        fig, ax = plt.subplots()
    if not normalize:
        ax_gps = ax.twinx()
        ax_gps.grid(False)
    else:
        ax_gps = ax  # Plot on the same axis

    # Plot GPS data

    component_label_mapper = {
        "u": "Vertical displacement", 
        "n": "North displacement", 
        "e": "East displacement"}
    handles, labels = [], []
    if component == "all":
        for _component in ("u", "n", "e"):
            _disp = gps_data.select(_component)[0]
            if normalize:
                _disp.observations /= _disp.observations[-1]
            handle = ax_gps.plot(gps_times, _disp.observations, zorder=2)[0]
            if plot_errors:
                ax_gps.fill_between(
                    gps_times, _disp.observations + _disp.errors, 
                    _disp.observations - _disp.errors, alpha=0.4)
            label = component_label_mapper[_component]
            handles.append(handle)
            labels.append(label)
    else:
        if normalize:
            gps_displacement /= gps_displacement[-1]
        handle = ax_gps.plot(gps_times, gps_displacement, zorder=2)[0]
        if plot_errors:
            ax_gps.fill_between(
                gps_times, gps_displacement + gps_errors, 
                gps_displacement - gps_errors, alpha=0.4)
        if component == "horizontal":
            label = f"Horizontal at ${int(gps_data.bearing):03d}\degree$"
        else:
            label = component_label_mapper.get(component, component)
        handles.append(handle)
        labels.append(label)

    # Plot Earthquake data
    if not split_on_fm:
        magnitudes = [quakes.magnitude.to_numpy()]
        times = [quakes.origintime.to_list()]
        if moment:
            eq_labels = ["Cumulative moment"]
            eq_units = "(dyn cm)"
        else:
            eq_labels = ["Cumulative number"]
            eq_units = ""
    else:
        magnitudes = [
            quakes.magnitude[quakes["Slip style"] == "Normal"].to_numpy(),
            quakes.magnitude[quakes["Slip style"] == "Reverse"].to_numpy(),
            quakes.magnitude[quakes["Slip style"] == "Strike-slip"].to_numpy(),
            quakes.magnitude[pd.isna(quakes["Slip style"])].to_numpy()]
        times = [
            quakes.origintime[quakes["Slip style"] == "Normal"].to_list(),
            quakes.origintime[quakes["Slip style"] == "Reverse"].to_list(),
            quakes.origintime[quakes["Slip style"] == "Strike-slip"].to_list(),
            quakes.origintime[pd.isna(quakes["Slip style"])].to_list()]
        if moment:
            eq_labels = [
                "Cumulative moment: Normal",
                "Cumulative moment: Reverse",
                "Cumulative moment: Strike-slip",
                "Cumulative moment: Unknown"]
            eq_units = "(dyn cm)"
        else:
            eq_labels = [
                "Cumulative number: Normal",
                "Cumulative number: Reverse",
                "Cumulative number: Strike-slip",
                "Cumulative number: Unknown"]
            eq_units = ""
    for mag, _times, eq_label in zip(magnitudes, times, eq_labels):
        # Convert to cumulative moment
        if moment:
            moment = 10 ** (1.5 * (mag + 10.73))  # Equation 5 p 266 Stein and Wysession
            cumulative = np.nan_to_num(moment, 0.0).cumsum()
        else:
            cumulative = np.arange(len(mag))
        _t = [t.to_pydatetime() for t in _times]
        if normalize:
            cumulative = cumulative / max(cumulative)
        handle = ax.step(
            _t, cumulative, lw=2, zorder=10, linestyle="--", color="red")[0]
        handles.append(handle)
        labels.append(eq_label)

    
    if plot_legend:
        ax.legend(handles=handles, labels=labels, loc="lower right", 
                  facecolor="white", framealpha=1.0)
    ax.set_xlabel("Date and time (UTC)")
    ax.set_xlim(starttime, endtime)
    ax_gps.set_ylabel(f"{gps_station} (mm)")

    return ax.get_figure()


def plot_regions(
    earthquakes: pd.DataFrame,
    starttime: dt.datetime = None, 
    endtime: dt.datetime = None,
    component: str = "all",
    moment: bool = False,
    normalize: bool = True,
    plot_errors: bool = False,
    split_on_fm: bool = False,
):
    import matplotlib.pyplot as plt
    
    regions = [
        ("Northern", (-42, -41.5, 174, 174.5), "CMBL"),
        ("Central", (-42.5, -42.1, 173.5, 174), "KAIK"),
        ("Epicentral", (-42.7, -42.4, 172.5, 173.6), "MRBL"),
        ("Papatea-Kek", (-42.15, -41.8, 173.5, 173.8), "GDS1")]
    fig, axes = plt.subplots(nrows=len(regions), sharex=True)
    if component == "horizontal":  # Label all axes
        plot_legend = True
    else:
        plot_legend = False
    for region, ax in zip(regions, axes):
        if region == regions[-1]:
            plot_legend = True
        plot_cumulative(
            earthquakes=earthquakes, gps_station=region[-1],
            min_latitude=region[1][0], max_latitude=region[1][1],
            min_longitude=region[1][2], max_longitude=region[1][3],
            component=component, starttime=starttime, endtime=endtime,
            ax=ax, plot_legend=plot_legend, moment=moment, 
            normalize=normalize, plot_errors=plot_errors, 
            split_on_fm=split_on_fm)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def compare_evolution(
    earthquakes: pd.DataFrame,
    starttime: dt.datetime = None, 
    endtime: dt.datetime = None,
    normalize: bool = True,
    log_time: bool = False,
):
    import matplotlib.pyplot as plt

    regions = [
        ("Northern", (-42, -41.5, 174, 174.5), "CMBL"),
        ("Central", (-42.5, -42.1, 173.5, 174), "KAIK"),
        ("Epicentral", (-42.7, -42.4, 172.5, 173.6), "MRBL"),
        ("Papatea-Kek", (-42.15, -41.8, 173.5, 173.8), "GDS1")]

    fig, ax = plt.subplots()
    for region_name, bounds, station in regions:
        quakes = filter_earthquakes(
            earthquakes=earthquakes, min_latitude=bounds[0], 
            max_latitude=bounds[1], min_longitude=bounds[2], 
            max_longitude=bounds[3], starttime=starttime, endtime=endtime)
        cumulative = np.arange(len(quakes))
        times = [t.to_pydatetime() for t in quakes.origintime.to_list()]
        if log_time:
            times = [(t - starttime).total_seconds() for t in times]
        if normalize:
            cumulative = cumulative / len(cumulative)
        ax.step(times, cumulative, lw=2, zorder=10, label=region_name)
    
    ax.legend()
    ax.set_ylabel("Cumulative events")
    if log_time:
        ax.semilogx()
        ax.set_xlabel(f"Seconds since {starttime}")
    else:
        ax.set_xlabel("Origin time")
    return fig


def plot_bvalues(magnitude_times):
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    ax.scatter(
        magnitude_times["origintime"], magnitude_times["magnitude"], 
        color="C0", marker="o", rasterized=True, s=2.0, alpha=0.5)
    # Don't plot where the bvaue wasn't stable
    mc_masked = magnitude_times.mc_max_curv.to_numpy()
    mc_masked[pd.isna(magnitude_times.bvalues)] = np.nan
    ax.step(
        magnitude_times["origintime"], mc_masked,
        color="C1", label="$M_C$", where="post")
    ax2 = ax.twinx()
    ax2.step(magnitude_times["origintime"], magnitude_times["bvalues"], 
            color="C2", label="b-value", where="post")
    # ax.plot(magnitude_times["window_median"], magnitude_times["mc_max_curv"], 
    #         color="C1", label="$M_C$")
    # ax2 = ax.twinx()
    # ax2.plot(magnitude_times["window_median"], magnitude_times["bvalues"], 
    #          color="C2", label="b-value")
    ax.set_xlim([magnitude_times["origintime"].min(), 
                 magnitude_times["origintime"].max()])
    ax.set_xlabel("Origin time (UTC)")
    ax.set_ylabel("Magnitude")
    ax.grid()
    ax2.set_ylabel("b-value")

    # Get grids to work well
    ax.set_ylim([0, 7])
    ax2.set_ylim([0.4, 1.1])

    fig.legend()
    return fig


def plot_freq_mag(magnitude_times, starttime=None, endtime=None):
    starttime = starttime or min(magnitude_times.origintime)
    endtime = endtime or max(magnitude_times.origintime)
    window = magnitude_times[magnitude_times.origintime >= starttime]
    window = window[window.origintime <= endtime]

    mags = [m for m in window.magnitude if not np.isnan(m)]
    _mc = calc_max_curv(mags, bin_size=0.5)
    bvals = calc_b_value(
        mags, completeness=np.arange(_mc - 0.75, _mc + 3.5, 0.05), 
        plotvar=False)
    best_index = np.argmax(list(zip(*bvals))[2])
    mc = bvals[best_index][0]
    fig = freq_mag(
        mags, completeness=mc, max_mag=7, show=False, return_figure=True)
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    earthquakes = pd.read_csv(
        "../../Locations/GrowClust_located_magnitudes_callibrated_focal_mechanisms.csv",
        parse_dates=["time"]) 
    earthquakes = earthquakes.rename(columns={"time": "origintime"})

    # relocated = earthquakes[earthquakes["origin method"] == "GrowClust"]
    relocated = earthquakes[pd.isna(earthquakes.stations)]

    magnitude_times = calculate_moving_b(earthquakes=earthquakes)
    magnitude_times.to_csv("Moving_mc.csv")

    bval_fig = plot_bvalues(magnitude_times)

    kaikoura = dt.datetime(2016, 11, 13, 11)
    flatten_out = dt.datetime(2016, 12, 1)

    pre_kaik_mf = plot_freq_mag(
        magnitude_times, endtime=kaikoura)
    pre_kaik_mf.suptitle("Pre-Kaikoura")
    post_kaikoura_mf = plot_freq_mag(
        magnitude_times, starttime=kaikoura)
    post_kaikoura_mf.suptitle("Post-Kaikoura")
    last_kaik_mf = plot_freq_mag(
        magnitude_times, starttime=flatten_out)
    last_kaik_mf.suptitle(f"After {flatten_out.strftime('%Y/%m/%d')}")

    early_kaik_mf = plot_freq_mag(
        magnitude_times, starttime=kaikoura, endtime=flatten_out)
    early_kaik_mf.suptitle(f"Kaikoura to {flatten_out.strftime('%Y/%m/%d')}")

    plt.show()
