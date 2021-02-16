import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from typing import Tuple, List
import cartopy.crs as ccrs

from kaikoura_csv_visualisations import (
    plot_locations, get_kaikoura_faults, get_williams_contours, read_faults,
    scale_bar, plot_x_section, project_onto_section, filter_earthquakes,
    distance_time_plot, plot_fm)

import os

HOME = os.path.expanduser("~")
# Set globals for all figures
TIME_COLORMAP = "turbo_r"
DEPTH_COLORMAP = "plasma_r"

BOUNDS = (-43.0, 172.5, -41.5, 174.7)
MAX_DEPTH = 40.0
CROSS_SECTIONS = {
    "A": [(-42.8, 172.75), (-41.58, 174.6)],  # Along strike
    "epi": [(-42.716175, 172.755662), (-42.497011, 173.399676)],  # Epicentral
    "thrust": [(-42.155254, 173.508222), (-42.419545, 173.895243)],  # Point Keen
    # "C": [(-42.119, 173.814), (-41.953, 173.644)],  # Clarence link
    "cape": [(-41.676246, 174.027403), (-41.899032, 174.357790)],  # Cape Campbell
    # "cape": [(-41.625783, 174.116114), (-41.851116, 174.417139)],  # Cape Campbell
    # "E": [(-41.958, 174.004), (-41.65296, 174.5287)]
}
REGIONS = {
    "Fig. 4": ((-42.74, 172.75), (-42.4, 173.4)),
    "Fig. 5": ((-42, 174), (-41.55, 174.5)),
    "Fig. 6": ((-42.55, 173.25), (-41.96, 174)),
}
RELOCATED_EVENTS = "../../Locations/GrowClust_located_magnitudes_callibrated_focal_mechanisms.csv"

MAINSHOCK = (-42.623531, 172.988788, 12.917969)  # My NLL location.
MAINSHOCK_TIME = dt.datetime(2016, 11, 13, 11, 2, 56)  # GeoNet origin time
# Shift the start of plots to a bit before
MAINSHOCK_TIME -= dt.timedelta(seconds=30)

OUT = f"{HOME}/Dropbox/Current_projects/Kaikoura_afterslip/Plots/Paper_figures"
PLOT_FORMATS = ("png", "svg", "pdf")

NZTM = ccrs.TransverseMercator(
    central_longitude=173.0, central_latitude=0.0, 
    false_easting=1600000, false_northing=10000000, 
    scale_factor=0.9996)

plt.style.use("default")


def plot_subduction_zone(
    map_ax, 
    min_latitude: float, 
    max_latitude: float, 
    min_longitude: float,
    max_longitude: float,
    levels: list = [0, 5, 10, 15, 20, 25, 30, 40, 50],
):
    import cartopy.crs as ccrs
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
        transform=ccrs.PlateCarree(), levels=levels)
    map_ax.clabel(contours, inline=1, fontsize=10, fmt="%i km")
    return contours


def plot_map_and_section(
    size, earthquakes, bounds, section, max_depth=20, half_width=20,
    fm_size=15):
    import copy
    import cartopy.crs as ccrs

    from matplotlib.colors import Normalize
    from cjc_utilities.animator.animator import _blank_map

    relocated = earthquakes[earthquakes.station_count == 0.0]

    (min_latitude, min_longitude), (max_latitude, max_longitude) = bounds
    scale_bar_length = 5
    
    pad = 0.05
    map_size = size
    fig = plt.figure(figsize=map_size)

    gs = fig.add_gridspec(24, 1)
    
    cbar_ax = fig.add_subplot(gs[12, :])
    section_ax = fig.add_subplot(gs[14:, :])

    quakes = filter_earthquakes(
        relocated, min_longitude=min_longitude, max_longitude=max_longitude,
        min_latitude=min_latitude, max_latitude=max_latitude)

    quakes = quakes.sort_values(
        by="origintime", ignore_index=True, ascending=False)
    
    lats, lons, depths, times = (
        quakes.latitude.to_numpy(), quakes.longitude.to_numpy(),
        quakes.depth.to_numpy() / 1000., quakes.origintime.to_list())

    # Get the projection
    if min(lons) < -150 and max(lons) > 150:
        max_lons = max(np.array(lons) % 360)
        min_lons = min(np.array(lons) % 360)
    else:
        max_lons = max(lons)
        min_lons = min(lons)
    lat_0 = max(lats) / 2. + min(lats) / 2.
    lon_0 = max_lons / 2. + min_lons / 2.
    if lon_0 > 180:
        lon_0 -= 360
    deg2m_lat = 2 * np.pi * 6371 * 1000 / 360
    deg2m_lon = deg2m_lat * np.cos(lat_0 / 180 * np.pi)
    if len(lats) > 1:
        height = (max(lats) - min(lats)) * deg2m_lat
        width = (max_lons - min_lons) * deg2m_lon
        margin = pad * (width + height)
        height += margin
        width += margin
    else:
        height = 2.0 * deg2m_lat
        width = 5.0 * deg2m_lon
    # Do intelligent aspect calculation for local projection
    # adjust to figure dimensions
    w, h = fig.get_size_inches()
    aspect = w / h
    aspect *= 1.2
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect

    proj_kwargs = {}
    proj_kwargs['central_latitude'] = lat_0
    proj_kwargs['central_longitude'] = lon_0
    proj_kwargs['standard_parallels'] = [lat_0, lat_0]
    # proj = ccrs.AlbersEqualArea(**proj_kwargs)
    proj = NZTM

    map_ax = fig.add_subplot(gs[0:12, :], projection=proj)
    
    sizes = quakes.magnitude ** 2
    colors = depths
    norm = Normalize(vmin=min(colors), vmax=20.0)
    colormap = copy.copy(plt.get_cmap("plasma_r"))
    colormap.set_over(color="k")

    fig, map_ax, cbar_ax, cb = _blank_map(
        lats=lats, lons=lons, color=colors, projection="local", 
        resolution="full", colormap=colormap, figsize=map_size,
        proj_kwargs={}, norm=norm, continent_fill_color="0.65",
        water_fill_color="0.9", fig=fig, map_ax=map_ax, cm_ax=cbar_ax)

    map_ax.scatter(
        lons, lats, marker="o", s=sizes, c=colors, zorder=10, alpha=0.9,
        cmap=colormap, transform=ccrs.PlateCarree(), rasterized=True,
        norm=norm)

    if MAINSHOCK[0] <= max_latitude and MAINSHOCK[0] >= min_latitude and MAINSHOCK[1] <= max_longitude and MAINSHOCK[1] >= min_longitude:
        mainshock = map_ax.scatter(
            MAINSHOCK[1], MAINSHOCK[0], marker="*", facecolor="gold", 
            edgecolor="k", s=200.0, zorder=11, transform=ccrs.PlateCarree())
    else:
        mainshock = None

    # Plot mechanisms
    if fm_size:
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
            color = _depth[i] / 1000.
            red, green, blue, _alpha = colormap(norm(color))
            plot_fm(strike=s[i], dip=d[i], rake=r[i],
                    latitude=lat[i], longitude=lon[i], zorder=20, axes=map_ax,
                    width=fm_size, color=(red, green, blue), alpha=0.6, 
                    rasterize=True)

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
                flons, flats, color="red", linewidth=2.0, zorder=9,
                transform=ccrs.PlateCarree(), rasterized=True)

    map_ax.set_extent(
        [min_longitude, max_longitude, min_latitude, max_latitude],
        crs=ccrs.PlateCarree())

    gl = map_ax.gridlines(
        draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gl.right_labels, gl.bottom_labels = False, False

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
    scale_bar(map_ax, (0.85, 0.05), scale_bar_length, angle=0)
    scale_bar(map_ax, (0.85, 0.05), scale_bar_length, angle=90)

    if mainshock:
        handles = [f_line, mainshock, contours.collections[0]]
        labels = ["Active Faults", "Mainshock", "Williams et al. Interface"]
    else:
        handles = [f_line, contours.collections[0]]
        labels = ["Active Faults", "Williams et al. Interface"]
    if kaik_f_line:
        handles.append(kaik_f_line)
        labels.append("Surface Rupture")

    map_ax.legend(handles=handles, labels=labels, framealpha=1.0, 
                  loc="upper left").set_zorder(10000)

    # Put the mainshock mechanism on
    if mainshock:
        mainshock_mask = earthquakes.magnitude == earthquakes.magnitude.max()
        mainshock = earthquakes[mainshock_mask].sort_values(
            "magnitude", ignore_index=True)
        s, d, r, lat, lon = (
            mainshock.strike[0], mainshock.dip[0], mainshock.rake[0],
            mainshock.latitude[0], mainshock.longitude[0])
        plot_fm(strike=s, dip=d, rake=r, longitude=lon - .03, latitude=lat + .03, 
                axes=map_ax, color="red",
                width=30,  edgecolor="k", alpha=1.0,
                linewidth=0.3)
    
        map_ax.plot(
            (lon, lon - 0.03), (lat, lat + 0.03), "k", zorder=4,
            transform=ccrs.PlateCarree())

    # Plot x-section line
    name, points = section, CROSS_SECTIONS[section]
    x_start, x_end = points
    x_lats, x_lons = [x_start[0], x_end[0]], [x_start[1], x_end[1]]
    map_ax.plot(x_lons, x_lats, color="cyan", linewidth=3.0, zorder=2,
                transform=ccrs.PlateCarree(), linestyle="--")

    # Plot x-section
    if mainshock:
        plot_mainshock = True
    else:
        plot_mainshock = False
    fig = plot_x_section(
        earthquakes=relocated, start_latitude=CROSS_SECTIONS[section][0][0],
        start_longitude=CROSS_SECTIONS[section][0][1],
        end_latitude=CROSS_SECTIONS[section][1][0],
        end_longitude=CROSS_SECTIONS[section][1][1],
        starttime=MAINSHOCK_TIME,
        max_depth=max_depth, swath_half_width=half_width, dip=90.0, colormap="turbo_r",
        size=None, logarithmic_color=True, color_by="timestamp", fig=fig,
        plot_mainshock=plot_mainshock, ax=section_ax)
    return fig


def fig_1(size=(10, 10)):
    """
    Overview map with inset of whole of NZ. Plot GPS, Seismographs, faults, geology?

    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=size)
    ax_x0, ax_width, ax_y0, ax_height = 0.1, 0.8, 0.1, 0.8

    min_latitude, min_longitude, max_latitude, max_longitude = BOUNDS

    lons, lats = (min_longitude, max_longitude), (min_latitude, max_latitude)
    if min(lons) < -150 and max(lons) > 150:
        max_lons = max(np.array(lons) % 360)
        min_lons = min(np.array(lons) % 360)
    else:
        max_lons = max(lons)
        min_lons = min(lons)
    lat_0 = max(lats) / 2. + min(lats) / 2.
    lon_0 = max_lons / 2. + min_lons / 2.
    if lon_0 > 180:
        lon_0 -= 360
    deg2m_lat = 2 * np.pi * 6371 * 1000 / 360
    deg2m_lon = deg2m_lat * np.cos(lat_0 / 180 * np.pi)
    height = (max(lats) - min(lats)) * deg2m_lat
    width = (max_lons - min_lons) * deg2m_lon
    # Do intelligent aspect calculation for local projection
    # adjust to figure dimensions
    w, h = fig.get_size_inches()
    aspect = w / h
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    proj_kwargs = dict()
    proj_kwargs['central_latitude'] = lat_0
    proj_kwargs['central_longitude'] = lon_0
    proj_kwargs['standard_parallels'] = [lat_0, lat_0]
    # proj = ccrs.AlbersEqualArea(**proj_kwargs)
    proj = NZTM

    map_ax = fig.add_axes([ax_x0, ax_y0, ax_width, ax_height],
                          projection=proj)

    coast = cfeature.GSHHSFeature(
        scale="high", levels=[1], facecolor="0.9", 
        edgecolor="0.4")
    map_ax.set_facecolor("0.65")
    map_ax.add_feature(coast)

    gl = map_ax.gridlines(
        draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gl.right_labels, gl.bottom_labels = False, False
    
    mainshock = map_ax.scatter(
        MAINSHOCK[1], MAINSHOCK[0], marker="*", facecolor="gold", 
        edgecolor="k", s=200.0, zorder=11, transform=ccrs.PlateCarree())
    
    # Plot Faults
    faults = read_faults(
        min_lat=min_latitude - 2, max_lat=max_latitude + 2,
        min_lon=min_longitude - 2, max_lon=max_longitude + 2)
    for fault in faults:
        flons, flats = zip(*fault)
        f_line, = map_ax.plot(
            flons, flats, color="k", linewidth=1.0, zorder=8,
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
                flons, flats, color="red", linewidth=2.0, zorder=9,
                transform=ccrs.PlateCarree(), rasterized=True)

    # Plot GPS stations
    gps_stations = pd.read_csv(
        "gps_station_locations.csv", parse_dates=["Start", "End"])
    gps_markers = map_ax.scatter(
        gps_stations.Longitude, gps_stations.Latitude, facecolor="lime",
        edgecolor="k", marker="^", zorder=10, transform=ccrs.PlateCarree(), 
        s=150)
    for lat, lon, code in zip(gps_stations.Latitude, gps_stations.Longitude, gps_stations["Station code"]):
        map_ax.text(lon + 0.015, lat - 0.01, code, transform=ccrs.PlateCarree(), 
                    zorder=100, clip_on=True)
    
    # Plot Seismographs
    seismographs = pd.read_csv(
        "seismograph_locations.csv", parse_dates=["Start", "End"])
    seismograph_markers = map_ax.scatter(
        seismographs.Longitude, seismographs.Latitude, facecolor="orange",
        edgecolor="k", marker="v", zorder=10, transform=ccrs.PlateCarree(),
        s=200)
    for lat, lon, code in zip(seismographs.Latitude, seismographs.Longitude, seismographs["Station code"]):
        if min_latitude < lat and lat < max_latitude and min_longitude < lon and lon < max_longitude:
            map_ax.text(lon - 0.02, lat - 0.01, code,
                        transform=ccrs.PlateCarree(), zorder=100, clip_on=True,
                        ha="right")

    # Plot subduction zone
    contours = plot_subduction_zone(
        map_ax=map_ax, min_latitude=min_latitude, min_longitude=min_longitude,
        max_latitude=max_latitude, max_longitude=max_longitude)

    # Label faults
    fault_labels = dict(
        needles=dict(
            name="Needles", label_lon=174.38, label_lat=-41.8, 
            fault_lon=174.28, fault_lat=-41.8, ha="left"),
        london=dict(
            name="London Hills", label_lon=174.33, label_lat=-41.7, 
            fault_lon=174.22, fault_lat=-41.74, ha="left"), 
        kek=dict(
            name="Kekerengu", label_lon=174.19, label_lat=-41.98, 
            fault_lon=173.99, fault_lat=-41.98, ha="left"), 
        paptea=dict(
            name="Papatea", label_lon=174, label_lat=-42.16, 
            fault_lon=173.87, fault_lat=-42.16, ha="left"),
        fidget=dict(
            name="Fidget", label_lon=173.73, label_lat=-42., 
            fault_lon=173.68, fault_lat=-42.12, ha="center"),
        jordan=dict(
            name="Jordan Thurst", label_lon=173.57, label_lat=-42.2, 
            fault_lon=173.77, fault_lat=-42.13, ha="right"),
        clarence=dict(
            name="Clarence", label_lon=173.64, label_lat=-41.82, 
            fault_lon=173.84, fault_lat=-41.92, ha="right"),
        hope=dict(
            name="Hope", label_lon=173.22, label_lat=-42.39, 
            fault_lon=173.42, fault_lat=-42.39, ha="right"),
        keen=dict(
            name="Point Keen", label_lon=173.95, label_lat=-42.38, 
            fault_lon=173.85, fault_lat=-42.38, ha="left"),
        whites=dict(
            name="Whites", label_lon=173.72, label_lat=-42.43, 
            fault_lon=173.52, fault_lat=-42.43, ha="left"),
        kowhai=dict(
            name="Upper Kowhai", label_lon=173.87, label_lat=-42.26, 
            fault_lon=173.57, fault_lat=-42.26, ha="left"),
        conway=dict(
            name="Conway-Charwell", label_lon=173.19, label_lat=-42.44, 
            fault_lon=173.29, fault_lat=-42.44, ha="right"),
        hundalee=dict(
            name="Hundalee", label_lon=173.55, label_lat=-42.61, 
            fault_lon=173.5, fault_lat=-42.51, ha="center"),
        stone=dict(
            name="Stone Jug", label_lon=173.58, label_lat=-42.5, 
            fault_lon=173.38, fault_lat=-42.45, ha="left"),
        leader=dict(
            name="Leader", label_lon=173.08, label_lat=-42.53, 
            fault_lon=173.28, fault_lat=-42.53, ha="right"),
        humps=dict(
            name="Humps", label_lon=173.23, label_lat=-42.65, 
            fault_lon=173.13, fault_lat=-42.61, ha="left"),
    )

    for _fault in fault_labels.values():
        map_ax.text(_fault["label_lon"], _fault["label_lat"], _fault["name"],
                    transform=ccrs.PlateCarree(), zorder=100, va="center",
                    ha=_fault["ha"],
                    bbox=dict(fc="white", boxstyle="round", ec="black"))
        map_ax.plot((_fault["label_lon"], _fault["fault_lon"]),
                    (_fault["label_lat"], _fault["fault_lat"]),
                    "k:", linewidth=1.0, transform=ccrs.PlateCarree())

    map_ax.set_extent(
        [min_longitude, max_longitude, min_latitude, max_latitude], 
        crs=ccrs.PlateCarree())

    # Plot scale bar
    scale_bar(map_ax, (0.05, 0.05), 20, angle=0)
    scale_bar(map_ax, (0.05, 0.05), 20, angle=90)

    # Add inset

    big_max_lat, big_min_lat, big_min_lon, big_max_lon = (
        -36, -47, 166, 178.5)
    big_map_ax = fig.add_axes([0.57, 0.05, 0.35, 0.35], projection=proj)
    
    coast_big = cfeature.GSHHSFeature(
        scale="full", levels=[1], facecolor="0.9", 
        edgecolor="0.4")
    big_map_ax.set_facecolor("0.65")
    big_map_ax.add_feature(coast_big)

    big_gl = big_map_ax.gridlines(
        draw_labels=True, dms=False, x_inline=False, y_inline=False)
    big_gl.left_labels, big_gl.top_labels = False, False
    
    big_map_ax.scatter(
        MAINSHOCK[1], MAINSHOCK[0], marker="*", facecolor="gold", 
        edgecolor="k", s=200.0, zorder=11, transform=ccrs.PlateCarree())
    
    big_map_ax.scatter(
        seismographs.Longitude, seismographs.Latitude, facecolor="orange",
        edgecolor="k", marker="v", zorder=10, transform=ccrs.PlateCarree(),
        s=100)
    # for lat, lon, code in zip(seismographs.Latitude, seismographs.Longitude, seismographs["Station code"]):
    #     big_map_ax.text(lon + 0.01, lat - 0.01, code,
    #                     transform=ccrs.PlateCarree(), zorder=100, clip_on=True)

    # Add subduction zone
    plot_subduction_zone(
        map_ax=big_map_ax, min_latitude=big_min_lat, min_longitude=big_min_lon,        max_latitude=big_max_lat, max_longitude=big_max_lon, 
        levels=[5, 25, 50, 100])

    # Add Faults
    faults = read_faults(
        min_lat=big_min_lat, max_lat=big_max_lat,
        min_lon=big_min_lon, max_lon=big_max_lon)
    for fault in faults:
        flons, flats = zip(*fault)
        f_line, = big_map_ax.plot(
            flons, flats, color="k", linewidth=0.5, zorder=8,
            transform=ccrs.PlateCarree(), rasterized=True)

    big_map_ax.text(168.4, -44.0, "Alpine Fault", transform=ccrs.PlateCarree(), 
                    zorder=100, rotation=35.)

    # Add inset box
    big_map_ax.plot(
        (min_longitude, max_longitude, max_longitude, min_longitude, min_longitude),
        (min_latitude, min_latitude, max_latitude, max_latitude, min_latitude),
        color="red", linewidth=2.0, transform=ccrs.PlateCarree())

    big_map_ax.set_extent(
        [big_min_lon, big_max_lon, big_min_lat, big_max_lat],
        crs=ccrs.PlateCarree())

    fig.legend(
        handles=(f_line, kaik_f_line, mainshock, contours.collections[0],
                gps_markers, seismograph_markers),
        labels=("Active Faults", "Surface Ruptures", "Mainshock", 
                "Williams et al. Interface", "cGPS", "Seismograph"),
        framealpha=1.0)
    # Save
    for _format in PLOT_FORMATS:
        fig.savefig(f"{OUT}/Figure_1.{_format}")
    return fig


def fig_2(size=(10, 10)):
    """
    Map of relocated seismicity - boxes showing insets of subsequent figures.

    """
    import cartopy.crs as ccrs

    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})
    relocated = earthquakes[earthquakes.station_count == 0.0]

    relocated_map = plot_locations(
        earthquakes=relocated, color_by="depth", rotated=False,
        max_depth=40.0, min_depth=None, min_latitude=BOUNDS[0],
        min_longitude=BOUNDS[1], max_latitude=BOUNDS[2],
        max_longitude=BOUNDS[3], starttime=None, endtime=None,
        logarithmic_color=False, size=0.2, colormap=DEPTH_COLORMAP,
        cross_sections=None, relief=False, focal_mechanisms=False,
        plot_stations=False, rupture_color="red")
    
    relocated_map.set_size_inches(size)

    map_ax = relocated_map.gca()
    name, points = "A", CROSS_SECTIONS["A"]
    x_start, x_end = points
    x_lats, x_lons = [x_start[0], x_end[0]], [x_start[1], x_end[1]]
    map_ax.plot(x_lons, x_lats, color="cyan", linewidth=3.0, zorder=2,
                transform=ccrs.PlateCarree(), linestyle="--")
    # Label
    map_ax.text(x_start[1], x_start[0], s=name, 
                transform=ccrs.PlateCarree(),
                bbox=dict(fc="white", ec="black", boxstyle="round"),
                zorder=20)
    map_ax.text(x_end[1], x_end[0], s=f"{name}'",
                transform=ccrs.PlateCarree(), 
                bbox=dict(fc="white", ec="black", boxstyle="round"),
                zorder=20)

    for name, bounds in REGIONS.items():
        bottom_left, top_right = bounds
        lats = [bottom_left[0], bottom_left[0], top_right[0], top_right[0], 
            bottom_left[0]]
        lons = [bottom_left[1], top_right[1], top_right[1], bottom_left[1], 
            bottom_left[1]]
        map_ax.plot(lons, lats, color="darkblue", linewidth=2.0, zorder=2,
                    transform=ccrs.PlateCarree(), linestyle="--", alpha=0.5)
        # Label
        map_ax.text(bottom_left[1], top_right[0], s=name,
                    transform=ccrs.PlateCarree(),
                    bbox=dict(fc="white", ec="black", boxstyle="round"),
                    zorder=20)

    for _format in PLOT_FORMATS:
        relocated_map.savefig(f"{OUT}/Figure_2.{_format}")

    return    


def fig_3(size=(9, 6)):
    """ Completeness and space-time. """
    mag_time = pd.read_csv(
        "Moving_mc.csv", parse_dates=["origintime", "window_median"])

    fig = plt.figure(figsize=size, constrained_layout=True)
    gs = plt.GridSpec(nrows=4, ncols=1, hspace=0.05, figure=fig)
    mc_ax = fig.add_subplot(gs[0:2, :])
    space_time_ax = fig.add_subplot(gs[2:4, :], sharex=mc_ax)
    
    # Plot catalogue completeness
    mags = mc_ax.scatter(
        mag_time.origintime, mag_time.magnitude, color="darkblue", 
        s=0.5, rasterized=True)
    mc_masked = mag_time.mc_max_curv.to_numpy()
    mc_masked[pd.isna(mag_time.bvalues)] = np.nan
    completeness, = mc_ax.step(
        mag_time["origintime"], mc_masked, color="red", 
        label="$M_C$", where="post")
    
    mc_ax.set_ylabel("Local magnitude")
    mc_ax.legend(handles=(mags, completeness), labels=("Earthquake", "$M_C$"))

    # Plot space and time
    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})
    fig = distance_time_plot(
        earthquakes=earthquakes, start_latitude=CROSS_SECTIONS["A"][0][0],
        start_longitude=CROSS_SECTIONS["A"][0][1],
        end_latitude=CROSS_SECTIONS["A"][1][0],
        end_longitude=CROSS_SECTIONS["A"][1][1],
        max_depth=50.0, swath_half_width=200.0, dip=90.0,
        colormap="plasma_r", ax=space_time_ax)
    
    mc_ax.set_xlim(earthquakes.origintime.min(), earthquakes.origintime.max())

    mc_ax.grid()
    space_time_ax.grid()

    # fig.subplots_adjust(hspace=0, wspace=0.5)

    for _format in PLOT_FORMATS:
        fig.savefig(f"{OUT}/Figure_3.{_format}")
    return


def fig_4(size=(9, 15)):
    """ Nucleation, 2 panels of map and along-strike cross-section """

    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})

    fig = plot_map_and_section(
        size=size, earthquakes=earthquakes, bounds=REGIONS["Fig. 4"], 
        section="epi")
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for _format in PLOT_FORMATS:
        fig.savefig(f"{OUT}/Figure_4.{_format}")
    return


def fig_5(size=(7, 16)):
    """
    Cape Campbell sections - illustrate lack of hard stop,
    illustrate many faults ruptured, show subd-like earthquakes, and similarity
    to Ck. St. events.

    TODO:
     - Plot Fault projections
     - Label faults
     - Highlight re-invigorated events
     - Focal Mechanisms for subd like events

    """
    import copy
    import cartopy.crs as ccrs

    from obspy.clients.fdsn import Client
    from matplotlib.colors import Normalize
    from cjc_utilities.animator.animator import _blank_map

    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})

    bounds = REGIONS["Fig. 5"]
    section="cape"

    # Get the Cook Strait earthquakes
    client = Client("GEONET")
    grassmere = client.get_events(eventid="2013p613797")[0]  # Lake Grassmere
    ck_st = client.get_events(eventid="2013p543824")[0]  # Cook Strait

    relocated = earthquakes[earthquakes.station_count == 0.0]

    (min_latitude, min_longitude), (max_latitude, max_longitude) = bounds
    scale_bar_length = 5
    
    pad = 0.05
    map_size = size
    fig = plt.figure(figsize=map_size)

    gs = fig.add_gridspec(24, 1)
    
    cbar_ax = fig.add_subplot(gs[11, :])
    section_ax = fig.add_subplot(gs[13:, :])

    quakes = filter_earthquakes(
        relocated, min_longitude=min_longitude, max_longitude=max_longitude,
        min_latitude=min_latitude, max_latitude=max_latitude)

    quakes = quakes.sort_values(
        by="origintime", ignore_index=True, ascending=False)
    
    lats, lons, depths, times = (
        quakes.latitude.to_numpy(), quakes.longitude.to_numpy(),
        quakes.depth.to_numpy() / 1000., quakes.origintime.to_list())

    # Get the projection
    if min(lons) < -150 and max(lons) > 150:
        max_lons = max(np.array(lons) % 360)
        min_lons = min(np.array(lons) % 360)
    else:
        max_lons = max(lons)
        min_lons = min(lons)
    lat_0 = max(lats) / 2. + min(lats) / 2.
    lon_0 = max_lons / 2. + min_lons / 2.
    if lon_0 > 180:
        lon_0 -= 360
    deg2m_lat = 2 * np.pi * 6371 * 1000 / 360
    deg2m_lon = deg2m_lat * np.cos(lat_0 / 180 * np.pi)
    if len(lats) > 1:
        height = (max(lats) - min(lats)) * deg2m_lat
        width = (max_lons - min_lons) * deg2m_lon
        margin = pad * (width + height)
        height += margin
        width += margin
    else:
        height = 2.0 * deg2m_lat
        width = 5.0 * deg2m_lon
    # Do intelligent aspect calculation for local projection
    # adjust to figure dimensions
    w, h = fig.get_size_inches()
    aspect = w / h
    aspect *= 1.2
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect

    proj_kwargs = {}
    proj_kwargs['central_latitude'] = lat_0
    proj_kwargs['central_longitude'] = lon_0
    proj_kwargs['standard_parallels'] = [lat_0, lat_0]
    # proj = ccrs.AlbersEqualArea(**proj_kwargs)
    proj = NZTM

    map_ax = fig.add_subplot(gs[0:11, :], projection=proj)
    
    sizes = quakes.magnitude ** 2
    colors = depths
    norm = Normalize(vmin=min(colors), vmax=20.0)
    colormap = copy.copy(plt.get_cmap("plasma_r"))
    colormap.set_over(color="k")

    fig, map_ax, cbar_ax, cb = _blank_map(
        lats=lats, lons=lons, color=colors, projection="local", 
        resolution="full", colormap=colormap, figsize=map_size,
        proj_kwargs={}, norm=norm, continent_fill_color="0.65",
        water_fill_color="0.9", fig=fig, map_ax=map_ax, cm_ax=cbar_ax)

    map_ax.scatter(
        lons, lats, marker="o", s=sizes, c=colors, zorder=7, alpha=0.9,
        cmap=colormap, transform=ccrs.PlateCarree(), rasterized=True,
        norm=norm)

    # Outline those events prior to Kaikoura
    pre_kaik_mask = np.array(times) < dt.datetime(2016, 11, 1)
    map_ax.scatter(
        lons[pre_kaik_mask], lats[pre_kaik_mask], marker="o", 
        s=sizes[pre_kaik_mask], c=colors[pre_kaik_mask], zorder=8, alpha=0.9,
        cmap=colormap, transform=ccrs.PlateCarree(), rasterized=True,
        norm=norm, edgecolor="k")

    # Plot large eqs
    for eq, eq_name in [(grassmere, "Lake Grassmere"), (ck_st, "Cook Strait")]:
        ori = eq.preferred_origin()
        mag = eq.preferred_magnitude().mag
        ck_st_handle = map_ax.scatter(
            ori.longitude, ori.latitude, marker="*", s=5 * (mag ** 2),
            facecolor="lime", edgecolor="k", zorder=10, alpha=1.0, 
            transform=ccrs.PlateCarree(), rasterized=True)

    # Plot mechanisms
    # deep_mask = quakes.depth > 23000
    # s, d, r, lat, lon, _depth = (
    #         quakes[deep_mask].strike.to_list(), 
    #         quakes[deep_mask].dip.to_list(),
    #         quakes[deep_mask].rake.to_list(),
    #         quakes[deep_mask].latitude.to_list(),
    #         quakes[deep_mask].longitude.to_list(),
    #         quakes[deep_mask].depth.to_list())
    # for i in range(len(quakes[deep_mask])):
    #     if np.any(np.isnan((s[i], d[i], r[i]))):
    #         continue
    #     color = _depth[i] / 1000.
    #     red, green, blue, _alpha = colormap(norm(color))
    #     plot_fm(strike=s[i], dip=d[i], rake=r[i],
    #             latitude=lat[i], longitude=lon[i], zorder=20, axes=map_ax,
    #             width=fm_size, color=(red, green, blue), alpha=0.6, 
    #             rasterize=True)

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
                flons, flats, color="red", linewidth=2.0, zorder=9,
                transform=ccrs.PlateCarree(), rasterized=True)

    map_ax.set_extent(
        [min_longitude, max_longitude, min_latitude, max_latitude],
        crs=ccrs.PlateCarree())

    gl = map_ax.gridlines(
        draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gl.right_labels, gl.bottom_labels = False, False

    # Plot subduction contours
    subd_lats, subd_lons, subd_depths = get_williams_contours()
    lat_mask, lon_mask = (np.ones_like(subd_lats, dtype=bool), 
                          np.ones_like(subd_lons, dtype=bool))
    if min_latitude:
        lat_mask = np.logical_and(lat_mask, subd_lats >= min_latitude - 0.5)
    if max_latitude:
        lat_mask = np.logical_and(lat_mask, subd_lats <= max_latitude + 0.5)
    if min_longitude:
        lon_mask = np.logical_and(lon_mask, subd_lons >= min_longitude - 0.5)
    if max_longitude:
        lon_mask = np.logical_and(lon_mask, subd_lons <= max_longitude + 0.5)

    subd_lats = subd_lats[lat_mask]
    subd_lons = subd_lons[lon_mask]
    subd_depths = subd_depths[lat_mask][:, lon_mask] * -1
    contours = map_ax.contour(
        subd_lons, subd_lats, subd_depths, colors="k", linestyles="dashed",
        transform=ccrs.PlateCarree(), levels=[0, 5, 10, 15, 20, 25, 30, 40, 50],
        zorder=9)
    map_ax.clabel(contours, inline=1, fontsize=10, fmt="%i km", zorder=9)

    # Plot scale bar
    scale_bar(map_ax, (0.85, 0.05), scale_bar_length, angle=0)
    scale_bar(map_ax, (0.85, 0.05), scale_bar_length, angle=90)

    handles = [f_line, contours.collections[0], ck_st_handle]
    labels = ["Active Faults", "Williams et al. Interface", "Cook Strait earthquakes"]
    if kaik_f_line:
        handles.append(kaik_f_line)
        labels.append("Surface Rupture")

    fig.legend(handles=handles, labels=labels, framealpha=1.0, 
               loc="upper left").set_zorder(10000)

    # Plot x-section line
    name, points = section, CROSS_SECTIONS[section]
    x_start, x_end = points
    x_lats, x_lons = [x_start[0], x_end[0]], [x_start[1], x_end[1]]
    map_ax.plot(x_lons, x_lats, color="cyan", linewidth=3.0, zorder=2,
                transform=ccrs.PlateCarree(), linestyle="--")

    # Plot x-section
    fig = plot_x_section(
        earthquakes=relocated, start_latitude=CROSS_SECTIONS[section][0][0],
        start_longitude=CROSS_SECTIONS[section][0][1],
        end_latitude=CROSS_SECTIONS[section][1][0],
        end_longitude=CROSS_SECTIONS[section][1][1],
        starttime=MAINSHOCK_TIME,
        max_depth=30, swath_half_width=5, dip=90.0, colormap="turbo_r",
        size=None, logarithmic_color=True, color_by="timestamp", fig=fig,
        plot_mainshock=False, ax=section_ax)
    
    # Add on the relevant faults.
    print("Adding faults computed manually - if you have changed the section line yur fucked.")
    london = [(16.0, 19.44), (0.0, 10.0)]  # 70 degree dip at dip direction = 115.6, projected onto section at 132.4 degrees
    needles = [(23.5, 19.9), (0.0, 10.0)]  # 70 Degree dip opposite direction to london
    section_ax.plot(london[0], london[1], color="k", zorder=5, alpha=0.6)
    section_ax.plot(needles[0], needles[1], color="r", zorder=5, alpha=0.6)

    fig.subplots_adjust(hspace=1.5)
    
    for _format in PLOT_FORMATS:
        fig.savefig(f"{OUT}/Figure_5.{_format}", dpi=200)
    return fig


def fig_6(size=(7, 16)):
    """
    Point Keen cross-section with faults and FM

    """
    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})

    bounds = REGIONS["Fig. 6"]
    section = "thrust"

    relocated = earthquakes[earthquakes.station_count == 0.0]

    (min_latitude, min_longitude), (max_latitude, max_longitude) = bounds
    scale_bar_length = 5
    
    pad = 0.05
    map_size = size
    fig = plt.figure(figsize=map_size)

    gs = fig.add_gridspec(24, 1)
    
    cbar_ax = fig.add_subplot(gs[11, :])
    section_ax = fig.add_subplot(gs[13:, :])

    quakes = filter_earthquakes(
        relocated, min_longitude=min_longitude, max_longitude=max_longitude,
        min_latitude=min_latitude, max_latitude=max_latitude)

    quakes = quakes.sort_values(
        by="origintime", ignore_index=True, ascending=False)
    
    lats, lons, depths, times = (
        quakes.latitude.to_numpy(), quakes.longitude.to_numpy(),
        quakes.depth.to_numpy() / 1000., quakes.origintime.to_list())

    # Get the projection
    if min(lons) < -150 and max(lons) > 150:
        max_lons = max(np.array(lons) % 360)
        min_lons = min(np.array(lons) % 360)
    else:
        max_lons = max(lons)
        min_lons = min(lons)
    lat_0 = max(lats) / 2. + min(lats) / 2.
    lon_0 = max_lons / 2. + min_lons / 2.
    if lon_0 > 180:
        lon_0 -= 360
    deg2m_lat = 2 * np.pi * 6371 * 1000 / 360
    deg2m_lon = deg2m_lat * np.cos(lat_0 / 180 * np.pi)
    if len(lats) > 1:
        height = (max(lats) - min(lats)) * deg2m_lat
        width = (max_lons - min_lons) * deg2m_lon
        margin = pad * (width + height)
        height += margin
        width += margin
    else:
        height = 2.0 * deg2m_lat
        width = 5.0 * deg2m_lon
    # Do intelligent aspect calculation for local projection
    # adjust to figure dimensions
    w, h = fig.get_size_inches()
    aspect = w / h
    aspect *= 1.2
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect

    proj_kwargs = {}
    proj_kwargs['central_latitude'] = lat_0
    proj_kwargs['central_longitude'] = lon_0
    proj_kwargs['standard_parallels'] = [lat_0, lat_0]
    proj = ccrs.AlbersEqualArea(**proj_kwargs)
    proj = NZTM

    map_ax = fig.add_subplot(gs[0:11, :], projection=proj)
    
    sizes = quakes.magnitude ** 2
    colors = depths
    norm = Normalize(vmin=min(colors), vmax=20.0)
    colormap = copy.copy(plt.get_cmap("plasma_r"))
    colormap.set_over(color="k")

    fig, map_ax, cbar_ax, cb = _blank_map(
        lats=lats, lons=lons, color=colors, projection="local", 
        resolution="full", colormap=colormap, figsize=map_size,
        proj_kwargs={}, norm=norm, continent_fill_color="0.65",
        water_fill_color="0.9", fig=fig, map_ax=map_ax, cm_ax=cbar_ax)

    map_ax.scatter(
        lons, lats, marker="o", s=sizes, c=colors, zorder=7, alpha=0.9,
        cmap=colormap, transform=ccrs.PlateCarree(), rasterized=True,
        norm=norm)

    # Plot mechanisms
    # fm_mask = np.logical_and(quakes.rake > 45, quakes.rake < 125)
    fm_mask = np.ones(len(quakes), dtype=np.bool)
    s, d, r, lat, lon, _depth = (
            quakes[fm_mask].strike.to_list(), 
            quakes[fm_mask].dip.to_list(),
            quakes[fm_mask].rake.to_list(),
            quakes[fm_mask].latitude.to_list(),
            quakes[fm_mask].longitude.to_list(),
            quakes[fm_mask].depth.to_list())
    for i in range(len(quakes[fm_mask])):
        if np.any(np.isnan((s[i], d[i], r[i]))):
            continue
        color = _depth[i] / 1000.
        red, green, blue, _alpha = colormap(norm(color))
        plot_fm(strike=s[i], dip=d[i], rake=r[i],
                latitude=lat[i], longitude=lon[i], zorder=20, axes=map_ax,
                width=15, color=(red, green, blue), alpha=0.6, 
                rasterize=True)

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
                flons, flats, color="red", linewidth=2.0, zorder=9,
                transform=ccrs.PlateCarree(), rasterized=True)

    map_ax.set_extent(
        [min_longitude, max_longitude, min_latitude, max_latitude],
        crs=ccrs.PlateCarree())

    gl = map_ax.gridlines(
        draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gl.right_labels, gl.bottom_labels = False, False

    # Plot subduction contours
    subd_lats, subd_lons, subd_depths = get_williams_contours()
    lat_mask, lon_mask = (np.ones_like(subd_lats, dtype=bool), 
                          np.ones_like(subd_lons, dtype=bool))
    if min_latitude:
        lat_mask = np.logical_and(lat_mask, subd_lats >= min_latitude - 0.5)
    if max_latitude:
        lat_mask = np.logical_and(lat_mask, subd_lats <= max_latitude + 0.5)
    if min_longitude:
        lon_mask = np.logical_and(lon_mask, subd_lons >= min_longitude - 0.5)
    if max_longitude:
        lon_mask = np.logical_and(lon_mask, subd_lons <= max_longitude + 0.5)

    subd_lats = subd_lats[lat_mask]
    subd_lons = subd_lons[lon_mask]
    subd_depths = subd_depths[lat_mask][:, lon_mask] * -1
    contours = map_ax.contour(
        subd_lons, subd_lats, subd_depths, colors="k", linestyles="dashed",
        transform=ccrs.PlateCarree(), levels=[0, 5, 10, 15, 20, 25, 30, 40, 50],
        zorder=9)
    map_ax.clabel(contours, inline=1, fontsize=10, fmt="%i km", zorder=9)

    # Plot scale bar
    scale_bar(map_ax, (0.85, 0.05), scale_bar_length, angle=0)
    scale_bar(map_ax, (0.85, 0.05), scale_bar_length, angle=90)

    handles = [f_line, contours.collections[0], ck_st_handle]
    labels = ["Active Faults", "Williams et al. Interface", "Cook Strait earthquakes"]
    if kaik_f_line:
        handles.append(kaik_f_line)
        labels.append("Surface Rupture")

    fig.legend(handles=handles, labels=labels, framealpha=1.0, 
               loc="upper left").set_zorder(10000)

    # Plot x-section line
    name, points = section, CROSS_SECTIONS[section]
    x_start, x_end = points
    x_lats, x_lons = [x_start[0], x_end[0]], [x_start[1], x_end[1]]
    map_ax.plot(x_lons, x_lats, color="cyan", linewidth=3.0, zorder=20,
                transform=ccrs.PlateCarree(), linestyle="--")

    # Plot x-section
    fig = plot_x_section(
        earthquakes=relocated, start_latitude=CROSS_SECTIONS[section][0][0],
        start_longitude=CROSS_SECTIONS[section][0][1],
        end_latitude=CROSS_SECTIONS[section][1][0],
        end_longitude=CROSS_SECTIONS[section][1][1],
        starttime=MAINSHOCK_TIME,
        max_depth=25, swath_half_width=7.5, dip=90.0, colormap="turbo_r",
        size=None, logarithmic_color=True, color_by="timestamp", fig=fig,
        plot_mainshock=False, ax=section_ax)
    
    # Add on the relevant faults.
    print("Adding faults computed manually - if you have changed the section line yur fucked.")
    section_faults = {
        "Manakau": [(11.25, 16.55), (0.0, 15.0)], # 70 deg SE projected onto section
        "Upper Kowahi": [(13.125, 7.825), (0.0, 15.0)], # 70 deg NW projected onto section
        "Hope": [(21.875, 16.415), (0.0, 15.0)], # 70 deg NW - assuming striking orthogonal to section
        "Point Keen": [(34.375, 23.875), (0.0, 15.0)],  # 55 deg NW - assuming striking orthogonal to section
    }
    for fault in section_faults.values():
        section_ax.plot(fault[0], fault[1], color="k", zorder=5, alpha=0.6)

    fig.subplots_adjust(hspace=1.5)
    
    for _format in PLOT_FORMATS:
        print(f"Saving as {_format}")
        fig.savefig(f"{OUT}/Figure_6.{_format}", dpi=200)
    return fig


def fig_7(size=(12.8, 9.6)):
    """ Cartoon of Papatea thrust block. """
    pass

def fig_8(size=(12, 8)):
    """
    Along-strike cross-section with aftershock density and slip models

    """
    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})
    relocated = earthquakes[earthquakes.station_count == 0.0]

    points = CROSS_SECTIONS["A"]
    x_section =  plot_x_section(
        earthquakes=relocated, start_latitude=points[0][0],
        start_longitude=points[0][1], end_latitude=points[1][0],
        end_longitude=points[1][1], max_depth=40.0,
        swath_half_width=30.0, dip=90.0, starttime=MAINSHOCK_TIME,
        endtime=None, logarithmic_color=True, size=None, colormap=TIME_COLORMAP,
        color_by="timestamp", plot_mainshock=True)
    x_section.set_size_inches(size)

    # Add subplot and adjust
    x_section_ax = x_section.gca()
    density_ax = x_section.add_subplot(3, 1, 1, sharex=x_section_ax)
    x_section.subplots_adjust(hspace=0.0)
    plt.setp(density_ax.xaxis.get_ticklabels(), visible=False)

    # Project
    quakes = filter_earthquakes(
        relocated, max_depth=50.0, starttime=MAINSHOCK_TIME,
        endtime=None)
    
    projected = project_onto_section(
        earthquakes=quakes, start_latitude=points[0][0],
        start_longitude=points[0][1], end_latitude=points[1][0],
        end_longitude=points[1][1], swath_half_width=30.0,
        dip=90)

    y = np.array([loc.y for loc in projected])
    # Compute density in bins
    bin_width = .5
    bin_mid_points = np.arange(bin_width / 2, y.max(), bin_width)
    quake_density = np.zeros_like(bin_mid_points)
    for i in range(len(bin_mid_points)):
        bin_start = bin_mid_points[i] - (.5 * bin_width)
        bin_end = bin_start + bin_width
        quake_density[i] = np.logical_and(y < bin_end, y > bin_start).sum()

    quake_handle, = density_ax.plot(
        bin_mid_points, quake_density, label="Earthquake density")
    density_ax.set_ylim(0, quake_density.max() + 10)
    density_ax.set_ylabel("Number of aftershocks in bin")

    # Plot Ulrich slip model
    ulrich_model = pd.read_csv(
        f"{HOME}/Dropbox/Current_projects/Kaikoura_afterslip/"
        "Ulrich_etal_dynamic_model/Ulrich_SlipXYZ_LATLON.csv")
    # Mask slip > 30 m
    # ulrich_model = ulrich_model[ulrich_model.slip < 30.0]
    # Rename for projection
    ulrich_model = ulrich_model.rename(
        columns={"slip": "magnitude", "Latitude": "latitude", 
                 "Longitude": "longitude", "Z": "depth"})
    # Make depth positive
    ulrich_model.depth = ulrich_model.depth * -1
    # Add a dummy origintime column
    ulrich_model["origintime"] = [
        dt.datetime(2016, 11, 13) for _ in range(len(ulrich_model))]
    # Project
    ulrich_projected = project_onto_section(
        earthquakes=ulrich_model, start_latitude=points[0][0],
        start_longitude=points[0][1], end_latitude=points[1][0],
        end_longitude=points[1][1], swath_half_width=30.0,
        dip=90)
    y, slip, z = (np.array([loc.y for loc in ulrich_projected]),
                  np.array([loc.magnitude for loc in ulrich_projected]),
                  np.array([loc.z for loc in ulrich_projected]))
    z *= -1

    # Compute total slip in that bin
    ulrich_slip_density = np.zeros_like(bin_mid_points)
    for i in range(len(bin_mid_points)):
        bin_start = bin_mid_points[i] - (.5 * bin_width)
        bin_end = bin_start + bin_width
        mask = np.logical_and(y < bin_end, y > bin_start)
        masked_z = z[mask]
        if len(masked_z):
            ulrich_slip_density[i] = slip[mask].sum() / (masked_z.max())

    slip_ax = density_ax.twinx()
    ulrich_handle, = slip_ax.plot(
        bin_mid_points, ulrich_slip_density, label="Total Slip", color="red")
    slip_ax.set_ylim((0, 1500))
    slip_ax.set_ylabel("Total slip (m) / total depth (m)")

    density_ax.legend(
        handles=[quake_handle, ulrich_handle],
        labels=["Aftershock density", "Ulrich et al. slip model"])

    for _format in PLOT_FORMATS:
        x_section.savefig(f"{OUT}/Figure_8.{_format}")

    return x_section


def compute_area(
    min_longitude: float, 
    max_longitude: float, 
    min_latitude: float, 
    max_latitude: float
) -> float:
    import pyproj    
    import shapely.ops as ops
    from shapely.geometry.polygon import Polygon
    from functools import partial


    geom = Polygon([
        (min_longitude, min_latitude), 
        (max_longitude, min_latitude),
        (max_longitude, max_latitude), 
        (min_longitude, max_latitude), 
        (min_longitude, min_latitude)])
    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat_1=geom.bounds[1],
                lat_2=geom.bounds[3])),
        geom)

    return geom_area.area / 1e6


def fig_9(size=(9.6, 12.8)):
    """ GPS and cumulative EQs. """
    from gps_data_play import GPSStation
    from kaikoura_csv_visualisations import filter_earthquakes
    import matplotlib.pyplot as plt

    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})

    starttime, endtime = dt.datetime(2016, 11, 14), dt.datetime(2019, 11, 1)

    earthquakes = earthquakes.sort_values(by="origintime", ignore_index=True)
    earthquakes = filter_earthquakes(earthquakes, starttime=starttime, endtime=endtime)

    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=size)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    regions = {
        "Cape Campbell": dict(
            stations=["CMBL"], region=(174.1, 174.5, -41.85, -41.69), ax=axes[0]),
        "Kekerengu": dict(
            stations=["LOK1", "GLOK"], region=(173.86, 174.1, -42.07, -41.85), ax=axes[1]),
        "Snowgrass": dict(
            stations=["MUL1", "LOOK"], region=(173.7, 173.86, -42.3, -42), ax=axes[2]),
        "Kaikoura": dict(
            stations=["KAIK"], region=(173.4, 173.7, -42.6, -42.3), ax=axes[3]),
        "Epicentral": dict(
            stations=["MRBL"], region=(172.77, 173.4, -42.76, -42.6), ax=axes[4])}

    handles, labels = [], []

    for region_name in regions.keys():
        region = regions[region_name]
        _region = region["region"]
        quakes = filter_earthquakes(
            earthquakes, min_longitude=_region[0], max_longitude=_region[1],
            min_latitude=_region[2], max_latitude=_region[3])
        cumulative = np.arange(len(quakes)) / compute_area(*_region)
        handle, = region["ax"].plot(quakes.origintime, cumulative, "--")
        if "Cumulative density" not in labels:
            labels.append("Cumulative density")
            handles.append(handle)
        
        gps_ax = region["ax"].twinx()

        gps_stations = [GPSStation.from_geonet(sta) for sta in region["stations"]]
        fudge_factors = {}
        for station in gps_stations:
            try:
                station.detrend(dt.datetime(2015, 1, 1), dt.datetime(2016, 11, 1))
            except:
                print(f"Could not detrend for {station[0].reciever}")
            station = station.trim(starttime, endtime).zero_start()
            for component, name, color in zip(("u", "n", "e"), ("Vertical", "North", "East"), colors):
                displacement = station.select(component)[0]
                times, data = displacement.times, displacement.observations
                if name in fudge_factors.keys():
                    # Apply a shift - only works if there is overlap of data
                    loc = np.where(times == fudge_factors[name][0])
                    if len(loc) == 0:
                        print("Cannot fudge")
                        continue
                    delta = fudge_factors[name][1] - data[loc[0][0]]
                    data += delta
                handle, = gps_ax.plot(times, data, zorder=2, color=color)
                if name not in labels:
                    labels.append(name)
                    handles.append(handle)
                # Keep track of the end of the data
                fudge_factors.update(
                    {name: (displacement.times[-1], displacement.observations[-1])})
        gps_ax.set_ylabel(f'{", ".join(region["stations"])} (mm)')
        gps_ax.grid("on")
        region["ax"].set_ylabel("Earthquakes per $km^2$")
    
    fig.legend(handles=handles, labels=labels)
    axes[-1].set_xlim(starttime, endtime)
    fig.subplots_adjust(hspace=0)


    for _format in PLOT_FORMATS:
        fig.savefig(f"{OUT}/Figure_9.{_format}")


def plot_all_gps(size=(18, 18)):
    from gps_data_play import GPSStation
    import matplotlib.pyplot as plt

    gps_stations = ("WITH", "SEDD", "CMBL", "LOK1", "GLOK", "GDS1", "LOOK", 
                    "MUL1", "CLRR", "KAIK", "HANM", "MRBL")



    fig, axes = plt.subplots(nrows=len(gps_stations), ncols=2, 
                             sharex=True, sharey=True, figsize=size)
    
    component_label_mapper = {
        "u": "Vertical displacement",
        "n": "North displacement",
        "e": "East displacement"}
    component, normalize, starttime, endtime, moment, split_on_fm, plot_errors = (
        "all", False, dt.datetime(2016, 11, 14),
        dt.datetime(2019, 11, 1), False, False, True)

    for station, row in zip(gps_stations, axes):
        ax, detrended_ax = row
        handles, labels = [], []
        gps_data = GPSStation.from_geonet(station)
        detrended = True
        try:
            gps_data_detrended = gps_data.copy().detrend(
                dt.datetime(2015, 1, 1), dt.datetime(2016, 11, 1))
        except Exception as e:
            detrended, gps_data_detrended = False, gps_data
            print(f"Could not detrend {station} due to {e}")
            detrended_ax.set_facecolor("lightgrey")
        gps_data = gps_data.trim(starttime, endtime).zero_start()
        if detrended:
            gps_data_detrended = gps_data_detrended.trim(starttime, endtime).zero_start()
        else:
            gps_data_detrended = None
        gps_times = gps_data[0].times
        for _ax, _gps_data in zip((ax, detrended_ax), (gps_data, gps_data_detrended)):
            if _gps_data is None:
                continue
            for _component in ("u", "n", "e"):
                _disp = _gps_data.select(_component)[0]
                if normalize:
                    _disp.observations /= _disp.observations[-1]
                handle = _ax.plot(gps_times, _disp.observations, zorder=2)[0]
                if plot_errors:
                    _ax.fill_between(
                        gps_times, _disp.observations + _disp.errors,
                        _disp.observations - _disp.errors, alpha=0.4)
                label = component_label_mapper[_component]
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            _ax.grid("on")
        ax.set_ylabel(f"{station} (mm)")
        
    fig.legend(handles=handles, labels=labels, loc="lower right",
              facecolor="white", framealpha=1.0)
    fig.subplots_adjust(wspace=0.01, hspace=0)
    axes[-1][0].set_xlim(starttime, endtime)
    axes[-1][0].set_ylim(-50, 350)
    axes[0][0].set_title("Raw")
    axes[0][1].set_title("Detrended between 2015-01-01 and 2016-11-01")

    for _format in PLOT_FORMATS:
        fig.savefig(f"{OUT}/Supplementary_Figure_GPS.{_format}")



def plot_donna_qs(
    x: float = 93.0,
    y: float = None,
    max_depth: float = 40.0,
    min_depth: float = 0.0,
    min_latitude: float = None,
    max_latitude: float = None,
    min_longitude: float = None,
    max_longitude: float = None,
    contour_vals: List[float] = None,
    label_contours: bool = False,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> Tuple[plt.Figure, Tuple[Tuple[float, float], Tuple[float, float]]]:

    assert x or y, "Requires either an X, or a Y to extract on"

    if fig and not ax:
        ax = fig.gca()
    elif ax and not fig:
        fig = ax.get_figure
    elif not ax and not fig:
        fig, ax = plt.subplots()

    qs = pd.read_csv(
        "../Locations/NonLinLoc_NZW3D_2.2/NZW3D_2.2/Qsnzw2p2xyzltln.tbl.txt",
        header=1, delim_whitespace=True)
    
    if x:
        if x not in set(qs["x(km)"]):
            raise NotADirectoryError(f"{x} is not in {set(qs['x(km)'])}")
        qs_section = qs[qs["x(km)"] == x]
    else:
        if y not in set(qs["y(km)"]):
            raise NotADirectoryError(f"{y} is not in {set(qs['y(km)'])}")
        qs_section = qs[qs["y(km)"] == y]
    
    mask = np.ones(len(qs_section)).astype(np.bool)
    if not min_latitude is None:
        mask = np.logical_and(mask, qs_section.Latitude > min_latitude)
    if not max_latitude is None:
        mask = np.logical_and(mask, qs_section.Latitude < max_latitude)
    if not min_longitude is None:
        mask = np.logical_and(mask, qs_section.Longitude > min_longitude)
    if not max_longitude is None:
        mask = np.logical_and(mask, qs_section.Longitude < max_longitude)
    if not min_depth is None:
        mask = np.logical_and(mask, qs_section["Depth(km_BSL)"] > min_depth)
    if not max_depth is None:
        mask = np.logical_and(mask, qs_section["Depth(km_BSL)"] < max_depth)

    qs_section = qs_section[mask]
    
    # Get dist (x-section distance) and Y (depth) values
    y = np.array(list(set(qs_section["Depth(km_BSL)"])))
    if x:
        dist = np.array(list(set(qs_section["y(km)"])))
    else:
        dist = np.array(list(set(qs_section["x(km)"])))
    y.sort()
    dist.sort()

    qs_vals = np.zeros((len(dist), len(y)))
    for i, _x in enumerate(dist): 
        for j, _depth in enumerate(y): 
             val = qs_section["Qs"][np.logical_and(
                 qs_section["Depth(km_BSL)"] == _depth, 
                 qs_section["y(km)"] == _x)].to_list()[0] 
             qs_vals[i, j] = val

    # Get the lat and lon of the start and end
    if x:
        sortby = "x(km)"
    else:
        sortby = "y(km)"
    qs_section.sort_values(by=[sortby, "Depth(km_BSL)"], inplace=True, 
                           ignore_index=True)
    start_lat, start_lon = qs_section.Latitude[0], qs_section.Longitude[0]
    end_lat, end_lon = (qs_section.Latitude.to_list()[-1], 
                        qs_section.Longitude.to_list()[-1])
    start = (start_lat, start_lon)
    end = (end_lat, end_lon)

    # Flip the y to a more normal orientation...
    if x:
        dist *= -1
        # Shift the origin
        dist -= min(dist)
        # Flip the start and end returned
        start, end = end, start

    if contour_vals:
        contours = ax.contour(dist, y, qs_vals.T, levels=contour_vals)
    else:
        contours = ax.contour(dist, y, qs_vals.T)
    if label_contours:
        ax.clabel(contours, inline=False, fontsize=10)
    ax.invert_yaxis()
    ax.set_ylabel("Depth (km)")
    # ax.invert_xaxis()


    return fig, (start, end)
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make figures")
    parser.add_argument(
        "-f", "--figure", type=int, help="Which figure number to plot",
        default=None, required=False)
    
    args = parser.parse_args()
    
    figures = {1: fig_1, 2: fig_2, 3: fig_3, 4: fig_4, 5: fig_5, 6: fig_6,
               7: fig_7, 8: fig_8, 9: fig_9}
    
    if args.figure:
        figures[args.figure]()
    else:
        for fig_number, fig in figures.items():
            print(f"Making figure {fig_number}")
            fig()

