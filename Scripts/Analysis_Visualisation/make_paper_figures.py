import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from typing import Tuple, List

from kaikoura_csv_visualisations import (
    plot_locations, get_kaikoura_faults, get_williams_contours, read_faults,
    scale_bar, plot_x_section, project_onto_section, filter_earthquakes)

import os

HOME = os.path.expanduser("~")
# Set globals for all figures
TIME_COLORMAP = "turbo_r"
DEPTH_COLORMAP = "plasma_r"
RAKE_COLORMAP = "cividis"  # three colors merged - plot as cosine of rake?

BOUNDS = (-43.0, 172.5, -41.5, 174.7)
MAX_DEPTH = 40.0
CROSS_SECTIONS = {
    "A": [(-42.76, 172.824), (-41.65296, 174.5287)],  # Along strike
    "epi": [(-42.738, 172.741), (-42.612, 173.031)],  # Epicentral
    "B": [(-42.433, 173.9), (-42.115, 173.639)],  # Point Keen
    "C": [(-42.119, 173.814), (-41.953, 173.644)],  # Clarence link
    "D": [(-41.827, 174.385), (-41.636, 174.147)],  # Cape Campbell
    "E": [(-41.958, 174.004), (-41.65296, 174.5287)]
}
RELOCATED_EVENTS = "../../Locations/GrowClust_located_magnitudes_callibrated_focal_mechanisms.csv"

MAINSHOCK = (-42.626466, 172.990578, 12.390625)  # My NLL location.
MAINSHOCK_TIME = dt.datetime(2016, 11, 13, 11, 2, 56)  # GeoNet origin time
# Shift the start of plots to a bit before
MAINSHOCK_TIME -= dt.timedelta(seconds=30)

OUT = f"{HOME}/Dropbox/Current_projects/Kaikoura_afterslip/Plots/Paper_figures"
PLOT_FORMATS = ("png", "pdf", "svg")

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
    proj = ccrs.AlbersEqualArea(**proj_kwargs)

    map_ax = fig.add_axes([ax_x0, ax_y0, ax_width, ax_height],
                          projection=proj)

    land = cfeature.NaturalEarthFeature(cfeature.LAND.category,
                                        cfeature.LAND.name, "10m",
                                        edgecolor='face', facecolor='none')
    ocean = cfeature.NaturalEarthFeature(cfeature.OCEAN.category,
                                         cfeature.OCEAN.name, "10m",
                                         edgecolor='face',
                                         facecolor='none')
    map_ax.set_facecolor("0.65")
    map_ax.add_feature(ocean, facecolor="0.65")
    map_ax.add_feature(land, facecolor="0.9")
    map_ax.coastlines(resolution="10m", color='0.4')

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
    
    land_big = cfeature.NaturalEarthFeature(
        cfeature.LAND.category, cfeature.LAND.name, "50m",
        edgecolor='face', facecolor='none')
    ocean_big = cfeature.NaturalEarthFeature(
        cfeature.OCEAN.category, cfeature.OCEAN.name, "50m",
        edgecolor='face', facecolor='none')

    big_map_ax.set_facecolor("0.65")
    big_map_ax.add_feature(ocean_big, facecolor="0.65")
    big_map_ax.add_feature(land_big, facecolor="0.9")
    big_map_ax.coastlines(resolution="50m", color='0.4')

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
    Map of relocated seismicity, and focal mechanism map of templates

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
    for name, points in CROSS_SECTIONS.items():
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

    for _format in PLOT_FORMATS:
        relocated_map.savefig(f"{OUT}/Figure_2a.{_format}")

    # TODO: Plot fig 2b with focal mechanisms
    return    


def fig_3(size=(12, 8)):
    """
    Along-strike cross-section with aftershock density and slip models

    TODO:
     - Other slip models?
     - Qs behind aftershocks

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
        x_section.savefig(f"{OUT}/Figure_3.{_format}")

    return x_section


def fig_4(size=(12.8, 9.6)):
    """
    Point Keen cross-section with faults and FM

    TODO:

     - Plot Point Keen projection and other faults

    """
    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})
    relocated = earthquakes[earthquakes.station_count == 0.0]

    points = CROSS_SECTIONS["B"]
    x_section =  plot_x_section(
        earthquakes=relocated, start_latitude=points[0][0],
        start_longitude=points[0][1], end_latitude=points[1][0],
        end_longitude=points[1][1], max_depth=20.0,
        swath_half_width=10.0, dip=90.0, starttime=MAINSHOCK_TIME,
        endtime=None, logarithmic_color=True, size=None, colormap=TIME_COLORMAP,
        color_by="timestamp", plot_mainshock=False, focal_mechanisms=True)
    x_section.set_size_inches(size)
    x_section.set_facecolor("None")
    
    for _format in PLOT_FORMATS:
        x_section.savefig(f"{OUT}/Figure_4.{_format}")
    return x_section


def fig_5(size=(8, 4)):
    """
    Clarence -- Papatea link along-strike, color by time, plot projection of
    Clarence

    TODO: 

     - Plot Clarence projection

    """
    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})
    relocated = earthquakes[earthquakes.station_count == 0.0]

    points = CROSS_SECTIONS["C"]
    x_section =  plot_x_section(
        earthquakes=relocated, start_latitude=points[0][0],
        start_longitude=points[0][1], end_latitude=points[1][0],
        end_longitude=points[1][1], max_depth=20.0,
        swath_half_width=10.0, dip=90.0, starttime=MAINSHOCK_TIME,
        endtime=None, logarithmic_color=True, size=None, colormap=TIME_COLORMAP,
        color_by="timestamp", plot_mainshock=False)
    x_section.set_size_inches(size)
    
    for _format in PLOT_FORMATS:
        x_section.savefig(f"{OUT}/Figure_5.{_format}")
    return x_section


def fig_6(size=(8, 8)):
    """
    Cape Campbell sections - illustrate hard stop (plot over Qs model),
    illustrate many faults ruptured.

    TODO:
     - Plot QS
     - Plot Fault projections
     - Plot 2013 earthquakes

    """
    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})
    relocated = earthquakes[earthquakes.station_count == 0.0]

    fig, ax = plt.subplots()
    fig, (start, end) = plot_donna_qs(
        min_latitude=-42.5, max_latitude=-41.5, fig=fig, x=86,
        contour_vals=[225.0])

    # Plot strike-parallel
    fig =  plot_x_section(
        earthquakes=relocated, start_latitude=start[0],
        start_longitude=start[1], end_latitude=end[0],
        end_longitude=end[1], max_depth=25.0,
        swath_half_width=10.0, dip=90.0, starttime=MAINSHOCK_TIME,
        endtime=None, logarithmic_color=True, size=None, colormap=TIME_COLORMAP,
        color_by="timestamp", plot_mainshock=False, fig=fig)

    fig.set_size_inches(size)
    
    for _format in PLOT_FORMATS:
        fig.savefig(f"{OUT}/Figure_6.{_format}")
    return fig


def fig_7(size=(8, 12)):
    """
    Temporal evolution of afterslip in regions with GPS
    """
    return


def fig_8(size=(12, 8)):
    """
    Space-time plot with logarithmic bounds?

    """
    return


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
               7: fig_7, 8: fig_8}
    
    if args.figure:
        figures[args.figure]()
    else:
        for fig_number, fig in figures.items():
            print(f"Making figure {fig_number}")
            fig()

