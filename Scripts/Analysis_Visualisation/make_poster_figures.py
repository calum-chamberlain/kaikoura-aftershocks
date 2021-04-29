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
from make_paper_figures import plot_subduction_zone, plot_topography

import os
import copy
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

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

OUT = f"{HOME}/Dropbox/Conferences/SSA_2021/Kaikoura_poster"
PLOT_FORMATS = ("png", "svg", "pdf")

NZTM = ccrs.TransverseMercator(
    central_longitude=173.0, central_latitude=0.0, 
    false_easting=1600000, false_northing=10000000, 
    scale_factor=0.9996)

ROTATED_PROJ = ccrs.RotatedPole(
    pole_longitude=60.0, pole_latitude=60.0,
    central_rotated_longitude=170.0)

plt.style.use("default")


def rotated_map(size=(20, 7)):
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

    proj = ROTATED_PROJ
    xs, ys, zs = proj.transform_points(
        ccrs.PlateCarree(), 
        np.array([min_longitude, max_longitude]),
        np.array([min_latitude, max_latitude])).T

    map_ax = fig.add_axes([ax_x0, ax_y0, ax_width, ax_height],
                          projection=proj)
    
    # Try to get rid of the squashed effect
    map_ax.set_xlim((xs[0] + .2, xs[1]))
    map_ax.set_ylim((ys[0] - 0.3, ys[1] + 0.2))
    map_ax.set_aspect(1.4)

    # Add topo
    plot_topography(
        map_ax, min_latitude=min_latitude - .5, min_longitude=min_longitude - 5,
        max_latitude=max_latitude + .5, max_longitude=max_longitude + .5, cmap="Greys",
        clip=2, hillshade=False)  # Hillshade doesn't work for rotated

    coast = cfeature.GSHHSFeature(
        scale="full", levels=[1], facecolor="none", 
        edgecolor="0.4")
    map_ax.set_facecolor("white")
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
    
    # # Plot Seismographs
    seismographs = pd.read_csv(
        "seismograph_locations.csv", parse_dates=["Start", "End"])
    # seismograph_markers = map_ax.scatter(
    #     seismographs.Longitude, seismographs.Latitude, facecolor="orange",
    #     edgecolor="k", marker="v", zorder=10, transform=ccrs.PlateCarree(),
    #     s=200)
    # for lat, lon, code in zip(seismographs.Latitude, seismographs.Longitude, seismographs["Station code"]):
    #     if min_latitude < lat and lat < max_latitude and min_longitude < lon and lon < max_longitude:
    #         map_ax.text(lon - 0.02, lat - 0.01, code,
    #                     transform=ccrs.PlateCarree(), zorder=100, clip_on=True,
    #                     ha="right")

    # Plot subduction zone
    contours = plot_subduction_zone(
        map_ax=map_ax, min_latitude=min_latitude, min_longitude=min_longitude,
        max_latitude=max_latitude, max_longitude=max_longitude)

    # Plot the earthquakes!
    earthquakes = pd.read_csv(RELOCATED_EVENTS, parse_dates=["time"])
    earthquakes = earthquakes.rename(columns={"time": "origintime"})
    quakes = earthquakes[earthquakes.station_count == 0.0]
    quakes = quakes.sort_values(
        by="origintime", ignore_index=True, ascending=False)

    lats, lons, depths, times = (
        quakes.latitude.to_numpy(), quakes.longitude.to_numpy(),
        quakes.depth.to_numpy() / 1000., quakes.origintime.to_list())
    sizes = quakes.magnitude ** 2
    colors = depths
    norm = Normalize(vmin=0.0, vmax=20.0)
    colormap = copy.copy(plt.get_cmap(DEPTH_COLORMAP))
    colormap.set_over(color="k")

    map_ax.scatter(
        lons, lats, marker="o", s=sizes, c=colors, zorder=10, alpha=0.9,
        cmap=colormap, transform=ccrs.PlateCarree(), rasterized=True,
        norm=norm)

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

    # map_ax.set_extent(
    #     [min_longitude, max_longitude, min_latitude, max_latitude], 
    #     crs=ccrs.PlateCarree())

    # Plot scale bar
    scale_bar(map_ax, (0.05, 0.05), 20, angle=0)
    scale_bar(map_ax, (0.05, 0.05), 20, angle=90)

    # Add colorbar
    cm_ax = fig.add_axes([0.1, 0.05, 0.8, 0.05])
    cb = ColorbarBase(
        cm_ax, norm=norm, cmap=colormap, orientation='horizontal')
    cb.set_label("Depth (km)")

    ############################################################################

    # Add inset

    big_max_lat, big_min_lat, big_min_lon, big_max_lon = (
        -36, -47, 166, 178.5)
    big_map_ax = fig.add_axes([0.65, 0.05, 0.35, 0.35], projection=NZTM)
    
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
    
    seismograph_markers = big_map_ax.scatter(
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
    # big_map_ax.plot(
    #     (min_longitude, max_longitude, max_longitude, min_longitude, min_longitude),
    #     (min_latitude, min_latitude, max_latitude, max_latitude, min_latitude),
    #     color="red", linewidth=2.0, transform=ccrs.PlateCarree())

    big_map_ax.set_extent(
        [big_min_lon, big_max_lon, big_min_lat, big_max_lat],
        crs=ccrs.PlateCarree())

    fig.legend(
        handles=(f_line, kaik_f_line, mainshock, contours.collections[0],
                 seismograph_markers, gps_markers),
        labels=("Active Faults", "Surface Ruptures", "Mainshock", 
                "Williams et al. Interface", "Seismograph", "cGNSS"),
        framealpha=1.0)
    # Save
    for _format in PLOT_FORMATS:
        fig.savefig(f"{OUT}/Main_map.{_format}")
    return fig


if __name__ == "__main__":
    rotated_map()
