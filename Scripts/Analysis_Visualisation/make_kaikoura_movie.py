"""
Use matplotlib animation to make a movie of Kaikoura

"""
import os
import json
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import cartopy.crs as ccrs

from catalog_to_dict import dict_to_catalog
from cjc_utilities.animator.animator import (
    AnimatedCatalog, _blank_map, _get_plot_coords)

import matplotlib.animation as animation
from matplotlib.colors import Normalize

from obspy.imaging.cm import obspy_sequential
from obspy.clients.fdsn import Client
from obspy import read_events


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger("Kaikoura movie maker")

WORKING_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))


def read_gmt_file(filename):
    with open(filename, "rb") as f:
        lines = f.read().decode().splitlines()
    feature, features = ([], [])
    for line in lines:
        if line.startswith("#"):
            continue
        if line == ">":
            if len(feature) > 0:
                features.append(feature)
                feature = []
            continue
        feature.append(tuple([float(l) for l in line.split()]))
    if len(feature) > 0:
        features.append(feature)
    return features


def plot_fault_db(ax):
    import os
    from matplotlib.collections import LineCollection

    fault_file = "/home/chambeca/.gmt_data/faults_NZ_WGS84.gmt"
    if not os.path.isfile(fault_file):
        fault_file = (
            "/Volumes/GeoPhysics_09/users-data/chambeca/.gmt_data/"
            "faults_NZ_WGS84.gmt")
    faults = read_gmt_file(fault_file)
    # for fault in faults:
    #     lons, lats = zip(*fault)
    #     ax.plot(lons, lats, c="red", transform=ccrs.Geodetic(), linewidth=0.5)
    lc = LineCollection(
        faults, colors="red", transform=ccrs.Geodetic(), linewidth=0.5)
    ax.add_collection(lc)
    return ax


if __name__ == "__main__":

    catalog_file = (f"{WORKING_DIR}/Locations/SIMUL_located.xml")

    client = Client("GEONET")

    Logger.info("Starting")
    if catalog_file.endswith(".json"):
        with open(catalog_file, "rb") as f:
            catalog_dict = json.load(f)
        Logger.info("Read in dictionary")
        catalog = dict_to_catalog(catalog_dict)
        Logger.info("Converted to catalog")
    else:
        catalog = read_events(catalog_file)
    filtered_catalog = catalog.filter("depth < 60000.0", "latitude < -41.0")
    Logger.info("Filtered deep events. There are now {0} events".format(
        len(filtered_catalog)))
    animated_catalog = AnimatedCatalog(filtered_catalog)

    # Generate blank map with faults and epicentres
    Logger.info("Building base map")
    lats, lons, mags, colors = _get_plot_coords(animated_catalog)
    norm = Normalize(vmin=min(colors), vmax=max(colors))
    fig, map_ax = _blank_map(
        lons, lats, colors, projection="local", resolution="f",
        continent_fill_color='0.9',
        water_fill_color='1.0', colormap=obspy_sequential,
        title="Kaikoura Detections 2013-2019", 
        color_label="Depth (km)", figsize=(15, 15))
    # Get the Cook Strait and Kaikoura epicenters
    mainshocks = client.get_events(eventid="2016p858000") # Kaikoura
    mainshocks += client.get_events(eventid="2013p613797") # Grassmere
    mainshocks += client.get_events(eventid="2013p543824") # Cook Strait
    lats, lons, mags, colors = _get_plot_coords(mainshocks)
    min_size = 1.0
    max_size = 30
    min_size_ = min(mags) - 1
    max_size_ = max(mags) + 1
    frac = [(0.2 + (_i - min_size_)) / (max_size_ - min_size_)
            for _i in mags]
    size_plot = [(_i * (max_size - min_size)) ** 2 for _i in frac]
    map_ax.scatter(
        lons, lats, marker="*", s=size_plot, c='gold', edgecolors="black",
        zorder=11, transform=ccrs.Geodetic(),
        alpha=1.0, norm=norm)
    # Get the NZ fault DB
    plot_fault_db(ax=map_ax)
    # Run the animation
    Logger.info("Starting animation")
    fig = animated_catalog.animate(
        fig=fig, decay=48, time_step=3600, show=False)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    Logger.info("Writing mp4")
    fig.save("Kaikoura_detections.mp4", writer=writer)
    Logger.info("Fin")
