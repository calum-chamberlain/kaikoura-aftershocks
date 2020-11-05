"""
Make figures for Kaikoura Jan 2019 workshop.

"""

import numpy as np
import glob

from itertools import cycle
from obspy import UTCDateTime

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)

root_dir = "/Volumes/GeoPhysics_09/users-data/chambeca/Kaik_afterslip"
plot_dir = "{0}/Plots".format(root_dir)
detect_dir = "{0}/Detections".format(root_dir)
COLORS = [
    "red", "blue", "green", "purple", "cyan", "orange", "darkgreen", "pink",
    "cyan", "slateblue", "navy", "aquamarine", "seagreen", "chartreuse", 
    "khaki", "gold", "saddlebrown", "lightsalmon", "darkmagenta", "black"]


def gmt_video(catalog, step=10, outfile="catalog_gif.gif", 
              region=[172.5, 174.6, -43.1, -41.1], frame_length=0.25,
              max_previous=10):
    """
    Catalog to plot and frame size in days
    """
    import os
    import imageio
    import shutil
    from collections import deque
    from gmt.clib import Session

    from obspy import Catalog

    working_dir = "gif_working"
    max_size = 1

    previous_sizes = [(i + 1) * (max_size / (max_previous + 1))
                      for i in range(max_previous)]

    catalog = sorted(catalog, key=lambda e: e.origins[0].time)
    ndays = (catalog[-1].origins[0].time - catalog[0].origins[0].time) / 86400

    frames = int((ndays // step) + 1)

    os.makedirs(working_dir)

    plot_start = UTCDateTime(catalog[0].origins[0].time.date)
    previous_chunks = deque(maxlen=max_previous)
    working_files = []
    _newdepth = True
    for frame in range(frames):
        frame_plot = None
        print("Plotting frame {0} of {1}".format(frame, frames))
        # Plot previous
        n_previous_chunks = len(previous_chunks)
        if n_previous_chunks > 0:
            size = previous_sizes[-n_previous_chunks]
            _cat = previous_chunks[0]
            if len(_cat) > 0:
                frame_plot = map_plot(
                    catalog=_cat, region=region,
                    sizes=[size] * len(previous_chunks[0]), normaliser=5,
                    new_depthcpt=False, max_depth=30, min_depth=0)
            if n_previous_chunks > 1:
                for i in range(1, n_previous_chunks):
                    size = previous_sizes[i]
                    _cat = previous_chunks[i]
                    if len(_cat) > 0:
                        frame_plot = map_plot(
                            catalog=_cat, region=region,
                            sizes=[size] * len(previous_chunks[i]),
                            fig=frame_plot, normaliser=5, new_depthcpt=False, 
                            max_depth=30, min_depth=0)
        plot_end = plot_start + (step * 86400)
        chunk_cat = [e for e in catalog 
                     if plot_start < e.origins[0].time < plot_end]
        if len(chunk_cat) > 0:
            frame_plot = map_plot(
                catalog=chunk_cat, region=region,
                sizes=[max_size] * len(chunk_cat), fig=frame_plot,
                normaliser=5, new_depthcpt=_newdepth, max_depth=30, min_depth=0)
        _newdepth = False
        # Timestamp
        if frame_plot is not None:
            with open("text.tmp", "w") as f:
                f.write("{0} {1} {2}".format(
                    region[0] + 0.5, region[3] - 0.15, plot_start))
            with Session() as lib:
                lib.call_module(
                    "pstext", "text.tmp -F+f{0}p,black,- -Ya0i -Xa0i".format(10))
            os.remove("text.tmp")
        # Save
        f_out = "{0}/frame_{1}.jpg".format(working_dir, frame)
        if frame_plot is not None:
            frame_plot.savefig(f_out)
            print("Saved to {0}".format(f_out))
        working_files.append(f_out)
        # Step
        plot_start = plot_end
        #Add chunk to queue
        previous_chunks.append(chunk_cat)
    # Make gif
    images = [imageio.imread(working_file) for working_file in working_files]
    imageio.mimsave("{0}/{1}".format(plot_dir, outfile), images, 
                    duration=frame_length)
    shutil.rmtree(working_dir)
    return


def scale_by_repeats(catalog):
    from collections import Counter
    from obspy.core.event import Event, Catalog, Origin

    locations = [
        (ev.origins[0].latitude, ev.origins[0].longitude, ev.origins[0].depth)
        for ev in catalog]

    counted = Counter(locations)
    plot_catalog = Catalog([
        Event(
            origins=[Origin(latitude=loc[0], longitude=loc[1], depth=loc[2])]) 
        for loc in counted.keys()])
    sizes = np.asarray(list(counted.values()))
    return map_plot(plot_catalog, sizes=sizes ** .33, 
                    region=[172.5, 174.6, -43.1, -41.1])


def map_plot(catalog, tick_spacing=0.5, region_pad=0.1, min_depth=None,
             max_depth=None, normaliser=None, min_lat=None, max_lat=None,
             min_lon=None, max_lon=None, fig=None, sizes=None, region=None,
             color=None, new_depthcpt=True):
    import gmt
    from gmt.clib import Session
    from obspy.core.event import Magnitude

    GMT_DATA = "/Volumes/GeoPhysics_09/users-data/chambeca/.gmt_data"

    _longitudes, latitudes, depths, magnitudes = ([], [], [], [])

    for event in catalog:
        try:
            origin = event.preferred_origin() or event.origins[0]
        except IndexError:
            print("Cannot plot event, no origin")
            continue
        try:
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
        except IndexError:
            magnitude = Magnitude(1)
        _longitudes.append(origin.longitude)
        latitudes.append(origin.latitude)
        depths.append(origin.depth)
        magnitudes.append(magnitude.mag)

    if sizes is None:
        sizes = np.asarray(magnitudes) ** 2

    longitudes = []
    for _l in _longitudes:
        if _l < 0:
            _l += 360
        longitudes.append(_l)
    longitudes, latitudes, depths, sizes = (
        np.asarray(longitudes), np.asarray(latitudes), np.asarray(depths),
        np.asarray(sizes))
    depths /= 1000.0

    if min_lon is None:
        min_lon = longitudes.min()
    if max_lon is None:
        max_lon = longitudes.max()
    lon_slice = np.where(np.logical_and(
        longitudes >= min_lon, longitudes <= max_lon))
    longitudes, latitudes, depths, sizes = (
        longitudes[lon_slice], latitudes[lon_slice], depths[lon_slice],
        sizes[lon_slice])

    if min_lat is None:
        min_lat = latitudes.min()
    if max_lat is None:
        max_lat = latitudes.max()
    lat_slice = np.where(np.logical_and(
        latitudes >= min_lat, latitudes <= max_lat))
    longitudes, latitudes, depths, sizes = (
        longitudes[lat_slice], latitudes[lat_slice], depths[lat_slice],
        sizes[lat_slice])

    if min_depth is None:
        min_depth = depths.min()
    if max_depth is None:
        max_depth = depths.max()
    depth_slice = np.where(np.logical_and(
        depths >= min_depth, depths <= max_depth))
    longitudes, latitudes, depths, sizes = (
        longitudes[depth_slice], latitudes[depth_slice], depths[depth_slice],
        sizes[depth_slice])

    with Session() as lib:
        lib.call_module('gmtset', 'FORMAT_GEO_MAP ddd.xx')
        lib.call_module('gmtset', 'MAP_FRAME_TYPE plain')
        lib.call_module('gmtset', 'FONT_ANNOT_PRIMARY 12p')
        lib.call_module('gmtset', 'FONT_LABEL 12p')
        if color is None and new_depthcpt:
            lib.call_module(
                "makecpt", "-Crainbow -Z -T{0}/{1}/{2} > depth.cpt".format(
                    min_depth, max_depth, (max_depth - min_depth) / 10))

    if fig is None:
        fig = gmt.Figure()
        if region is None:
            region=[min(longitudes) - region_pad, max(longitudes) + region_pad,
                    min(latitudes) - region_pad, max(latitudes) + region_pad]
        fig.basemap(
            region=region,
            projection="M5i", frame="a4")
        fig.plot(data=GMT_DATA + "/faults_NZ_WGS84.gmt", W=0.85)
        # Plot lakes (global lakes suck for NZ)
        fig.plot(data=GMT_DATA + "/nz-lakes.gmt", color='white', pen='black')
        fig.coast(frame="a{0}nWSe".format(tick_spacing), resolution="f",
                  shorelines="1/black")
        if color is None:
            with Session() as lib:
                lib.call_module(
                    "psscale", '-Cdepth.cpt -D{0}i/{1}i/{2}i/{3}h '
                    '-Bpx{4}+l"Depth (km)"'.format(
                        3.125, 0.7, 2.5, 0.2, round((max_depth - min_depth) / 4)))

    if normaliser is None:
        normaliser = sizes.max()
    if color is None:
        try:
            fig.plot(x=longitudes, y=latitudes, style='cc', pen="black",
                    sizes=sizes / normaliser, C="depth.cpt",
                    color=depths)
        except:
            print("Couldn't plot the quakes man! There were {0} "
                  "of them".format(len(longitudes)))
    else:
        fig.plot(x=longitudes, y=latitudes, style='cc', pen="black",
                sizes=sizes / normaliser, color=color)

    return fig


def get_geonet_aftershocks(startdate=UTCDateTime(2013, 1, 1),
                           enddate=UTCDateTime(2019, 2, 1), min_lat=-43,
                           max_lat=-41, min_lon=172, max_lon=175.5,
                           min_depth=0, max_depth=50, full_cat=False):
    from obspy.clients.fdsn import Client
    from obspy.core.event import (
        Catalog, Event, Origin, ResourceIdentifier, Magnitude)
    from cjc_utilities.get_geonet_basic_info import get_geonet_events
    from eqcorrscan.utils.catalog_utils import spatial_clip
    from matplotlib.path import Path

    if full_cat:
        client = Client("GEONET")
        cat = Catalog()

        ndays = (enddate - startdate) / (24 * 60 * 60)
        chunk_size = 50
        day = 0
        chunk_start = startdate
        while day < ndays:
            chunk_end = chunk_start + (chunk_size * 24 * 60 * 60)
            print("Downloading for {0}-{1}".format(chunk_start, chunk_end))
            cat += client.get_events(
                starttime=chunk_start, endtime=chunk_end, minlatitude=min_lat,
                maxlatitude=max_lat, minlongitude=min_lon, maxlongitude=max_lon)
            day += chunk_size
            chunk_start = chunk_end
        cat += client.get_events(
            starttime=chunk_start, endtime=enddate, minlatitude=min_lat,
            maxlatitude=max_lat, minlongitude=min_lon, maxlongitude=max_lon)
    else:
        events = get_geonet_events(
                startdate=startdate.datetime, enddate=enddate.datetime,
                bbox=(min_lon, min_lat, max_lon, max_lat))
        cat = Catalog()
        for ev in events:
            event = Event(
                origins=[Origin(latitude=ev['latitude'], 
                                longitude=ev['longitude'],
                                depth=1000 * ev['depth'],
                                time=UTCDateTime(ev['origin-time']))],
                magnitudes=[Magnitude(mag=ev['magnitude'])],
                id=ResourceIdentifier(ev['id']))
            cat += event

    # Filter to a useful area
    corners = Path([
        (-42.0, 172.7), (-41.25, 174.5), (-42.5, 174.5), (-43.0, 173.5),
        (-43.0, 172.7), (-42.0, 172.7)])
    cat = spatial_clip(cat, corners, mindepth=min_depth, maxdepth=max_depth)
    cat = Catalog(sorted(cat, key=lambda e: e.origins[0].time))
    return cat


def cumulative_geonet(detections, geonet_cat):
    import matplotlib.dates as mdates
    from eqcorrscan.utils.plotting import cumulative_detections

    detections = [d for d in detections
                  if d.detect_time <= geonet_cat[-1].origins[0].time]
    cum_det_plot = cumulative_detections(
        detections=detections, plot_grouped=True, show=False,
        return_figure=True)
    ax = cum_det_plot.gca()
    dates = [mdates.date2num(e.origins[0].time.datetime) 
             for e in geonet_cat]
    counts = np.arange(len(dates))
    ax.plot(dates, counts, label="GeoNet")
    ax.legend()
    ax.set_ylim((0, max(len(detections), len(dates))))
    return cum_det_plot


def cumulative_by_fault(fault_catalogs, unassociated):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots()

    max_count = 0
    for fault, catalog in fault_catalogs.items():
        catalog = sorted(catalog, key=lambda e: e.origins[0].time)
        dates = [mdates.date2num(ev.origins[0].time.datetime) 
                 for ev in catalog]
        counts = np.arange(len(dates))
        if len(dates) > max_count:
            max_count = len(dates)
        ax.plot(dates, counts, label=fault)
    
    unassociated = sorted(unassociated, key=lambda e: e.origins[0].time)
    dates = [mdates.date2num(ev.origins[0].time.datetime) 
             for ev in unassociated]
    if max_count < len(dates):
        max_count = len(dates)
    counts = np.arange(len(dates))
    ax.plot(dates, counts, '--', color="grey", label="Unassociated")
    ax.legend()

    hours = mdates.AutoDateLocator()
    mins = mdates.HourLocator(byhour=np.arange(0, 24, 3))
    hrFMT = mdates.DateFormatter('%Y/%m/%d')
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(hrFMT)
    fig.autofmt_xdate()

    ax.set_ylim([0, max_count])
    return fig


def make_match_catalog(catalog_set="full_catalog"):
    import glob
    from eqcorrscan.core.match_filter import Party, read_party
    from obspy.core.event import Catalog, Event, Origin

    det_files = glob.glob("{0}/{1}/*_party.tgz".format(
        detect_dir, catalog_set))
    det_files.sort()
    party = Party()
    for det_file in det_files:
        print("Reading from {0}".format(det_file))
        party += read_party(det_file)
    
    match_cat = Catalog()
    for fam in party:
        master_origin = (
            fam.template.event.preferred_origin() or 
            fam.template.event.origins[0])
        for det in fam:
            det_ev = det.event.copy()
            det_ev.origins = [
                Origin(latitude=master_origin.latitude, 
                       longitude=master_origin.longitude, 
                       depth=master_origin.depth, time=det.detect_time)]
            match_cat += det_ev
    return party, match_cat


def map_plot_by_fault(fault_catalogs, unassociated, 
                      region=[172.5, 174.6, -43.1, -41.1], **kwargs):
    eq_plot = map_plot(
        unassociated, color="grey", sizes=[1] * len(unassociated), 
        normaliser=5, region=region, **kwargs)
    colors = cycle(COLORS)
    for name, catalog in fault_catalogs.items():
        if len(catalog) == 0:
            continue
        color = next(colors)
        print("{0:20s} is\t {1}".format(name, color))
        eq_plot = map_plot(
            catalog, color=color, fig=eq_plot, sizes=[1] * len(catalog),
            normaliser=5, region=region, **kwargs)
    return eq_plot


def merge_repicked_to_detections(match_cat, repicked_cat):
    from obspy import Catalog

    merged_cat = Catalog()
    match_cat_keyed = {
        ev.resource_id.id.split('/')[-1]: ev for ev in match_cat}
    repicked_cat_keyed = {}
    for ev in repicked_cat:
        key = list(ev.resource_id.id.split('/')[-1])
        key[28] = 'T'
        key = ''.join(key[0:35]) + '.' + ''.join(key[35:])
        repicked_cat_keyed.update({key: ev})
    for key, ev in repicked_cat_keyed.items():
        main_event = match_cat_keyed[key].copy()
        main_event.picks = ev.picks
        merged_cat.append(main_event)
    return merged_cat


def get_detections_for_catalog(detections, catalog):
    detections_out = []
    catalog_keys = [ev.resource_id.id.split('/')[-1] for ev in catalog]
    for detection in detections:
        key = list(detection.id)
        key[28] = 'T'
        key = ''.join(key[0:35]) + '.' + ''.join(key[35:])
        if key in catalog_keys:
            detections_out.append(detection)
    return detections_out



if __name__ == "__main__":
    import json, glob, os
    from obspy import read_events, Catalog
    from eqcorrscan.utils.plotting import freq_mag
    from eqcorrscan.core.match_filter import read_detections, Party

    from catalog_to_faults import (
        hamling_fault_mesh, make_fault_catalogs, plot_events, read_dsaa_grd)
    from catalog_to_dict import dict_to_catalog, catalog_to_dict

    Logger.info("Downloading GeoNet")
    geonet_cat = get_geonet_aftershocks()
    Logger.info("GeoNet download complete")
    # Background GeoNet map
    # geonet_map = map_plot(catalog=geonet_cat)
    # geonet_map.savefig("{0}/geonet_catalog.eps".format(plot_dir))

    # # Magnitude-frequency plot
    # freq_mag(
    #     magnitudes=[ev.magnitudes[0].mag for ev in geonet_cat],
    #     completeness=4, max_mag=8, show=False, save=True,
    #     savefile="{0}/geonet_mf.eps".format(plot_dir))

    # # Old cumulative detections
    # old_dets = read_detections("focal_mechanism_detections_old.csv")
    # cum_det_plot = cumulative_geonet(old_dets, geonet_cat)
    # cum_det_plot.savefig("{0}/old_dets_8_MAD.eps".format(plot_dir))

    # Second go with 652 well-defined templates
    # match_cat = make_match_catalog()
    # match_cat = read_events(
    #     "{0}/633_catalog/all_detections.xml".format(detect_dir))
    # new_dets = read_detections(
    #     "{0}/633_catalog/all_detections.csv".format(detect_dir))
    # cum_det_plot = cumulative_geonet(new_dets, geonet_cat)
    # cum_det_plot.savefig(
    #     "{0}/633_catalog_new_dets_0.15_threshold.eps".format(plot_dir))

    ## Third go with 2,600 well-defined templates
    ## Collecting/writing
    # match_party, match_cat = make_match_catalog()
    # match_cat.write(
    #      "../Detections/full_catalog/all_detections.xml", format="QUAKEML")
    # match_cat_dict = catalog_to_dict(match_cat)
    # with open("{0}/full_catalog/all_detections.json".format(detect_dir), 'w') as f:
    #      json.dump(match_cat_dict, f)
    # if os.path.isfile("../Detections/full_catalog/all_detections.tgz"):
    #     os.remove("../Detections/full_catalog/all_detections.tgz")
    # match_party.write("../Detections/full_catalog/all_detections.tgz")
    # # Get the re-picked detections
    # repicked_cat = Catalog()
    # repicked_files = glob.glob("{0}/full_catalog/*_repicked_catalog.xml".format(detect_dir))
    # repicked_files.sort()
    # for f in repicked_files:
    #     print("Reading from {0}".format(f))
    #     repicked_cat += read_events(f)
    # repicked_cat.write("../Detections/full_catalog/all_repicked.xml", format="QUAKEML")
    # at_least_five_stations = Catalog()
    # for event in repicked_cat:
    #     stations = {p.waveform_id.station_code for p in event.picks}
    #     if len(stations) > 4:
    #         at_least_five_stations.append(event)
    # at_least_five_stations_dict = catalog_to_dict(at_least_five_stations)
    # with open("{0}/full_catalog/at_least_five_stations_repicked.json".format(detect_dir), 'w') as f:
    #     json.dump(at_least_five_stations_dict, f)
    ## Reading old stuff
    # match_cat = read_events(
    #     "{0}/full_catalog/all_detections.xml".format(detect_dir))

    Logger.info("Reading in detection catalog")
    with open("{0}/full_catalog/all_detections.json".format(detect_dir), 'rb') as f:
        match_cat_dict = json.load(f)
    match_cat = dict_to_catalog(match_cat_dict)

    Logger.info("Reading in repicked catalog")
    with open("{0}/full_catalog/at_least_five_stations_repicked.json".format(detect_dir), 'rb') as f:
        at_least_five_stations_dict = json.load(f)
    at_least_five_stations = dict_to_catalog(at_least_five_stations_dict)

    # Logger.info("Merging the catalogs and dumping")
    # merged_cat = merge_repicked_to_detections(match_cat, at_least_five_stations)
    # with open("{0}/full_catalog/at_least_five_stations_repicked_with_origins.json".format(detect_dir), 'w') as f:
    #     json.dump(catalog_to_dict(merged_cat), f)
    Logger.info("Reading the merged catalog")
    with open("{0}/full_catalog/at_least_five_stations_repicked_with_origins.json".format(detect_dir), 'rb') as f:
        merged_cat_dict = json.load(f)
    merged_cat = dict_to_catalog(merged_cat_dict)

    Logger.info("Getting the detections")
    match_party = Party().read("../Detections/full_catalog/all_detections.tgz.tgz")
    new_dets = [d for f in match_party for d in f]
    if os.path.isfile("{0}/full_catalog/all_detections.csv".format(detect_dir)):
        os.remove("{0}/full_catalog/all_detections.csv".format(detect_dir))
    match_party.write(
        "{0}/full_catalog/all_detections.csv".format(detect_dir), format="csv")
    # new_dets = read_detections(
    #     "{0}/full_catalog/all_detections.csv".format(detect_dir))
    repicked_dets = get_detections_for_catalog(new_dets, merged_cat)
    if os.path.isfile("{0}/full_catalog/at_least_five_stations_repicked_detections.csv".format(detect_dir)):
        os.remove("{0}/full_catalog/at_least_five_stations_repicked"
                  "_detections.csv".format(detect_dir))
    for d in repicked_dets:
        d.write(
            "{0}/full_catalog/at_least_five_stations_repicked_detections.csv".format(detect_dir),
            append=True)
    # repicked_dets = read_detections(
    #     "{0}/full_catalog/at_least_five_stations_repicked"
    #     "_detections.csv".format(detect_dir))
    
    Logger.info("Plotting")
    cum_det_plot = cumulative_geonet(repicked_dets, geonet_cat)
    cum_det_plot.savefig(
        "{0}/new_dets_0.15_threshold.eps".format(plot_dir))

    # # Plot detections scaled by number of repeats
    repeat_map = scale_by_repeats(merged_cat)
    repeat_map.savefig("{0}/new_dets_scaled_by_number.eps".format(plot_dir))

    # # Plot earthquakes associated with faults
    # faults = hamling_fault_mesh(
    #     in_file="Hamling_etal_supplement/aam7194_Data_S3.dat")
    litchfield_faults = glob.glob("grid_files_Litchfield_faults/*.grd")
    fault_points = {}
    for f in litchfield_faults:
        name = f.split("to model 3D - ")[-1].split(".grd")[0]
        if len(name.split(' - ')) > 1:
            name = name.split(' - ')[1]
        fault_points.update({
            name: read_dsaa_grd(f, return_grid=True)})
    fault_catalogs, unassoc = make_fault_catalogs(
            earthquakes=merged_cat, fault_points=fault_points)
    for name, catalog in fault_catalogs.items():
        _name = name.replace("/", "_").replace(" ", "-")
        catalog.write("{0}/full_catalog/{1}_catalog.xml".format(
            detect_dir, _name), format="QUAKEML")
    unassoc.write("{0}/full_catalog/Unassociated_events.xml".format(
        detect_dir), format="QUAKEML")
    # fault_catalogs, unassoc = make_fault_catalogs(match_cat, faults)
    fault_map = map_plot_by_fault(fault_catalogs, unassoc)
    fault_map.savefig("{0}/new_dets_fault_associations.eps".format(plot_dir))

    # # Plot cumulative number per fault
    cum_faults = cumulative_by_fault(fault_catalogs, unassoc)
    cum_faults.savefig("{0}/new_dets_per_fault.eps".format(plot_dir))

    # Temporal analysis of five interface events
    # interface_events = [
    #     '2016p900146', '2017p002213', '2017p007903', '2017p198583', 
    #     '2017p217600']

    # Video of detections in time.
    # gmt_video(match_cat, step=0.5, frame_length=0.1, max_previous=40)
