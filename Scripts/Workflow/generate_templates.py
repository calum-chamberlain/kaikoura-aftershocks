"""
Script for making Kaikoura aftershock templates.

Written by: Calum J Chamberlain
Date:       22/06/2018
"""
import os
import logging

from collections import Counter

from obspy import read_events, Inventory
from obspy.core.event import WaveformStreamID
from obspy.clients.fdsn import Client
from eqcorrscan import Tribe
from eqcorrscan.utils.catalog_utils import filter_picks


def project_dir():
    import platform

    if "XPS-13" in platform.node():
        return "/mnt/GeoPhysics_09/users-data/chambeca/Kaik_afterslip"
    elif "kea" in platform.node():
        return "/home/chambeca/Desktop/kaikoura-afterslp"
    return "/Volumes/GeoPhysics_09/users-data/chambeca/Kaik_afterslip"

TEMPLATE_PATH = "{0}/Templates".format(project_dir())
CATALOG_FILE = "{0}/STREWN_merged_catalog.xml".format(TEMPLATE_PATH)
BAD_STATIONS = ['RDCS', 'RCBS', 'MRZ', 'POWZ', 'WPHZ', 'PRWZ', 'QRZ']
OUTFILE = "{0}_4s_1.5Hz-12Hz_2019".format(os.path.splitext(CATALOG_FILE)[0])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger("generate_templates.__main__")

def generate_templates(cat, swin=["P", "S"], lowcut=1.5, highcut=12,
                       samp_rate=30, length=4, prepick=0.15, filt_order=4,
                       all_horiz=True, plot=False):
    cat.events.sort(key=lambda e: e.origins[0].time)
    Logger.info("Working on a catalog of {0} events".format(len(cat)))

    # Note that picks are not associated with channels, which EQcorrscan needs
    picked_stations = set([pick.waveform_id.station_code for event in cat 
                        for pick in event.picks])
    t1 = cat[0].origins[0].time
    t2 = cat[-1].origins[0].time
    bulk = []
    for station in picked_stations:
        bulk.append(("NZ", station, "*", "*", t1, t2))
    client = Client("GEONET")
    inventory = client.get_stations_bulk(bulk, level="channel")
    # We only care about continuous channels at 50 or 100 Hz
    inv = Inventory(
        networks=[], source=inventory.source, created=inventory.created,
        module=inventory.module, module_uri=inventory.module_uri,
        sender=inventory.sender)
    for network in inventory:
        net = network.copy()
        net.stations = []
        net._selected_number_of_stations = 0
        for station in network:
            sta = station.copy()
            sta.channels = []
            sta._selected_number_of_channels = 0
            for channel in station:
                if channel.code[1] == "N":
                    # Marks accelerometer - strong-motion, we don't want to
                    # use these because they are unreliable in GeoNet.
                    continue
                sampling_rate = (
                    float(channel.sample_rate_ratio_number_samples) / 
                    float(channel.sample_rate_ratio_number_seconds))
                if sampling_rate >= 50.0 and 'CONTINUOUS' in channel.types:
                    sta.channels.append(channel.copy())
                    sta._selected_number_of_channels += 1
            if sta._selected_number_of_channels >= 3:
                net.stations.append(sta)
                net._selected_number_of_stations += 1
        if net._selected_number_of_stations > 0:
            inv.networks.append(net)

    # Associate station info with picks
    for event in cat:
        used_picks = []
        for pick in event.picks:
            station = inv.select(station=pick.waveform_id.station_code)
            if pick.waveform_id.station_code in BAD_STATIONS:
                continue
            if len(station) == 0:
                continue
            channels = station[0][0].channels
            verticals = [chan for chan in channels if chan.code[-1] == "Z"]
            horizontals = [chan for chan in channels 
                        if chan.code[-1] in ['1', '2', 'N', 'E']]
            if pick.phase_hint == 'P':
                pick.waveform_id = WaveformStreamID(
                    station_code=pick.waveform_id.station_code,
                    channel_code=verticals[0].code, 
                    location_code=verticals[0].location_code,
                    network_code=station[0].code)
                used_picks.append(pick)
            elif pick.phase_hint == 'S':
                pick.waveform_id = WaveformStreamID(
                    station_code=pick.waveform_id.station_code,
                    channel_code=horizontals[0].code, 
                    location_code=horizontals[0].location_code,
                    network_code=station[0].code)
                used_picks.append(pick)
        Logger.info("After association this event has {0} picks".format(
            len(used_picks)))
        event.picks = used_picks
    picked_stations = [pick.waveform_id.station_code 
                       for event in cat for pick in event.picks]
    counted = Counter(picked_stations).most_common()
    # Use only the top 20 most-picked stations!
    top_stations = [station[0] for station in counted[0:20]]
    cat = filter_picks(cat, stations=top_stations)
    used_events = []
    for event in cat:
        picked_stations = set(
            [pick.waveform_id.station_code for pick in event.picks])
        if len(picked_stations) >= 5:
            used_events.append(event)
    cat.events = sorted(used_events, key=lambda e: e.origins[0].time)
    Logger.info("After QC {0} events remain".format(len(cat)))
    tribe = Tribe().construct(
        method="from_client", swin=swin, lowcut=lowcut, highcut=highcut, 
        samp_rate=samp_rate, length=length, prepick=prepick,
        filt_order=filt_order, catalog=cat, client_id="GEONET",
        process_len=86400, all_horiz=all_horiz, plot=plot, min_snr=4, 
        save_progress=False, parallel=True, num_cores=4)
    print("Generated tribe of {0} templates".format(len(tribe)))
    return tribe


if __name__ == "__main__":
    Logger.info("Reading catalog from {0}".format(CATALOG_FILE))
    cat = read_events(CATALOG_FILE)
    Logger.info("Read in {0} events".format(len(cat)))
    # cat = cat[0:2] # Used a short catalog for testing
    tribe = generate_templates(cat=cat)
    Logger.info("Generated tribe of {0} templates".format(len(tribe)))
    tribe.write(OUTFILE)
    Logger.info("Written tribe to {0}.tgz".format(OUTFILE))
    Logger.info("FINISHED")
