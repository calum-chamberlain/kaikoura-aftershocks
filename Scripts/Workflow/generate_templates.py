"""
Script for making Kaikoura aftershock templates.

Written by: Calum J Chamberlain
Date:       22/06/2018
"""
import os
import logging
import numpy as np

from collections import Counter

from obspy import read_events, Inventory, UTCDateTime, Catalog, Stream
from obspy.core.event import WaveformStreamID
from obspy.clients.fdsn import Client
from eqcorrscan import Tribe
from eqcorrscan.core.template_gen import _group_events
from eqcorrscan.utils.catalog_utils import filter_picks


def project_dir():
    import platform

    if "XPS-13" in platform.node():
        return "/mnt/GeoPhysics_09/users-data/chambeca/Kaik_afterslip"
    elif "kea" in platform.node():
        return "/home/chambeca/Desktop/kaikoura-aftershocks"
    return "/Volumes/GeoPhysics_09/users-data/chambeca/Kaik_afterslip"

TEMPLATE_PATH = "{0}/Templates".format(project_dir())
CATALOG_FILE = "{0}/STREWN_merged_catalog.xml".format(TEMPLATE_PATH)
BAD_STATIONS = ['RDCS', 'RCBS', 'MRZ', 'POWZ', 'WPHZ', 'PRWZ', 'QRZ']

# 2021 update uses extra stations and removes response
OUTFILE = "{0}_4s_1.5Hz-12Hz_2021".format(os.path.splitext(CATALOG_FILE)[0])

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
    # GEONET doesn't provide what I would expect when requesting a time-span.
    # t1, t2 = cat[0].origins[0].time, cat[-1].origins[0].time
    t1, t2 = UTCDateTime(1990, 1, 1), UTCDateTime(2019, 1, 1)
    bulk = []
    for station in picked_stations:
        bulk.append(("NZ", station, "*", "*", t1, t2))
    client = Client("GEONET")
    inventory = client.get_stations_bulk(bulk, level="response")
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

    # Add in Cape Campbell STREWN sites
    strewn_sites = ["STWC", "CSCP", "STAW", "CSWB"]
    client = Client("IRIS")
    strewn_inv = client.get_stations(
        network="Z1", starttime=cat[0].origins[0].time, 
        endtime=cat[-1].origins[0].time, level="response")
    strewn_inv.networks[0].stations = [
        sta for sta in strewn_inv.networks[0] if sta.code in strewn_sites]
    inv += strewn_inv

    # CHECK - not action taken:
    #  Look for stations that have a change in sensor between start of scan and end
    starttime, endtime = UTCDateTime(2009, 1, 1), UTCDateTime(2020, 1, 1)
    inv = inv.select(starttime=starttime, endtime=endtime)
    for net in inv:
        for station in net:
            channel_codes = {chan.code[0:-1] for chan in station}
            if len(channel_codes) > 1:
                print(f"Multiple channels codes used for {station.code}")
            for code in {chan.code[0:-1] for chan in station}:
                channels = station.select(channel=f"{code}?")
                sensors = {chan.sensor.model for chan in channels}
                if len(sensors) > 1:
                    print(f"Multiple sensors for station {station.code}")

    # Associate station info with picks
    for event in cat:
        used_picks = []
        for pick in event.picks:
            station = inv.select(
                station=pick.waveform_id.station_code, 
                starttime=pick.time - 100, endtime=pick.time + 100)
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
    top_stations = {station[0] for station in counted[0:20]}
    # Nearby GeoNet stations that should be included and used for picking
    mandatory_stations = {'CRSZ', 'CMWZ', 'KEKS', 'PLWZ'}
    # Include STREWN sites around Cape Campbell
    mandatory_stations.update(set(strewn_sites))

    top_stations.update(mandatory_stations)

    cat = filter_picks(cat, stations=top_stations)
    used_events = []
    for event in cat:
        picked_stations = set(
            [pick.waveform_id.station_code for pick in event.picks])
        if len(picked_stations) >= 5:
            used_events.append(event)
    cat.events = sorted(used_events, key=lambda e: e.origins[0].time)
    Logger.info("After QC {0} events remain".format(len(cat)))

    # SOME Responses checked visually and confirmed to be the same in the pass-band
    tribe, tribe_corrected = construct(
        swin=swin, lowcut=lowcut, highcut=highcut, 
        samp_rate=samp_rate, length=length, prepick=prepick,
        filt_order=filt_order, catalog=cat,
        process_len=86400, all_horiz=all_horiz, plot=plot, min_snr=4, 
        save_progress=False, parallel=True, num_cores=10)
    print("Generated tribe of {0} templates".format(len(tribe)))
    # return tribe, tribe_corrected
    return tribe


def _remove_response(tr, inventory):
    print(f"Working on {tr.id}")
    tr_corrected = tr.copy().split().detrend().taper(0.05).merge(fill_value=0)
    tr_corrected = tr_corrected.remove_response(
        inventory, "VEL", pre_filt = [0.001, 0.005, 45, 50])
    return tr_corrected


def construct(swin, lowcut, highcut, samp_rate, length, prepick, filt_order,
              catalog, process_len, all_horiz, plot, min_snr, save_progress,
              parallel, num_cores, inventory):
    from functools import partial
    from multiprocessing import Pool
    # Break cat into sub-catalogs
    sub_catalogs = _group_events(
        catalog=catalog, process_len=process_len, template_length=length,
        data_pad=90)

    if "P_all" in swin or "S_all" in swin or all_horiz:
        all_channels = True
    else:
        all_channels = False

    tribe = Tribe()
    # tribe_corrected = Tribe()

    for sub_catalog in sub_catalogs:
        st = Stream()
        # Download data from GeoNet
        st += _download_from_client(
                client=Client("GEONET"), 
                catalog=sub_catalog, data_pad=90.,
                process_len=process_len, networks=["NZ"],
                all_channels=all_channels)
        # Download from IRIS
        st += _download_from_client(
                client=Client("IRIS"), 
                catalog=sub_catalog, data_pad=90.,
                process_len=process_len, networks=["Z1"],
                all_channels=all_channels)
        
        st = st.merge()
        bad_ids = []
        for tr in st:
            if np.ma.is_masked(tr.data):
                real_len = tr.data.shape[0] - tr.data.mask.sum()
                if real_len < .8 * (process_len * tr.stats.sampling_rate):
                    print(f"Insufficient data on {tr.id}")
                    bad_ids.append(tr.id)
        st.traces = [tr for tr in st if tr.id not in bad_ids]
        # Remove response - correct to velocity
        # st_corrected = Stream()
        # remove_response = partial(_remove_response, inventory=inventory)
        # if not parallel:
        #     for tr in st:
        #         st_corrected += remove_response(tr)
        # else:
        #     with Pool(num_cores) as pool:
        #         results = pool.map_async(remove_response, st)
        #         for res in results.get():
        #             st_corrected += res
        # Construct
        _tribe = Tribe().construct(
            method="from_meta_file", swin=swin, lowcut=lowcut, highcut=highcut, 
            samp_rate=samp_rate, length=length, prepick=prepick,
            filt_order=filt_order, catalog=sub_catalog, st=st,
            process_len=process_len, all_horiz=all_horiz, plot=plot, min_snr=min_snr, 
            save_progress=save_progress, parallel=parallel, num_cores=num_cores)
        # _tribe_corrected = Tribe().construct(
        #     method="from_meta_file", swin=swin, lowcut=lowcut, highcut=highcut, 
        #     samp_rate=samp_rate, length=length, prepick=prepick,
        #     filt_order=filt_order, catalog=cat, st=st_corrected,
        #     process_len=process_len, all_horiz=all_horiz, plot=plot, min_snr=min_snr, 
        #     save_progress=save_progress, parallel=parallel, num_cores=num_cores)

        tribe += _tribe
        # tribe_corrected += _tribe_corrected

    return tribe
    # return tribe, tribe_corrected




def _download_from_client(client, catalog, data_pad, process_len, networks,
                          all_channels=False):
    """
    Internal function to handle downloading from either seishub or fdsn client
    """
    from eqcorrscan.utils import pre_processing

    if isinstance(networks, str):
        networks = [networks, ]
    st = Stream()
    catalog = Catalog(sorted(catalog, key=lambda e: e.origins[0].time))
    all_waveform_info = []
    for event in catalog:
        for pick in event.picks:
            if not pick.waveform_id:
                Logger.warning(
                    "Pick not associated with waveforms, will not use:"
                    " {0}".format(pick))
                continue
            if all_channels:
                channel_code = pick.waveform_id.channel_code[0:2] + "?"
            else:
                channel_code = pick.waveform_id.channel_code
            if pick.waveform_id.station_code is None:
                Logger.error("No station code for pick, skipping")
                continue
            if pick.waveform_id.network_code and pick.waveform_id.network_code not in networks:
                continue
            all_waveform_info.append((
                pick.waveform_id.network_code or "*",
                pick.waveform_id.station_code,
                channel_code, pick.waveform_id.location_code or "*"))
    starttime = UTCDateTime(
        catalog[0].origins[0].time - data_pad)
    endtime = starttime + process_len
    # Check that endtime is after the last event
    if not endtime > catalog[-1].origins[0].time + data_pad:
        raise NotImplementedError(
            'Events do not fit in processing window')
    all_waveform_info = sorted(list(set(all_waveform_info)))
    dropped_pick_stations = 0
    for waveform_info in all_waveform_info:
        net, sta, chan, loc = waveform_info
        Logger.info('Downloading for start-time: {0} end-time: {1}'.format(
            starttime, endtime))
        Logger.debug('.'.join([net, sta, loc, chan]))
        query_params = dict(
            network=net, station=sta, location=loc, channel=chan,
            starttime=starttime, endtime=endtime)
        try:
            st += client.get_waveforms(**query_params)
        except Exception as e:
            Logger.error(e)
            Logger.error('Found no data for this station: {0}'.format(
                query_params))
            dropped_pick_stations += 1
    if not st and dropped_pick_stations == len(event.picks):
        raise Exception('No data available, is the server down?')
    st.merge()
    # clients download chunks, we need to check that the data are
    # the desired length
    final_channels = []
    for tr in st:
        tr.trim(starttime, endtime)
        if len(tr.data) == (process_len * tr.stats.sampling_rate) + 1:
            tr.data = tr.data[1:len(tr.data)]
        if tr.stats.endtime - tr.stats.starttime < 0.8 * process_len:
            Logger.warning(
                "Data for {0}.{1} is {2} hours long, which is less than 80 "
                "percent of the desired length, will not use".format(
                    tr.stats.station, tr.stats.channel,
                    (tr.stats.endtime - tr.stats.starttime) / 3600))
        elif not pre_processing._check_daylong(tr):
            Logger.warning(
                "Data are mostly zeros, removing trace: {0}".format(tr.id))
        else:
            final_channels.append(tr)
    st.traces = final_channels
    return st


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
