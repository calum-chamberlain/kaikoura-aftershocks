"""
Make magnitude picks for detected and located events.
"""
import os
from math import log10

from obspy import read_events, Stream, Inventory, UTCDateTime, Catalog, read
from obspy.core.event import (
    Event, Magnitude, StationMagnitude, ResourceIdentifier, 
    StationMagnitudeContribution, Amplitude, WaveformStreamID)
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

from obsplus.events.pd import event_to_dataframe

from eqcorrscan import Tribe
from eqcorrscan.utils.mag_calc import amp_pick_event


CLIENT = Client("GEONET")
EVENT_FILE = "../Locations/NLL_located.xml"


def read_station_corrections(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    assert lines[0] == "Code 	Correction 	Error 	Readings 	Latitude (°) 	Longitude (°)"
    lines = [l.split() for l in lines]
    return {l[0]: float(l[1].replace("−", "-")) for l in lines[1:]}


STATION_CORRECTIONS = read_station_corrections("../Ristau2016_station_corrections.tsv")


def make_bulk_for_event(
    event: Event, 
    starttime: UTCDateTime, 
    endtime: UTCDateTime
) -> list:
    seed_ids = {
        (p.waveform_id.network_code or "*", p.waveform_id.station_code or "*",
         p.waveform_id.location_code or "*", p.waveform_id.channel_code or "*") 
         for p in event.picks}
    bulk = []
    for s in seed_ids:
        s = list(s)
        for i in range(len(s)):
            if len(s[i]) == 0:
                s[i] = "*"
        if s[-1] == "*":
            bulk.append((s[0], s[1], s[2], "HH?", starttime, endtime))
            bulk.append((s[0], s[1], s[2], "EH?", starttime, endtime))
        else:
            bulk.append((s[0], s[1], s[2], s[3], starttime, endtime))
    return bulk


def make_bulk_for_catalog(catalog: Catalog) -> list:
    starttime = min((ev.preferred_origin() or ev.origins[-1]).time 
                    for ev in catalog)
    endtime = max((ev.preferred_origin() or ev.origins[-1]).time 
                  for ev in catalog)
    starttime -= 3600
    endtime += 3600
    seed_ids = set(
        (p.waveform_id.network_code or "*", p.waveform_id.station_code or "*",
         p.waveform_id.location_code or "*", p.waveform_id.channel_code or "*") 
         for event in catalog for p in event.picks)
    bulk = []
    for s in seed_ids:
        s = list(s)
        for i in range(len(s)):
            if len(s[i]) == 0:
                s[i] = "*"
        if s[-1] == "*":
            bulk.append((s[0], s[1], s[2], "HH?", starttime, endtime))
            bulk.append((s[0], s[1], s[2], "EH?", starttime, endtime))
        else:
            bulk.append((s[0], s[1], s[2], s[3], starttime, endtime))
    return bulk


def get_waveforms_for_event(
    event: Event, 
    length: float = 140., 
    pre_origin: float = 20.
) -> Stream:
    starttime = (event.preferred_origin() or event.origins[0]).time
    starttime -= pre_origin
    endtime = starttime + length
    bulk = make_bulk_for_event(event, starttime, endtime)
    try:
        st = CLIENT.get_waveforms_bulk(bulk)
    except Exception as e:
        print(e)
        st = Stream()
        for b in bulk:
            try:
                st += CLIENT.get_waveforms(*b)
            except Exception as e:
                pass
    return st


def get_inv_for_event(event: Event) -> Inventory:
    starttime = (event.preferred_origin() or event.origins[0]).time
    endtime = starttime + 600
    bulk = make_bulk_for_event(event, starttime, endtime)
    return CLIENT.get_stations_bulk(bulk, level="response")


def geonet_magnitude(event: Event, inv: Inventory) -> Event:
    """
    Calculate magnitude based on relation in Ristau et al., 2016, BSSA.

    Station corrections from http://www.seismosoc.org/Publications/BSSA_html/bssa_106-2/2015293-esupp/2015293_esupp_Table_S1.html
    """
    origin_lat, origin_lon, origin_depth = (
        (event.preferred_origin() or event.origins[0]).latitude,
        (event.preferred_origin() or event.origins[0]).longitude,
        (event.preferred_origin() or event.origins[0]).depth / 1000.)
    station_magnitudes, full_station_magnitudes = [], []
    stations_used = set()
    for amplitude in event.amplitudes:
        correction = STATION_CORRECTIONS.get(amplitude.waveform_id.station_code, None)
        if correction is None:
            print(f"No station correction for {amplitude.waveform_id.station_code}, using 0.")
            correction = 0
        amp = amplitude.generic_amplitude
        # TODO: Unit check - should be in mm
        if amplitude.unit == "m":
            amp *= 1000
        elif amplitude.unit != "mm":
            raise NotImplementedError(f"Amplitude measured in {amplitude.unit}, which I ain't designed fur!")
        station = inv.select(
            network=amplitude.waveform_id.network_code,
            station=amplitude.waveform_id.station_code,
            location=amplitude.waveform_id.location_code,
            channel=amplitude.waveform_id.channel_code)
        station_lat, station_lon = station[0][0].latitude, station[0][0].longitude
        distance, _, _ = gps2dist_azimuth(station_lat, station_lon, origin_lat, origin_lon)
        distance /= 1000.
        distance = (distance ** 2 + origin_depth ** 2) ** .5

        # Calculate magnitude according to equations 9 and 10 of Ristau et al 2016
        logA0 = 0.29 - (1.27 * 10 ** -3) * distance - 1.49 * log10(distance)
        mag = log10(amp) - (logA0 + correction)
        station_magnitudes.append(mag)
        stations_used.update({amplitude.waveform_id.station_code})
        full_station_magnitudes.append(StationMagnitude(
            origin_id=(event.preferred_origin() or event.origins[0]).resource_id,
            mag=mag, amplitude_id=amplitude.resource_id, 
            station_magnitude_type="ML", 
            method_id=ResourceIdentifier("smi:local/pick_magnitudes.py")))
    # Put everything into the event
    event.station_magnitudes.extend(full_station_magnitudes)
    mag = sum(station_magnitudes) / len(station_magnitudes)
    magnitude = Magnitude(
        mag=mag,
        magnitude_type="ML", station_count=len(stations_used),
        method_id=ResourceIdentifier("smi:local/pick_magnitudes.py"),
        station_magnitude_contributions=[
            StationMagnitudeContribution(
                sta_mag.resource_id, residual=sta_mag.mag - mag, weight=1) 
            for sta_mag in full_station_magnitudes])
    event.magnitudes.append(magnitude)
    event.preferred_magnitude_id = magnitude.resource_id
    return event


def relative_magnitude(
    event: Event, 
    st: Stream, 
    tribe: Tribe
) -> Event:
    from eqcorrscan.utils.mag_calc import relative_magnitude

    template_id = "_".join(event.resource_id.id.split('/')[-1].split("_")[0:-1])
    template_event = tribe.select(template_name=template_id).event
    tid = template_event.resource_id.id.split('/')[-1]
    template_path = f"../Templates/waveforms/{tid}/{tid}.ms"
    if not os.path.isdir(f"../Templates/waveforms/{tid}"):
        os.makedirs(f"../Templates/waveforms/{tid}")
    if os.path.isfile(template_path):
        template_st = read(template_path)
    else:
        template_st = get_waveforms_for_event(template_event)
        template_st.write(template_path, format="MSEED")
    
    freqmin, freqmax = 2, 20
    st_filt = st.copy().detrend().filter("bandpass", freqmin=freqmin, 
                                         freqmax=freqmax)
    template_st.detrend().filter("bandpass", freqmin=freqmin, freqmax=freqmax)
    relative_magnitudes = relative_magnitude(
        st2=st_filt, st1=template_st, event2=event, event1=template_event,
        min_snr=1.0, min_cc=0.0, weight_by_correlation=False)
    
    station_magnitudes = []
    template_mag = template_event.preferred_magnitude() or template_event.magnitudes[-1]
    for seed_id, addition in relative_magnitudes.items():
        net, sta, loc, chan = seed_id.split('.')
        sta_mag = StationMagnitude(
            origin_id=(event.preferred_origin() or event.origins[0]).resource_id,
            mag=template_mag.mag + addition, 
            waveform_id=WaveformStreamID(
                network_code=net, station_code=sta, location_code=loc, 
                channel_code=chan),
            station_magnitude_type=template_mag.magnitude_type,
            method_id=ResourceIdentifier(
                "smi:local/eqcorrscan.utils.mag_calc.relative_magnitude"))
        event.station_magnitudes.append(sta_mag)
        station_magnitudes.append(sta_mag)
    if len(station_magnitudes) == 0:
        return event
    mag = sum(sta_mag.mag for sta_mag in station_magnitudes) / len(station_magnitudes)
    event.magnitudes.append(Magnitude(
        mag=mag, magnitude_type=template_mag.magnitude_type, 
        origin_id=(event.preferred_origin() or event.origins[0]).resource_id,
        method_id=ResourceIdentifier(
            "smi:local/eqcorrscan.utils.mag_calc.relative_magnitude"),
        station_count=len(station_magnitudes),
        station_magnitude_contributions=[
            StationMagnitudeContribution(
                sta_mag.resource_id, residual=sta_mag.mag - mag, weight=1) 
            for sta_mag in station_magnitudes]))
    event.preferred_magnitude_id = event.magnitudes[-1].resource_id
    return event


def event_magnitude(
    event: Event, 
    tribe: Tribe, 
    inventory: Inventory = None, 
    skip_done: bool = True
) -> Event:   
    origin = event.preferred_origin() or event.origins[0]
    event_dir = f"../Detected_events/Y{origin.time.year}/R{origin.time.julday:03d}"
    event_fname = event.resource_id.id.split('/')[-1]
    event_dir = f"{event_dir}/{event_fname}"
    if not os.path.isdir(event_dir):
        os.makedirs(event_dir)
    if os.path.isfile(f"{event_dir}/{event_fname}.xml") and skip_done:
        return read_events(f"{event_dir}/{event_fname}.xml")[0]
    if os.path.isfile(f"{event_dir}/{event_fname}.ms"):
        print("Reading waveforms from disk")
        st = read(f"{event_dir}/{event_fname}.ms")
        pick_min = min(p.time for p in event.picks)
        early_enough = True
        for tr in st:
            if tr.stats.starttime > pick_min - 20:
                early_enough = False
                break
        if not early_enough:
            st = get_waveforms_for_event(event, length=220, pre_origin=100)
            st.write(f"{event_dir}/{event_fname}.ms", format="MSEED")
    else:
        print("Downloading waveforms from server")
        st = get_waveforms_for_event(event, length=120)
        st.write(f"{event_dir}/{event_fname}.ms", format="MSEED")
    if inventory is None:
        inventory = get_inv_for_event(event)
    event = amp_pick_event(
        event, st=st, inventory=inventory, chans=["1", "2", "N", "E"], 
        iaspei_standard=False)
    if len(event.amplitudes) == 0:
        return event
    event = geonet_magnitude(event, inv)
    # Relative magnitude
    event = relative_magnitude(event, st, tribe)
    event.write(f"{event_dir}/{event_fname}.xml", format="QUAKEML")
    return event


def link_arrivals(event):
    """ Link arrivals to picks if they are not linked. """
    from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

    ori = event.preferred_origin() or event.origins[-1]
    bulk = [
        (p.waveform_id.network_code, p.waveform_id.station_code, 
         p.waveform_id.location_code, p.waveform_id.channel_code, 
         ori.time - 3600, ori.time + 3600) for p in event.picks]
    inv = CLIENT.get_stations_bulk(bulk)

    sta_dist = dict() 
    for network in inv: 
        for station in network: 
            dist, _, baz = gps2dist_azimuth(
                lat1=station.latitude, lat2=ori.latitude, 
                lon1=station.longitude, lon2=ori.longitude) 
            dist = ((dist ** 2) + (ori.depth ** 2)) ** .5 
            dist /= 1000 
            sta_dist.update(
                {station.code: {"dist": kilometer2degrees(dist), "az": baz}}) 
    
    # Match arrival to pick based on distance, azimuth and phase.
    for arrival in ori.arrivals:
        possible_matches = []
        for key, value in sta_dist.items():
            relative_distance = abs(value["dist"] - arrival.distance)
            relative_azimuth = abs(value["az"] - arrival.azimuth)
            if relative_distance < 0.1 and relative_azimuth < 5:
                possible_matches.append(key)
        for possible_station in possible_matches:
            picks = [p for p in event.picks 
                     if p.waveform_id.station_code == possible_station and 
                     p.phase_hint == arrival.phase]
            if len(picks) == 0:
                print("No matches, ugh")
                continue
            assert len({p.time.datetime for p in picks}) == 1  
            # Check that the picks have the same time, so we can take whatever
            print(f"Matched to {picks[0]}")
            arrival.pick_id = picks[0].resource_id
            break
    return event



# Done once on the 19th of June 2020.
# def update_geonet_magnitudes(tribe: Tribe) -> Tribe:
#     """ Update the magnitudes in the tribe to GeoNet's preferred magnitudes. """
#     for template in tribe:
#         public_id = template.event.resource_id.id.split('/')[-1]
#         print(f"Updating magnitude to {public_id}")
#         try:
#             ev = CLIENT.get_events(eventid=public_id)[0]
#         except:
#             print(f"No event found for {public_id}")
#             continue
#         geonet_preferred = ev.preferred_magnitude() or ev.magnitudes[-1]
#         add_mag = Magnitude(
#             mag=geonet_preferred.mag, 
#             magnitude_type=geonet_preferred.magnitude_type,
#             creation_info=geonet_preferred.creation_info)
#         template.event.magnitudes.append(add_mag)
#         template.event.preferred_magnitude_id = add_mag.resource_id
#     return tribe


if __name__ == "__main__":
    from progressbar import ProgressBar

    cat = read_events(EVENT_FILE)
    # template_cat = read_events("../Templates/STREWN_merged_catalog.xml")
    tribe = Tribe().read("../Templates/STREWN_merged_catalog_4s_1.5Hz-12Hz_2019_geonet_mags.tgz")
    bulk = make_bulk_for_catalog(cat)
    inv = CLIENT.get_stations_bulk(bulk, level="response")
    cat_out = Catalog()
    bar = ProgressBar(max_value=len(cat))
    i = 0
    for ev in cat[i:]:
        ev_mag = event_magnitude(ev, tribe, inventory=inv, skip_done=False)
        cat_out.append(ev_mag)
        # print(ev_mag)
        bar.update(i)
        i += 1
    bar.finish()

    cat_out.write("../Locations/NLL_located_magnitudes.xml", format="QUAKEML")

    # Write out a summary csv
    df = event_to_dataframe(cat_out[0])
    not_added = []  # Cope with events have arrivals not linked to picks
    for ev in cat_out[1:]:
        try:
            df = df.append(event_to_dataframe(ev))
        except Exception as e:
            print(e)
            not_added.append(ev)
    
    # Fix the arrival linkage
    to_add = []
    for ev in not_added:
        to_add.append(link_arrivals(ev))
    
    for ev in to_add:
        df = df.append(event_to_dataframe(ev))
    
    df.to_csv("NLL_located_magnitudes.csv")
