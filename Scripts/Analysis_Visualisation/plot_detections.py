"""
Script to extract and plot detections. 
"""
import numpy as np
import matplotlib.pyplot as plt

from eqcorrscan import Party
from eqcorrscan.utils.plotting import detection_multiplot

from obspy.clients.fdsn import Client

from cjc_utilities.get_geonet_basic_info import get_geonet_events


TOP_DIR = "/Volumes/GeoPhysics_09/users-data/chambeca/Kaik_afterslip"
PARTY_FILE = "{0}/Detections/2016_11_01-2018_11_02_party.tgz".format(TOP_DIR)
FAMILY_INDEX = 0
DETECTION_INDEX = 0


def plot_detection_rate(party):
    """
    Compare GeoNet detections to matched-filter detections.
    """
    all_detection_times = [d.detect_time for f in party for d in f]
    all_detection_times.sort()
    t1 = min(all_detection_times)
    t2 = max(all_detection_times) + 86400
    event_info = get_geonet_events(
        startdate=t1.date, enddate=t2.date,
        bbox=(172.06, -43.014, 175.261, -40.086))
    geonet_times = sorted([e['origin-time'] for e in event_info])
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sorted(geonet_times), np.arange(len(geonet_times)), 'r',
            label="geonet")
    all_detection_times = [d.datetime for d in all_detection_times]
    ax.plot(sorted(all_detection_times), np.arange(len(all_detection_times)),
            'k', label="matched-filter")
    ax.legend()
    fig.show()


def plot_catalog_rate(catalog):
    """
    Comparse GeoNet detections to matched-filter, repicked detections.
    """
    all_detection_times = []
    for event in catalog:
        try:
            origin = event.preferred_origin() or event.origins[0]
            detect_time = origin.time
        except IndexError:
            detect_time = min([p.time for p in event.picks])
        all_detection_times.append(detect_time)
    all_detection_times.sort()
    t1 = min(all_detection_times)
    t2 = max(all_detection_times) + 86400
    event_info = get_geonet_events(
        startdate=t1.date, enddate=t2.date,
        bbox=(172.06, -43.014, 175.261, -40.086))
    geonet_times = sorted([e['origin-time'] for e in event_info])
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sorted(geonet_times), np.arange(len(geonet_times)), 'r',
            label="geonet")
    all_detection_times = [d.datetime for d in all_detection_times]
    ax.plot(sorted(all_detection_times), np.arange(len(all_detection_times)),
            'k', label="matched-filter")
    ax.legend()
    ax.autoscale(tight=True)
    fig.show()


def plot_detection(template, detection, client=None, st=None, process=True):
    bulk = []
    if st is None:
        if client is None:
            raise AttributeError("Need either a client or a stream.")
        for tr in template.st:
            bulk.append((
                tr.stats.network, tr.stats.station, tr.stats.location, 
                tr.stats.channel, detection.detect_time - 60, 
                detection.detect_time + 120))
        st = client.get_waveforms_bulk(bulk)
    if process:
        st = st.detrend()
        if template.lowcut and template.highcut:
            st.filter(
                'bandpass', freqmin=template.lowcut, freqmax=template.highcut,
                corners=template.filt_order)
        elif template.lowcut:
            st.filter(
                'highpass', freq=template.lowcut, corners=template.filt_order)
        elif template.highcut:
            st.filter(
                'lowpass', freq=template.lowcut, corners=template.filt_order)
        st.resample(template.st[0].stats.sampling_rate)
    detection_multiplot(
        stream=st, template=template.st, times=[detection.detect_time])


if __name__ == "__main__":
    party = Party().read(PARTY_FILE)
    client = Client("GEONET")
    plot_detection(
        template=party[FAMILY_INDEX].template,
        detection=party[FAMILY_INDEX][DETECTION_INDEX], client=client)