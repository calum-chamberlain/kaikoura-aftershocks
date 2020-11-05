"""
Script for QC-ing detections after lag-calc.

Calum Chamberlain: 16/07/2018
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from obspy import read_events, Catalog, Stream, read
from obspy.core.event import Event


def manual_check(
    catalog: Catalog, 
    waveform_dir: str, 
    checked_dict: dict = None,
    save_progress: bool = True,
    fig: plt.Figure = None,
    check: bool = True,
) -> dict:
    """ Perform manual checks of the detections. """
    import json
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    fig = None  # Reuse figure
    checked_dict = checked_dict or dict()
    catalog.events.sort(key=lambda e: e.picks[0].time)  # Sort by pick time
    total_events = len(catalog)
    for i, event in enumerate(catalog):
        if event.resource_id.id.split('/')[-1] in checked_dict.keys():
            continue
        status, fig = check_event(
            event=event, waveform_dir=waveform_dir, fig=fig, check=check,
            event_no=i, total_events=total_events)
        if check:
            checked_dict.update({event.resource_id.id.split('/')[-1]: status})
            if save_progress:
                with open("manual_check_progress.json", "w") as f:
                    json.dump(checked_dict, f)
    return checked_dict, fig


def check_event(
    event: Event, 
    waveform_dir: str, 
    fig: plt.Figure = None, 
    min_stations: int = 4,
    check: bool = True,
    event_no: int = 1,
    total_events: int = 1,
    min_p_picks: int = 0,
) -> str:
    """ Check a single event. """
    from cjc_utilities.plot_event.plot_event import plot_event

    try:
        event_time = (event.preferred_origin() or event.origins[0]).time
    except IndexError:
        event_time = sorted(event.picks, key=lambda p: p.time)[0].time
    status_mapper = {"G": "good", "B": "bad", "U": "Undecided"}
    fig = fig or plt.figure()
    if fig is not None:
        fig.clf()
    picked_stations = {pick.waveform_id.station_code for pick in event.picks}
    if len(picked_stations) < min_stations:
        return "bad", fig
    st = get_stream_for_event(event, waveform_dir)
    status = None
    if check:
        p_picks = [p for p in event.picks if p.phase_hint[0] == "P"]
        if len(p_picks) < min_p_picks:
            print("Fewer than {0} P picks for event {1}, "
                  "considered bad".format(min_p_picks, event_no))
            return "bad", fig
        st = st.detrend().filter(
            "bandpass", freqmin=1.5, freqmax=12., corners=4)
        fig = plot_event(event=event, st=st, fig=fig, size=(8.5, 8.5),
                         length=None)
        fig.canvas.draw()
        fig.show()
        status = None
        while not status:
            status_short = input(
                "Event {0} of {1} at {2}\tType your verdict: "
                "(G=good, B=bad, U=undecided)".format(
                    event_no, total_events, event_time))
            if status_short.upper() in status_mapper.keys():
                status = status_mapper[status_short.upper()]
            else:
                print("Unknown status {0}, try again".format(status_short))
                continue
    else:
        print("got data for event {0} of {1} at {2}".format(
            event_no, total_events, event_time))
    return status, fig


def plot_for_all_events(catalog: Catalog, waveform_dir: str, plot_dir: str):
    from cjc_utilities.plot_event.plot_event import plot_event

    fig = plt.figure()
    for i, event in enumerate(catalog):
        fig.clf()
        print("Working on event {0} of {1}".format(i, len(catalog)))
        st = get_stream_for_event(event, waveform_dir)
        st = st.detrend().filter(
            "bandpass", freqmin=1.5, freqmax=12., corners=4)
        fig = plot_event(event=event, st=st, fig=fig, size=(8.5, 8.5),
                         length=None)
        fig_name = "{0}/{1}_{2}".format(
            plot_dir, event.origins[0].time, 
            event.resource_id.id.split('/')[-1])
        for extension in ("png", "eps"):
            fig.savefig("{0}.{1}".format(fig_name, extension))
    return


def get_stream_for_event(event: Event, waveform_dir: str) -> Stream:
    """ Get the waveform for the event. """
    from obspy.clients.fdsn import Client

    event_time = sorted(event.picks, key=lambda p: p.time)[0].time
    event_id = event.resource_id.id.split('/')[-1]
    expected_waveform_dir = "{head}/{year}/{month:02d}".format(
        head=waveform_dir, year=event_time.year, month=event_time.month)
    wavefile_name = "{head}/{rid}.ms".format(
        head=expected_waveform_dir, rid=event_id)
    if not os.path.isdir(expected_waveform_dir):
        os.makedirs(expected_waveform_dir)
    elif os.path.isfile(wavefile_name):
        st = read(wavefile_name)
        return st
    # If we get to here then we need to download the data
    client = Client("GEONET")
    first_pick = sorted(event.picks, key=lambda p: p.time)[0]
    last_pick = sorted(event.picks, key=lambda p: p.time)[-1]
    bulk = [
        (pick.waveform_id.network_code or "*",
         pick.waveform_id.station_code or "*",
         pick.waveform_id.location_code or "*",
         pick.waveform_id.channel_code or "*",
         first_pick.time - 10, last_pick.time + 10)
        for pick in event.picks]
    for retry in range(5):
        try:
            st = client.get_waveforms_bulk(bulk)
            break
        except Exception as e:
            pass
    else:
        raise e
    st.write(wavefile_name, format="MSEED")
    return st


if __name__ == "__main__":
    import json
    import glob

    from catalog_to_dict import dict_to_catalog

    print("Running manual check")
    if os.path.isfile("manual_check_progress.json"):
        with open("manual_check_progress.json", "rb") as f:
            check_dict = json.load(f)
    else:
        check_dict = None
    
    fig = None
    
    json_file = ('../Detections/full_catalog/'
                 'at_least_five_stations_repicked_with_origins.json')

    with open(json_file, 'rb') as f:
        cat_dict = json.load(f)
    print("Loading json catalog, converting to obspy Catalog")
    cat = dict_to_catalog(cat_dict, inplace=True)

    print("There are {0} events in this file".format(len(cat)))
    check_dict, fig = manual_check(
        catalog=cat, waveform_dir="../Detections/waveforms",
        checked_dict=check_dict, save_progress=True, fig=fig)
    with open("manual_check_complete.json", "w") as f:
        json.dump(check_dict, f)
