#!/usr/bin/env python
"""
Pick template polarities for focal mechanism analysis

"""

import json

from typing import Tuple
import progressbar

from obspy import read_events, Trace, Stream
from obspy.core.event import Pick, Event, Catalog
from obspy.clients.fdsn import Client

from matplotlib.lines import Line2D
from matplotlib.pyplot import Figure


KEY_MAPPING = dict(
    u="positive",
    up="positive",
    c="positive",
    d="negative",
    down="negative",
    right="undecidable"
)

STREWN_STATIONS = [
    "CSCP", "CSWB", "LTW1A", "LTW1", "LTW2", "LTW3", "S002", "S007", "S012",
    "S018", "S021", "S030", "S034", "S039", "SEAS1", "STAW", "STRN1", "STRN2",
    "STRN3", "STRN4", "STRN5", "STRN6", "STWC", "S029"]

STREWN_MAP = {
    "STR1": "STRN1",
    "STR2": "STRN2",
    "STR3": "STRN3",
    "STR4": "STRN4",
    "STR5": "STRN5",
    "STR6": "STRN6",
    "SEAS": "SEAS1",
}

GEONET_CLIENT, IRIS_CLIENT = Client("GEONET"), Client("IRIS")


def pick_polarity(
    trace: Trace,
    pick: Pick,
    pre_pick: float = 2.0,
    post_pick: float = 3.0,
    fig: Figure = None,
):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timezone

    utc = timezone.utc

    fuckup, _quit = False, False


    def updown(event):
        nonlocal fuckup, _quit
        # print(f"You selected {event.key}")
        if event.key in KEY_MAPPING.keys():
            pick.polarity = KEY_MAPPING.get(event.key)
            # plt.close()
            fig.canvas.stop_event_loop()
            return
        elif event.key == "left":
            # Go back to the previous one...
            print("Fucked-up, going back")
            fuckup = True
            # plt.close()
            fig.canvas.stop_event_loop()
            return
        elif event.key == "q":
            print("Quitting")
            _quit = True
            fig.canvas.stop_event_loop()
            return

    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        ax = fig.gca()
        ax.clear()  # Remove the old plot

    cid = fig.canvas.mpl_connect('key_press_event', updown)

    trace = trace.slice(pick.time - pre_pick, pick.time + post_pick)
    ax.plot(trace.times(), trace.data)
    pick_time = pick.time - trace.stats.starttime
    ax.add_line(
        Line2D(xdata=[pick_time, pick_time],
               ydata=list(ax.get_ylim()), color='r'))
    ax.set_title(f"{trace.id}: pick up or down. Right to ignore")
    ax.set_xlabel(f"Seconds from {trace.stats.starttime}")

    fig.show()
    fig.canvas.draw()  # Redraw
    fig.canvas.start_event_loop()
    # Picking happens here
    fig.canvas.mpl_disconnect(cid)
    if not fuckup and not _quit:
        print(f"{trace.id} picked as {pick.polarity}")
    return pick, fuckup, _quit


def pick_event_polarities(
    st: Stream,
    event: Event,
):
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")  # Easier on my eyes

    # Keep all the non P-picks
    picked = [p for p in event.picks if not p.phase_hint.startswith("P")]

    # Get available stations
    stations = {tr.stats.station for tr in st}
    # List of picks to work with
    p_picks = [
        p for p in event.picks
        if p.phase_hint.startswith("P") and p.waveform_id.station_code in stations]
    polarity_picked = []
    i = 0  # Using a while loop to allow us to go back if we fuckup.
    fig, ax = plt.subplots(figsize=(12, 8))
    while i < len(p_picks):
        pick = p_picks[i]
        tr = st.select(station=pick.waveform_id.station_code)
        if len(tr) == 0:
            print(f"No data for {pick.waveform_id.station_code}")
            pick.polarity = "undecidable"
        for _tr in tr.merge():
            polarity_pick, mistake, _quit = pick_polarity(
                _tr, pick.copy(), fig=fig)
            if _quit:
                print("Quitting")
                return None
            if mistake is True:
                print("You fucked up - removing last pick")
                if i == 0:
                    print("Already at zeroth pick, try again")
                else:
                    i -= 1  # Go back.
                    polarity_picked = polarity_picked[0:-1]  # Remove the last pick.
                break
            polarity_pick.waveform_id.channel_code = _tr.stats.channel
            polarity_pick.waveform_id.network_code = _tr.stats.network
            polarity_pick.waveform_id.location_code = _tr.stats.location
            polarity_picked.append(polarity_pick)
        else:
            # If we don't break, continue
            i += 1
    event_picked = event.copy()
    event_picked.picks = picked + polarity_picked
    plt.close(fig)
    return event_picked


def get_stream(event: Event, pre_pick: float = 2.0, post_pick: float = 3.0):
    from concurrent.futures import ThreadPoolExecutor
    from obspy.clients.fdsn import Client
    from obspy import read
    from functools import partial
    import os

    # Try reading from disk first
    filename = f"Polarity_waveforms/{event.resource_id.id.split('/')[-1]}.ms"
    if os.path.isfile(filename):
        return read(filename)
    print("Downloading stream from client")

    geonet_bulk, iris_bulk = [], []
    for pick in event.picks:
        if not pick.phase_hint.startswith("P"):
            continue
        for chan_code in "EHB":
            _bulk = ["*", pick.waveform_id.station_code, "*", f"{chan_code}?Z",
                     pick.time - pre_pick, pick.time + post_pick]
            if pick.waveform_id.station_code not in STREWN_STATIONS:
                _bulk[0] = "NZ"
                geonet_bulk.append(tuple(_bulk))
            else:
                _bulk[0] = "Z1"
                iris_bulk.append(tuple(_bulk))

    executor = ThreadPoolExecutor(
        max_workers=len(geonet_bulk) + len(iris_bulk))
    st = Stream()

    def get_trace(client, bulk):
        try:
            tr = client.get_waveforms(*bulk)
        except Exception as e:
            tr = None
        return tr

    geonet_getter = partial(get_trace, client=GEONET_CLIENT)
    iris_getter = partial(get_trace, client=IRIS_CLIENT)

    futures = [executor.submit(geonet_getter, bulk=_bulk)
               for _bulk in geonet_bulk]
    futures += [executor.submit(iris_getter, bulk=_bulk)
                for _bulk in iris_bulk]

    traces = [f.result() for f in futures]
    for tr in traces:
        if tr:
            st += tr
    # For each station, if there are multiple channels, take H, then E, then B
    _st = Stream()
    for station in {tr.stats.station for tr in st}:
        sta_stream = st.select(station=station)
        channel_codes = [tr.stats.channel[0] for tr in sta_stream]
        if "H" in channel_codes:
            _st += sta_stream[channel_codes.index("H")]
        elif "E" in channel_codes:
            _st += sta_stream[channel_codes.index("E")]
        else:
            _st += sta_stream[0]
    st = _st.merge()
    st.write(filename, format="MSEED")
    return st


def pick_template_events(template_file: str, index: int = 0):
    from concurrent.futures import ProcessPoolExecutor as Executor
    from obsplus.events.json import cat_to_dict, dict_to_cat

    with open(template_file, "r") as f:
        cat_dict = json.load(f)
    # cat = read_events(template_file)

    executor = Executor(max_workers=2)

    sub_cat = dict(
            events=[cat_dict["events"][index]],
            resource_id=cat_dict["resource_id"],
            description=cat_dict["description"],
            comments=cat_dict["comments"],
            creation_info=cat_dict["creation_info"])
    event = dict_to_cat(sub_cat)[0]

    # Map station names
    for pick in event.picks:
        pick.waveform_id.station_code = STREWN_MAP.get(
            pick.waveform_id.station_code, pick.waveform_id.station_code)

    future_next_st = executor.submit(get_stream, event)

    # bar = progressbar.ProgressBar(max_value=len(cat_dict['events']))
    print(f"############### THERE ARE {len(cat_dict['events'])} EVENTS ######")
    for i in range(index, len(cat_dict['events'])):
        print(f"############# WORKING ON EVENT {i} ###################")
        # bar.update(i)
        sub_cat = dict(
            events=[cat_dict["events"][i]],
            resource_id=cat_dict["resource_id"],
            description=cat_dict["description"],
            comments=cat_dict["comments"],
            creation_info=cat_dict["creation_info"])
        event = dict_to_cat(sub_cat)[0]
        # Map station names
        for pick in event.picks:
            pick.waveform_id.station_code = STREWN_MAP.get(
                pick.waveform_id.station_code, pick.waveform_id.station_code)
        # print(sub_cat)
        # print("Getting stream")
        st = future_next_st.result()

        # Submit the next event for getting the stream
        if i < len(cat_dict['events']):
            sub_cat = dict(
                events=[cat_dict["events"][i + 1]],
                resource_id=cat_dict["resource_id"],
                description=cat_dict["description"],
                comments=cat_dict["comments"],
                creation_info=cat_dict["creation_info"])
            next_event = dict_to_cat(sub_cat)[0]
            # Map station names
            for pick in next_event.picks:
                pick.waveform_id.station_code = STREWN_MAP.get(
                    pick.waveform_id.station_code,
                    pick.waveform_id.station_code)
            future_next_st = executor.submit(get_stream, next_event)

        event_picked = pick_event_polarities(st=st, event=event)
        if event_picked is None:
            # Quit called internally, close the threads!
            break
        outfile = (
            f"../Templates/Polarities/"
            f"{event.resource_id.id.split('/')[-1]}.xml")
        print(f"Writing out to {outfile}")
        event_picked.write(outfile, format="QUAKEML")
        # event_picked = cat_to_dict(Catalog([event_picked]))
        # cat_dict["events"][i] = event_picked["events"][0]
        # print("Writing out")
        # with open(template_file, "w") as f:
        #     json.dump(cat_dict, f)
    print("Calling shutdown on futures")
    executor.shutdown()
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--start-index", type=int, default=0,
        help="Index to start picking from")
    args = parser.parse_args()

    cat = pick_template_events(
        "../Templates/STREWN_merged_polarities.json", args.start_index)
