"""
Script for Post-processing detections after lag-calc.

Calum Chamberlain: 29/12/2019
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from obspy import read_events, Catalog, Stream, read
from obspy.core.event import Event


def average_cc(event):
    """ 
    Calculate average correlation from comments in picks. 
    
    Averages each station first, then averages over network to avoid bias from
    one well correlated station.
    """
    stations = {p.waveform_id.station_code for p in event.picks}
    average_cc = 0.0
    n_picks = 0
    for station in stations:
        picks = [pick for pick in event.picks 
                 if pick.waveform_id.station_code == station]
        n_station_picks = 0
        station_cc = 0.0
        for pick in picks:
            pick_cc = get_pick_cc(pick)
            if pick_cc is not None:
                station_cc += pick_cc
                n_station_picks += 1
        if station_cc != 0.0:
            average_cc += (station_cc / n_station_picks)
            n_picks += 1
    return average_cc / n_picks


def get_pick_cc(pick):
    if len(pick.comments) == 0:
        return 0.0
    for comment in pick.comments:
        try:
            comment_type, value = comment.text.split("=")
            if comment_type == "cc_max":
                return float(value)
        except ValueError:
            pass
    else:
        return 0.0


def plot_average_cc_time(catalog, bin_width=0.005):
    """
    Plot the average correlation in time.
    """
    import matplotlib.pyplot as plt

    all_detection_times, correlations = ([], [])
    for event in catalog:
        try:
            origin = event.preferred_origin() or event.origins[0]
            detect_time = origin.time
        except IndexError:
            detect_time = min([p.time for p in event.picks])
        all_detection_times.append(detect_time.datetime)
        correlations.append(average_cc(event))
    
    fig, (ax, ax_hist) = plt.subplots(
        1, 2, sharey=True, gridspec_kw = {'width_ratios':[3, 1]})
    ax.scatter(x=np.array(all_detection_times), y=np.array(correlations), c='k')
    ax.set_xlabel("Detection time")
    ax.set_ylabel("Average Correlation")
    ax.autoscale(tight=True)
    # Make a histogram
    bins = np.arange(min(correlations), max(correlations), bin_width)
    ax_hist.hist(correlations, bins=bins, orientation='horizontal', color='k')
    ax_hist.set_xlabel("Bin size")
    ax_hist.autoscale(tight=True)
    # Plot cumulative as well
    ax_cum = ax_hist.twiny()
    ax_cum.plot(np.arange(len(correlations)), 
                 sorted(correlations, reverse=True), c='r')
    ax_cum.set_xlabel("Cumulative number")
    ax_cum.autoscale(tight=True)
    return fig


def rethreshold_catalog(catalog, minimum_individual_cc, minimum_average_cc, 
                        minimum_stations):
    """
    Rethreshold a catalog based on correlation and number of stations.
    
    :param catalog: catalog to rethreshold
    :param minimum_individual_cc: 
        Minimum normalised correlation value per pick.
    :param minimum_average_cc: 
        Minimum average correlation after removing picks below 
        minimum_individual_cc
    :param minimum_stations: Minimum number of picked stations for an event.
    """
    rethresholded_catalog = Catalog()
    for event in catalog:
        # Check individual picks first
        _picks = []
        for pick in event.picks:
            if get_pick_cc(pick) < minimum_individual_cc:
                continue
            _picks.append(pick)
        event.picks = _picks
        # Check number of stations next
        if len(set([pick.waveform_id.station_code 
                    for pick in event.picks])) < minimum_stations:
            continue
        # Finally check the average correlation
        if average_cc(event) < minimum_average_cc:
            continue
        # If we got to here, all tests pass
        rethresholded_catalog.append(event)
    return rethresholded_catalog


def rethreshold_detected_events(minimum_individual_cc, minimum_average_cc, 
                                minimum_stations):
    rethresholded_catalog = Catalog()
    repicked_files = glob.glob("Detections/*.xml")
    repicked_files.sort()
    for repicked_file in repicked_files:
        print("Working on file: {0}".format(repicked_file))
        catalog = read_events(repicked_file)
        in_len = len(catalog)
        _rethresholded_catalog = rethreshold_catalog(
            catalog=catalog,
            minimum_individual_cc=minimum_individual_cc,
            minimum_average_cc=minimum_average_cc,
            minimum_stations=minimum_stations)
        out_len = len(_rethresholded_catalog)
        rethresholded_catalog += _rethresholded_catalog
        print("After thresholding {0} events were removed and {1} "
              "remain".format(in_len - out_len, out_len))
    return rethresholded_catalog


def get_self_detections(template_cat, catalog, threshold=0.85):
    first_pick_times = [
        min([p.time for p in event.picks]) for event in catalog]
    good_pairs, bad_pairs = ([], [])
    for event in template_cat:
        origin = event.preferred_origin() or event.origins[0]
        time_diffs = [abs(p_time - origin.time) for p_time in first_pick_times]
        time_diff = min(time_diffs)
        closest_det = catalog[time_diffs.index(time_diff)]
        av_cc = average_cc(closest_det)
        if av_cc < threshold:
            bad_pairs.append({'template': event, 'detection': closest_det})
        else:
            good_pairs.append({'template': event, 'detection': closest_det})
    return good_pairs, bad_pairs


def get_snr(event: Event, waveform_dir: str) -> dict:
    """ Get SNR for an event. Return dict of SNRs keyed by seed_id. """
    from QC_detections import get_stream_for_event

    st = get_stream_for_event(event=event, waveform_dir=waveform_dir)
    snr_dict = dict()
    for pick in event.picks:
        try:
            tr = st.select(
                station=pick.waveform_id.station_code,
                network=pick.waveform_id.network_code,
                channel=pick.waveform_id.channel_code)[0]
        except IndexError:
            continue
        signal = np.sqrt(
            np.mean(np.square(
                tr.slice(pick.time, pick.time + 5).data)))
        noise = np.sqrt(
            np.mean(np.square(
                tr.slice(pick.time - 10, pick.time - .2).data)))
        seed_id = "{net}.{sta}.{loc}.{chan}".format(
            net=pick.waveform_id.network_code, 
            sta=pick.waveform_id.station_code,
            loc=pick.waveform_id.location_code or "",
            chan=pick.waveform_id.channel_code
        )
        snr_dict.update({seed_id: signal / noise})
    return snr_dict


def thresholdable_criteria(catalog):
    """ Get a dictionary of "thresholdable" criteria for a catalogue. """
    event_dict = dict()
    for event in catalog:
        thresholds = dict(
            nobs=len(event.picks),
            nppicks=len([p for p in event.picks if p.phase_hint[0] == "P"]),
            average_correlation=average_cc(event),
            template='_'.join(
                event.resource_id.id.split('/')[-1].split('_')[0:-1])
        )
        event_dict.update({event.resource_id.id.split('/')[-1]: thresholds})
    return event_dict


def origin_time_decluster(
    catalog: Catalog, 
    origin_sep: float = 1.0,
    metric: str = "avg_cor"
) -> Catalog:
    """
    Decluster based on origin-time
    """
    import numpy as np
    from eqcorrscan.utils.findpeaks import decluster
    from eqcorrscan.core.match_filter.helpers import _total_microsec
    assert metric in ("avg_cor", "cor_sum")

    declustered = Catalog()
    
    if metric == "cor_sum":
        detect_info = [
            (ev.origins[0].time, len(ev.picks) * average_cc(ev)) 
            for ev in catalog]
    else:
        detect_info = [
            (ev.origins[0].time, average_cc(ev)) for ev in catalog]
    min_det = sorted([d[0] for d in detect_info])[0]
    detect_vals = np.array([d[1] for d in detect_info], dtype=np.float32)
    detect_times = np.array([
        _total_microsec(d[0].datetime, min_det.datetime)
        for d in detect_info])
    # Trig_int must be converted from seconds to micro-seconds
    peaks_out = decluster(
        peaks=detect_vals, index=detect_times, trig_int=origin_sep * 10 ** 6)
    # Need to match both the time and the detection value
    for ind in peaks_out:
        matching_time_indices = np.where(detect_times == ind[-1])[0]
        matches = matching_time_indices[
            np.where(detect_vals[matching_time_indices] == ind[0])[0][0]]
        declustered.append(catalog[matches])
    return declustered


if __name__ == "__main__":
    # Options: 
    #   1) build from detect-files, 
    #   2) remove events with fewer than n stations, 
    #   3) decluster origins.
    import argparse
    import glob

    from obspy import read_events

    parser = argparse.ArgumentParser(
        description="Build, rethreshold and decluster detections")
    parser.add_argument(
        "-b", "--build", 
        help="Build catalog from detections: glob-able string", type=str)
    parser.add_argument(
        "-i", "--input", type=str,
        help="Input catalog file, not needed with the --build/-b option")
    parser.add_argument(
        "-r", "--remove", help="Remove detections with fewer than n stations",
        type=int)
    parser.add_argument(
        "-d", "--decluster", type=float,
        help="Decluster based on n second origin-time separation")
    args = parser.parse_args()

    def write_cat(cat, tag=None, path="."):
        if tag:
            tag = tag + "_"
        else:
            tag = ""
        cat.events.sort(key=lambda e: e.origins[0].time)
        starttime = cat[0].origins[0].time.strftime("%Y-%m-%d")
        endtime = cat[-1].origins[0].time.strftime("%Y-%m-%d")
        out_file = "{0}/{1}{2}-{3}_detections.xml".format(
            path, tag, starttime, endtime)
        print("Writing complete catalog to {0}".format(out_file))
        cat.write(out_file, format="QUAKEML")

    if args.build:
        cat_files = glob.glob(args.build)
        cat_files.sort()
        args.input = cat_files[0]
        cat = Catalog()
        for i, cat_file in enumerate(cat_files):
            print("Reading from {0} file {1} of {2}".format(
                cat_file, i, len(cat_files)))
            cat += read_events(cat_file)
        write_cat(cat=cat, path=os.path.dirname(cat_files[0]))
    else:
        if args.input is None:
            raise NotImplementedError("Requires an input file.")
        cat = read_events(args.input)
    
    tag = ""
    if args.remove is not None:
        rethresh_cat = rethreshold_catalog(
            catalog=cat, minimum_average_cc=0.5, minimum_individual_cc=0.3,
            minimum_stations=args.remove)
        tag = "minimum_{0}_stations".format(args.remove) + tag
        write_cat(
            cat=rethresh_cat, path=os.path.dirname(args.input), 
            tag=tag)
        cat = rethresh_cat

    if args.decluster is not None:
        declustered_cat = origin_time_decluster(
            catalog=cat, origin_sep=args.decluster)
        tag = "declustered_{0}s".format(args.decluster) + tag
        write_cat(
            cat=declustered_cat, path=os.path.dirname(args.input), 
            tag=tag)  



