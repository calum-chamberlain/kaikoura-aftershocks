"""
Preparing template files for NonLinLoc, then Bayesian FM calculation (using R)
"""

import numpy as np
import logging

import contextlib
import os
import glob
import json

from obspy.core.event import Event, Catalog
from obspy.clients.fdsn import Client
from obspy import read_events, read, UTCDateTime

from obsplus.events.json import dict_to_cat


#Stations to ignore inverted motion in inventory for - The Geospace HS-1-3C may be wrong on IRIS
UNFLIPPED_STATIONS = ["CSCP", "STAW", "CSWB", "STWC", "LTW2"]


def plot_focal_mechanisms(event, show=True):
    from focal_mechanism_plotting import FocalMechanism, Polarity, NodalPlane

    polarities = []
    for arr in event.preferred_origin().arrivals:
        try:
            pick = arr.pick_id.get_referred_object()
        except AttributeError:
            continue
        if pick.polarity not in ("positive", "negative"):
            continue
        if not arr.takeoff_angle:
            continue

        polarities.append(Polarity(
            azimuth=arr.azimuth, toa=arr.takeoff_angle, polarity=pick.polarity))
    
    figs = []
    for focal_mechanism in event.focal_mechanisms:
        np1 = focal_mechanism.nodal_planes.nodal_plane_1
        np2 = focal_mechanism.nodal_planes.nodal_plane_2
        np1 = NodalPlane(strike=np1.strike, dip=np1.dip, rake=np1.rake)
        np2 = NodalPlane(strike=np2.strike, dip=np2.dip, rake=np2.rake)
        fm = FocalMechanism(
            nodal_plane_1=np1,
            nodal_plane_2=np2,
            polarities=polarities)
        figs.append(fm.plot(show=show))
    return figs


ONSETS = {"i": "impulsive", "e": "emergent"}
ONSETS_REVERSE = {"impulsive": "i", "emergent": "e"}
POLARITIES = {"c": "positive", "u": "positive", "d": "negative"}
POLARITIES_REVERSE = {"positive": "u", "negative": "d"}


def write_nlloc_obs(event, filename):
    """
    Write a NonLinLoc Phase file (NLLOC_OBS) from a
    :class:`~obspy.core.event.Catalog` object.

    :type filename: str or file
    :param filename: Filename to write or open file-like object.
    """
    info = []

    # Open filehandler or use an existing file like object.
    if not hasattr(filename, "write"):
        file_opened = True
        fh = open(filename, "wb")
    else:
        file_opened = False
        fh = filename

    fmt = '%s %s %s %s %s %s %s %s %7.4f GAU %9.2e %9.2e %9.2e %9.2e %9.2e'
    for pick in event.picks:
        wid = pick.waveform_id
        station = wid.station_code or "?"
        component = wid.channel_code and wid.channel_code[-1].upper() or "?"
        if component not in "ZNEH":
            component = "?"
        onset = ONSETS_REVERSE.get(pick.onset, "?")
        phase_type = pick.phase_hint or "?"
        polarity = POLARITIES_REVERSE.get(pick.polarity, "?")
        date = pick.time.strftime("%Y%m%d")
        hourminute = pick.time.strftime("%H%M")
        seconds = pick.time.second + pick.time.microsecond * 1e-6
        time_error = pick.time_errors.uncertainty or -1
        if pick.time_errors.confidence_level == 0.0:
            priorwt = 0
        else:
            priorwt = 1
        if time_error == -1:
            try:
                time_error = (pick.time_errors.upper_uncertainty +
                              pick.time_errors.lower_uncertainty) / 2.0
            except Exception:
                pass
        info_ = fmt % (station.ljust(6), "?".ljust(4), component.ljust(4),
                       onset.ljust(1), phase_type.ljust(6), polarity.ljust(1),
                       date, hourminute, seconds, time_error, -1, -1, -1, priorwt)
        info.append(info_)

    if info:
        info = "\n".join(sorted(info) + [""])
    else:
        msg = "No pick information, writing empty NLLOC OBS file."
        print(msg)
    fh.write(info.encode())

    # Close if a file has been opened by this function.
    if file_opened is True:
        fh.close()
    return


def write_nonlinloc_obs(catalog: Catalog, directory: str):
    """ 
    Write NonLinLoc obs files for each event in a catalog - hack to 
    provide a-priori weight 
    
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    for event in catalog:
        outfile = os.path.join(directory, event.resource_id.id.split('/')[-1])
        write_nlloc_obs(event, outfile)
        print(f"Written to {outfile}")
    return


def main():
    
    # Read in dict of full catalog - lighter to play with
    with open("../../Locations/NonLinLoc_NZW3D_2.2/NLL_located_magnitudes_callibrated.json", "r") as f:
        located_cat_dict = json.load(f)
    located_origin_times = [
        UTCDateTime(ev["origins"][0]["time"])
        for ev in located_cat_dict["events"]]

    # Get list of template events to work with
    template_files = sorted(glob.glob("../../Templates/Polarities/*.xml"))

    # Get inventory - used for checking station polarity reversal
    client = Client("IRIS")
    inv = client.get_stations(
        network="Z1", starttime=UTCDateTime(2016, 11, 13), 
        endtime=UTCDateTime(2018, 1, 1), level="channel")
    client = Client("GEONET")
    inv += client.get_stations(
        network="NZ", starttime=UTCDateTime(2016, 11,13), 
        endtime=UTCDateTime(2018, 1, 1), level="channel")    

    focal_mechanism_cat = Catalog()
    for template_file in template_files:
        template = read_events(template_file)[0]
        # Find the template event in the final located events
        t_diffs = np.array(
            [t - template.origins[0].time for t in located_origin_times])
        # Threshold in seconds set to 3s difference in origin time
        possible_locs = np.where(np.abs(t_diffs) < 3.0)[0]
        if len(possible_locs) < 1:
            print("!!!!!!!!!WARNING: Could not find matching event for template!")
            print(template)
            continue
        # Convert to ObsPy
        possible_matches = dict_to_cat(
            {'events': [located_cat_dict["events"][loc] for loc in possible_locs],
             'resource_id': "bob",
             'description': "",
             'comments': "",
             'creation_info': ""})
        # Compare pick times
        matched_event = None
        for possible_match in possible_matches:
            similar_picks = 0
            for pick in template.picks:
                matched_picks = [
                    p for p in possible_match.picks 
                    if p.waveform_id.station_code == pick.waveform_id.station_code 
                    and p.phase_hint == pick.phase_hint]
                for matched_pick in matched_picks:
                    if abs(matched_pick.time - pick.time) < 0.5:
                        similar_picks += 1
            if similar_picks > len(possible_match.picks) / 2:
                print("Found match")
                matched_event = possible_match
                break
        else:
            print("No match found")
            continue

        # We want the matched-event and we will add polarities in to that
        picks_differ = False  # There is potential for rate issues
        _template = matched_event.copy()
        for pick in _template.picks:
            # Only care about P-picks - we don't need to add unused S-picks
            if pick.phase_hint not in "Pp":
                continue
            # Find similar pick in "template"
            strong_motion = False
            for template_pick in template.picks:
                if template_pick.phase_hint not in "Pp":
                    continue
                if pick.waveform_id == template_pick.waveform_id:
                    # This should be a match!
                    if abs(pick.time - template_pick.time) > 0.5:
                        print(f"{pick.time} is different to {template_pick.time}")
                        picks_differ = True
                    break
                # Sometimes picks were made on strong-motions
                elif pick.waveform_id.station_code == template_pick.waveform_id.station_code:
                    # This is fine, we can skip
                    strong_motion = True
            else:
                if not strong_motion:
                    # We don't need to warn about strong-motions - we removed those.
                    print(f"No match found for {pick}")
                continue
            # Assign manually checked polarity to template pick.
            pick.polarity = template_pick.polarity
        if picks_differ:
            print("Match rejected due to differing picks")
            continue

        # Add in the other picks that have polarities
        used_stations = {pick.waveform_id.station_code for pick in _template.picks if pick.phase_hint in "Pp"}
        for pick in template.picks:
            if pick.phase_hint not in "Pp":
                continue
            if not pick.polarity or pick.polarity == "undecidable":
                continue
            if pick.waveform_id.station_code in used_stations:
                # Don't add in picks that we already have
                continue
            _pick = pick.copy()
            _pick.time_errors.confidence_level = 0.0 # Do not use this pick in location
            _template.picks.append(_pick)
        
        # Check polarities
        for pick in _template.picks:
            if pick.phase_hint not in "Pp":
                continue
            sta = inv.select(
                network=pick.waveform_id.network_code,
                station=pick.waveform_id.station_code,
                channel=pick.waveform_id.channel_code,
                location=pick.waveform_id.location_code,
                time=pick.time)
            try:
                chan = sta[0][0][0]
            except IndexError:
                print(f"WARNING: No inventory info for pick: {pick}")
                continue
            if chan.dip > 0 and pick.waveform_id.station_code not in UNFLIPPED_STATIONS:
                # Reversed polarity, normal is dip = -90
                if pick.polarity == "positive":
                    print(f"Flipping polarity for {pick.waveform_id.station_code} to negative")
                    pick.polarity = "negative"
                elif pick.polarity == "negative":
                    print(f"Flipping polarity for {pick.waveform_id.station_code} to postive")
                    pick.polarity = "positive"

        focal_mechanism_cat.append(_template)

    # Remove events with fewer than 8 polarities
    focal_mechanism_cat = Catalog(
        [ev for ev in focal_mechanism_cat 
         if len([p for p in ev.picks if p.polarity and p.polarity != "undecidable"]) > 8])
    # focal_mechanism_cat.write(
    #     "../Templates/Polarities/Manually_checked_polarities.xml",
    #     format="QUAKEML")
    
    write_nonlinloc_obs(
        focal_mechanism_cat, 
        "../../Templates/Polarities/Manually_checked_polarities_NLL")


if __name__ == "__main__":
    main()
