"""
Tools for computing focal mechanisms from ObsPy events using HASHPy.
"""

import numpy as np
import logging

LOGGER = logging.getLogger("HASH")

import contextlib
import os
import glob
import json

from obspy.core.event import Event, Catalog
from obspy.clients.fdsn import Client
from obspy import read_events, read, UTCDateTime

from obsplus.events.json import dict_to_cat

from hashpy import HashPype, HashError

# TODO: Hash is fucking up the takeoff angles - getting some ~ +90degrees off.  Need to use Bayesian R code and NLL
def obspy_hash(event: Event, vmodel_files: list) -> Event:
    """
    Run HASH for a given event.
    """
    # Set configuration
    config = {"npolmin": 5,  # This controls the mimum polarities to compute a mechanism
              "max_agap": 120, # These control the maximum azimuthal gap allowed
              "max_pgap": 90,  # Maximum plunge gap (possibly?!).
              "dang": 2, # Degrees to step through
              "cangle": 10,  # Probability degrees?
              "nmc": 100, # Number of mechanisms to try
              "vmodels": vmodel_files}

    print("Initialising HashPype")
    hp = HashPype(**config)
    print("Reading event")
    hp.input(event, format="OBSPY")
    # Check that there is more than one polarity
    if hp.npol <= 1:
        print("Insufficient polarities")
        return None
    # Cope with poorly loaded depths - convert to km
    # hp.qdep /= 1000.0
    # Add in depth uncertainty
    hp.sez = event.preferred_origin().depth_errors.uncertainty / 1000.0
    print(f"Depth error: {hp.sez}")
    # Depth is in km - make sure that vmodel is in km as well!
    print("Loading velocity")
    hp.load_velocity_models()
    print("Generating trial data")
    hp.generate_trial_data()
    print("Calculating takeoff angles")
    hp.calculate_takeoff_angles()

    pass1 = hp.check_minimum_polarity()
    pass2 = hp.check_maximum_gap()
    if pass1 and pass2:
        print("Computing mechanisms")
        hp.calculate_hash_focalmech()
        hp.calculate_quality()
        hash_event = hp.output(format="OBSPY")
    else:
        print("Didn't pass user checks!")
        print(f"Number of polaritites: {hp.npol}")
        print(f"Maximum azimuthal gap: {hp.magap}")
        print(f"Maximum plunge gap: {hp.mpgap}")
        return None
    event_out = event.copy()
    event_out.focal_mechanisms = hash_event.focal_mechanisms

    hp.view_polarity_data()
    # Put the calculated takeoff angles into the arrivals
    for arrival in event_out.preferred_origin().arrivals:
        if arrival.phase not in "Pp":
            continue
        try:
            pick = arrival.pick_id.get_referred_object()
        except AttributeError:
            continue
        station = pick.waveform_id.station_code
        # Find the matching index
        for k in range(hp.npol):
            # print(str(hp.sname[k]))
            if hp.sname[k].decode() == station:
                azi, toa = hp.p_azi_mc[k, 0], hp.p_the_mc[k, 0]
                assert abs(azi - arrival.azimuth) < 5.0
                if arrival.takeoff_angle:
                    if abs(toa - arrival.takeoff_angle) > 10.0:
                        print(f"WARNING: {toa} is not close to input {arrival.takeoff_angle}")
                if not arrival.takeoff_angle:
                    print(f"Setting takeoff angle for {station} to {toa}")
                    arrival.takeoff_angle = toa
    return event_out


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
    


def make_arrival_for_pick(pick, origin):
    """ Make a fake arrival for a pick! """
    from obspy.core.event import Arrival
    from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

    if pick.waveform_id.network_code == "NZ":
        client = Client("GEONET")
    else:
        client = Client("IRIS")
    
    inv = client.get_stations(
        network=pick.waveform_id.network_code,
        station=pick.waveform_id.station_code, 
        startbefore=pick.time, endafter=pick.time)
    station = inv[0][0]
    dist, azi, baz = gps2dist_azimuth(
        lat1=origin.latitude, lat2=station.latitude,
        lon1=origin.longitude, lon2=station.longitude)
    arrival = Arrival(
        pick_id=pick.resource_id, phase=pick.phase_hint, azimuth=azi,
        distance=kilometer2degrees(dist / 1000.0))  # Other attributes not needed
    return arrival


def main():
    # Set up - write out velocities to file.
    velocities = [
        dict(top=-15.0, velocity=2.40),
        dict(top=-1.0, velocity= 1.90),
        dict(top=3.0, velocity= 4.34),
        dict(top=8.0, velocity= 6.00),
        dict(top=15.0, velocity= 6.08),
        dict(top=23.0, velocity= 6.46),
        dict(top=30.0, velocity= 8.01),
        dict(top=38.0, velocity= 8.16, moho=True),
        dict(top=48.0, velocity= 8.18),
        dict(top=65.0, velocity= 8.60),
        dict(top=85.0, velocity= 8.31),
        dict(top=105.0, velocity= 8.31),
        dict(top=130.0, velocity= 8.35),
        dict(top=155.0, velocity= 8.38),
        dict(top=185.0, velocity=8.45),
        dict(top=225.0, velocity=8.59),
        dict(top=275.0, velocity=8.68),
        dict(top=370.0, velocity=8.97),
    ]

    # vpvs = 1.78  # Unused
    with open("Kaik_1d.vmodel", "w") as f:
        for layer in velocities:
            f.write(f"{layer['top']} {layer['velocity']}\n")
    
    # Read in dict of full catalog - lighter to play with
    with open("../Locations/NLL_located_magnitudes_callibrated.json", "r") as f:
        located_cat_dict = json.load(f)
    located_origin_times = [
        UTCDateTime(ev["origins"][0]["time"])
        for ev in located_cat_dict["events"]]

    # Get list of template events to work with
    template_files = sorted(glob.glob("../Templates/Polarities/*.xml"))

    # Get inventory - used for checking station polarity reversal
    client = Client("IRIS")
    inv = client.get_stations(
        network="Z1", starttime=UTCDateTime(2016, 11,13), 
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

        # Strip out origin and use this in the template
        _template = template.copy()  # Copying for debugging in iPython
        _template.origins.append(matched_event.preferred_origin().copy())
        _template.preferred_origin_id = _template.origins[-1].resource_id
        # Hack to link arrivals to picks
        for arrival in _template.origins[-1].arrivals:
            old_linked_pick = arrival.pick_id.get_referred_object()
            for pick in _template.picks:
                if pick.waveform_id == old_linked_pick.waveform_id and pick.phase_hint == old_linked_pick.phase_hint:
                    # print(pick)
                    arrival.pick_id = pick.resource_id
                    break
            else:
                if arrival.phase in "Pp":
                    print(f"Pick not found for arrival: {arrival}")
                arrival.pick_id = None

        # Need to add in fake arrivals for the picks that are not referred to
        # - e.g. those not used for matching.
        linked_pick_ids = {arr.pick_id for arr in _template.origins[-1].arrivals}
        for pick in _template.picks:
            if pick.resource_id not in linked_pick_ids and pick.phase_hint[0].upper() == "P":
                fake_arrival = make_arrival_for_pick(
                    pick=pick, origin=_template.origins[-1])
                _template.origins[-1].arrivals.append(fake_arrival)

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
            if chan.dip > 0:
                # Reversed polarity, normal is dip = -90
                if pick.polarity == "positive":
                    print(f"Flipping polarity for {pick.waveform_id.station_code} to negative")
                    pick.polarity = "negative"
                elif pick.polarity == "negative":
                    print(f"Flipping polarity for {pick.waveform_id.station_code} to postive")
                    pick.polarity = "positive"

        template_fm = obspy_hash(
            _template.copy(), vmodel_files=["Kaik_1d.vmodel"])
        if template_fm:
            # Make and save plots
            figs = plot_focal_mechanisms(template_fm, show=False)
            for i, fig in enumerate(figs):
                outdir = f"../Plots/FM_plots/{template.resource_id.id.split('/')[-1]}"
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                fig.savefig(f"{outdir}/FM_{i}.png")
            focal_mechanism_cat.append(template_fm)

    focal_mechanism_cat.write(
        "../Templates/STREWN_merged_focal_mehcanisms.xml", format="QUAKEML")
    
    # Cleanup
    if os.path.isfile("Kaik_1d.vmodel"):
        os.remove("Kaik_1d.vmodel")


if __name__ == "__main__":
    main()