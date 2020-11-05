"""
Wrapper for running SIMUL locations - runs on event at a time to allow
for skipping events that have issues and to simply link Origins to events.
"""

import subprocess
import contextlib
import os

from typing import Union
from progressbar import ProgressBar

from obspy import read_events, Catalog
from obspy.core.event import Event

from cjc_utilities.io.write_simul_phs import write_simul
from cjc_utilities.io.simul_to_obspy import read_simul

# Path to SIMUL - change this for your SIMUL install.
SIMUL = "/home/chambeca/my_programs/Building/simul2014/bin/simul2014"
# Where your CNTL, STNS and MOD file are located.
LOC_DIR = "/mnt/Big_Boy/kaikoura-afterslp/Locations"


@contextlib.contextmanager
def working_directory(path):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.

    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def _clean_simul_files(directory: str):
    temp_files = [
        'hypo71list', 'pseudo', 'TEST_PRINT', 'hypo.gmt', 'newstns', 
        'residuals', 'tteq.out', 'finalsmpout', 'hyp_prt.out', 'nodes.out',
        'resol.out', 'tteqrev.out', 'fort.38', 'itersum', 'output',
        'summary']
    for f in temp_files:
        if os.path.isfile(f"{directory}/{f}"):
            os.remove(f"{directory}/{f}")


def _simul_locate(event: Event) -> Event:
    """ Locate an event using SIMUL - origin is appeneded to event origins. """
    _clean_simul_files(LOC_DIR)  # Remove old files.
    event.preferred_origin_id = event.origins[0].resource_id
    events_written = write_simul(
        Catalog([event]), f"{LOC_DIR}/EQKS", min_stations=3)
    if events_written == 0:
        return None
    with working_directory(LOC_DIR):
        result = subprocess.run(SIMUL, capture_output=True, check=False)
        if result.returncode != 0:
            print("Could not locate, captured output:\n"
                  f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            return None
    if result.returncode == 0:
        try:
            simul_event = read_simul(
                phase_file=f"{LOC_DIR}/EQKS", 
                location_file=f"{LOC_DIR}/hypo71list",
                max_sep=None)
        except (IndexError, ValueError):
            return None  # No location
    else:
        return None
    if len(simul_event) != 1:
        print("Could not read event")
        return None
    event_out = event.copy()
    simul_origin = simul_event[0].origins[0].copy()
    event_out.origins.append(simul_origin)
    # Fix the pick-ids in the arrivals
    for arrival in simul_origin.arrivals:
        simul_pick = arrival.pick_id.get_referred_object()
        if simul_pick is None:
            print("No pick matched to arrival... WTF?")
            continue
        pick_station = simul_pick.waveform_id.station_code.strip()
        original_pick = [p for p in event_out.picks 
                         if p.phase_hint[0] == simul_pick.phase_hint[0]
                         and p.waveform_id.station_code == pick_station
                         and abs(p.time - simul_pick.time) < 0.5]
        if len(original_pick) == 0:
            print(f"No pick matched, WTF? {simul_pick}")
            continue
        if len(original_pick) > 1:
            # Get the pick closest to the simul-pick - simul rounds output :(
            _picks = [(p, abs(p.time - simul_pick.time)) for p in original_pick]
            _picks.sort(key=lambda p: p[1])
            original_pick = [_picks[0][0]]
        arrival.pick_id = original_pick[0].resource_id
    event_out.preferred_origin_id = simul_origin.resource_id
    return event_out


def simul_locate(catalog: Union[Event, Catalog]) -> Catalog:
    if isinstance(catalog, Event):
        catalog = Catalog([catalog])
    catalog_out = Catalog()
    bar = ProgressBar(max_value=len(catalog))
    for i, event in enumerate(catalog):
        bar.update(i)
        try:
            simul_event = _simul_locate(event)
        except Exception as e:
            print(f"Error: {e}")
            simul_event = None
        if simul_event:
            catalog_out.append(simul_event)
    return catalog_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Locate a catalogue using SIMUL")
    
    parser.add_argument(
        "-i", "--input", help="Obspy readable catalogue file", type=str,
        default="/mnt/Big_Boy/kaikoura-afterslp/Detections/AWS_run1/"
                "bad_picks_removed_2009-01-01-2019-11-01_repicked_catalog.xml")
    parser.add_argument(
        "-o", "--output", help="Output catalogue - written to QUAKEML format",
        type=str, default=f"{LOC_DIR}/SIMUL_located.xml")
    
    args = parser.parse_args()
    print(f"Reading events from {args.input}")
    cat_in = read_events(args.input)
    cat_in.events.sort(key=lambda e: e.origins[0].time)
    print(f"Starting location")
    cat_out = simul_locate(cat_in)
    _clean_simul_files(LOC_DIR)
    print(f"Location complete, writing to {args.output}")
    cat_out.write(args.output, format="QUAKEML")
