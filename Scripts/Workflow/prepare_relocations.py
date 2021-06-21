"""
Prepare dt.cc and dt.ct files for relocation.

"""

import os

from typing import Dict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial

from obspy import read_events, Stream, read, Catalog
from obspy.core.event import Event

from eqcorrscan.utils.catalog_to_dd import (
    write_event, write_phase, write_catalog, write_correlations)

from pick_magnitudes import get_waveforms_for_event


WAVEFORM_DB = "../Detected_events"
LOWCUT, HIGHCUT = (2., 20.)


def get_waveform(event: Event, rebuild: bool = False) -> Stream:

    origin = event.preferred_origin() or event.origins[0]
    event_dir = f"{WAVEFORM_DB}/Y{origin.time.year}/R{origin.time.julday:03d}"
    event_fname = event.resource_id.id.split('/')[-1]
    event_dir = f"{event_dir}/{event_fname}"

    wavefile = f"{event_dir}/{event_fname}.ms"
    # ori = event.preferred_origin() or event.origins[-1]
    # wavefile = (
    #     f"{WAVEFORM_DB}/{ori.time.strftime('%Y/%m')}/"
    #     f"{ori.time.strftime('%Y-%m-%dT%H-%M-%S')}.ms")
    if not os.path.isfile(wavefile) or rebuild == True:
        if not os.path.isdir(os.path.dirname(wavefile)):
            os.makedirs(os.path.dirname(wavefile))
        st = get_waveforms_for_event(event=event, length=140., pre_origin=20.)
        st.write(wavefile, format="MSEED")
    else:
        st = read(wavefile)
    return st.detrend().filter(
        "bandpass", freqmin=LOWCUT, freqmax=HIGHCUT, corners=4, zerophase=True)


def make_wavedict(
    catalog: Catalog, 
    rebuild: bool = False, 
    cores: int = None,
) -> Dict[str, Stream]:

    cores = cores or cpu_count()
    chunksize = len(catalog) // cores
    func = partial(get_waveform, rebuild=rebuild)
    wavedict = dict()
    with ProcessPoolExecutor(max_workers=cores) as executor:
        for event, st in zip(catalog, executor.map(func, catalog, chunksize=chunksize)):
            event_id = event.resource_id.id
            wavedict.update({event_id: st})
    return wavedict


if __name__ == "__main__":
    import json
    from obsplus import json_to_cat

    cat = read_events(
        "../../Locations/NonLinLoc_NZW3D_2.2/NLL_located_magnitudes_callibrated.xml")

    # print("Reading from json")
    # with open("../Locations/NLL_located_magnitudes_callibrated.json", "r") as f:
    #     cat_dict = json.load(f)
    # print("Converting to Catalog")
    # cat = json_to_cat(cat_dict)
    print(f"Working on {len(cat)} events")

    if os.path.isfile("event_mapper.json"):
        with open("event_mapper.json", "r") as f:
            event_mapper = json.load(f)
    else:
        print("Making phase.dat")
        event_mapper = write_phase(cat)
        print("Writing event mapper to event_mapper.json")
        with open("event_mapper.json", "w") as f:
            json.dump(event_mapper, f)
    
    print("Making dt.ct file")
    event_mapper = write_catalog(
        cat, event_id_mapper=event_mapper, max_sep=5, min_link=8)

    print("Constructing the stream dictionary")
    stream_dict = make_wavedict(cat, rebuild=False, cores=10)

    print("Making the dt.cc file")
    event_mapper = write_correlations(
        cat, stream_dict=stream_dict, extract_len=2.0, pre_pick=0.2, 
        shift_len=0.1, lowcut=None, highcut=None, max_sep=5.0, min_link=8,
        min_cc=0.5, interpolate=False)
