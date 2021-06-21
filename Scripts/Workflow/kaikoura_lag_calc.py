"""
A bug in kaikoura detect (now fixed) missed channels when running lag-calc.

This script takes the days for which detections were made, and runs lag-calc
for them.

"""

import logging
import os

from obspy import Stream, Catalog, UTCDateTime

from eqcorrscan import Party, Tribe

from get_data import TemplateParams


TOTAL_CORES = 6
WORK_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRIBE_FILE = "{0}/Templates/STREWN_merged_catalog_4s_1.5Hz-12Hz_2021.tgz".format(WORK_PATH)
DETECTION_PATH = "{0}/Detections_2021".format(WORK_PATH)

Logger = logging.getLogger(__name__)


def lag_calc(party_file: str, st: Stream):
    party = Party().read(party_file)

    # Need to hack to add in chans not used for detection
    for fam in party:
        # TODO: this is where the bug was - fam.template.st was template.st
        template_stachans = {
                (tr.stats.station, tr.stats.channel) 
                for tr in fam.template.st}
        stachans = template_stachans.intersection(
            {(tr.stats.station, tr.stats.channel) for tr in st})
        print(stachans)
        for det in fam:
            det.chans = stachans
            det._calculate_event(template=fam.template)
    party.lag_calc(
        stream=st, pre_processed=True, shift_len=0.5, min_cc=0.4,
        cores=TOTAL_CORES, process_cores=1)
    repicked_cat = Catalog()
    for family in party:
        for det in family:
            event = det.event.copy()
            det._calculate_event(
                template=family.template, estimate_origin=True)
            event.origins.append(det.event.origins[0])
            det.event = event
            repicked_cat.append(event)
    
    parts = list(os.path.split(party_file))
    parts[-1] = "relagged_" + parts[-1]
    outfile = "/".join(parts)
    Logger.info(
        "Writing party of {0} detections to {1}".format(
            len(party), outfile))
    party.write(outfile, overwrite=True)
    cat_file = party_file.replace("_party.tgz", "_repicked_catalog.xml")
    
    repicked_cat.write(cat_file, format="QUAKEML")


if __name__ == "__main__":
    """
    Allow to be called on an iterative basis from the shell to allow for lack
    of python garbage collection and associated memory issues
    """
    import argparse
    import subprocess
    from obspy import read

    import glob

    parser = argparse.ArgumentParser(
        description="Re-run lag-calc for Kaikoura")
    parser.add_argument("-s", "--start", type=int, help="Index to start at", 
                        default=0)
    parser.add_argument("-e", "--end", type=int, help="Index to end at",
                        default=None)

    args = parser.parse_args()
    _start, _end = args.start, args.end

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")


    tribe = Tribe().read(TRIBE_FILE)
    # Get the first stream
    template_params = TemplateParams.from_template(tribe[0])
    template_params.seed_ids = list(
        {tr.id for template in tribe for tr in template.st})

    template_params_filename = f"temp_waveforms/_template_params.json"
    template_params.write(template_params_filename)

    pick_files = sorted(glob.glob(f"{DETECTION_PATH}/*_repicked_catalog.xml"))

    _end = min(_end or len(pick_files), len(pick_files))
    assert _start < _end, "Needs something to work with!"


    pick_file = pick_files[_start]
    parts = pick_file.split('/')[-1].split("_repicked")[0]
    startdate, enddate = UTCDateTime(parts[0:27]), UTCDateTime(parts[28:])
    # Open subprocess downloading the day of data
    st_filename_format = "temp_waveforms/{startdate}-{enddate}.ms"
    st_filename = st_filename_format.format(startdate=startdate, enddate=enddate)
    if os.path.isfile(st_filename):
        os.remove(st_filename)
    downloader_process = subprocess.Popen(
        ["python", "get_data.py", f"-s={startdate}", f"-l=86400", 
         f"-p={template_params_filename}", f"-o={st_filename}", "-c=4"])

    for i in range(_start, _end):
        Logger.info(f"working on file {i} of {len(pick_files)}")
        pick_file = pick_files[i]
        party_file = pick_file.replace("_repicked_catalog.xml", "_party.tgz")
        assert os.path.isfile(party_file)
        # Get the days stream from the future
        ret_val = downloader_process.wait()  # Wait for process to finish
        # st = read(st_filename).merge()

        j = i + 1
        if j <= len(pick_files):
            pick_file = pick_files[j]
            parts = pick_file.split('/')[-1].split("_repicked")[0]
            startdate, enddate = UTCDateTime(parts[0:27]), UTCDateTime(parts[28:])
            # Open subprocess downloading the day of data
            st_filename_format = "temp_waveforms/{startdate}-{enddate}.ms"
            next_st_filename = st_filename_format.format(
                startdate=startdate, enddate=enddate)
            if os.path.isfile(next_st_filename):
                os.remove(next_st_filename)
            # Submit the job for the next block
            downloader_process = subprocess.Popen(
                ["python", "get_data.py", f"-s={startdate}", f"-l=86400", 
                f"-p={template_params_filename}", f"-o={next_st_filename}", "-c=4"])
        # Submit job to main threads
        # lag_calc(party_file=party_file, st=st)
        if ret_val == 0:
            lag_process = subprocess.Popen(
                ["python", "relag.py", f"-p={party_file}", f"-s={st_filename}"])
            lag_ret_val = lag_process.wait()
            if lag_ret_val:
                Logger.error(f"Some error in lag-calc, lag_ret_val is {lag_ret_val}")
        else:
            Logger.error(f"Some error downloading data, ret_val is {ret_val}")
        if os.path.isfile(st_filename):
            os.remove(st_filename)  # cleanup

        st_filename = next_st_filename
        # Get the next days stream
        Logger.info("Fin")
