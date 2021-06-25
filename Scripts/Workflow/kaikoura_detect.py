"""
Script for detecting earthquakes using Kaikoura aftershock templates.

Written by: Calum J Chamberlain
Date:       22/06/2018
"""
import argparse
import os
import logging
import sys
import traceback
import numpy as np

from typing import List
from tempfile import NamedTemporaryFile
from multiprocessing import cpu_count

from obspy import UTCDateTime, Catalog, Stream, read
from obspy.clients.fdsn import Client

# sys.path.insert(0, "/nesi/project/nesi02337/EQcorrscan")
from eqcorrscan import Tribe
from eqcorrscan.core.match_filter.matched_filter import _group_process

from get_data import TemplateParams

WORK_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIBE_FILE = "{0}/Templates/STREWN_merged_catalog_4s_1.5Hz-12Hz_2021.tgz".format(WORK_PATH)
DETECTION_PATH = "{0}/Detections_2021".format(WORK_PATH)
EXPECTED_LENGTH = 4 * 30
TOTAL_CORES = min(cpu_count(), 8)
CORES_OUTER = 1
CORES_INNER = TOTAL_CORES // CORES_OUTER

DETECTION_STATIONS = {
    'BHW', 'BSWZ', 'CAW', 'CMWZ', 'DSZ', 'DUWZ', 'GVZ', 'KHZ',
    'KIW', 'LTZ', 'MQZ', 'MRNZ', 'MTW', 'NNZ', 'OGWZ', 'OXZ',
    'PLWZ', 'TCW', 'THZ', 'TKNZ', 'TUWZ', 'WEL'}

Logger = logging.getLogger(__name__)


import numpy as np

from eqcorrscan.utils.correlate import (
    register_array_xcorr, numpy_normxcorr)


def detect(tribe, st, gpu):
    # To ensure catalog continuity we need to use just a subset of stations that
    # were continuous in the study period
    detection_st = Stream([tr for tr in st if tr.stats.station in DETECTION_STATIONS])

    Logger.info("Running detections between {0} and {1}".format(
        st[0].stats.starttime, st[0].stats.endtime))
    xcorr_func = "fftw"
    GROUP_SIZE = 200
    if gpu:
        xcorr_func = "fmf"
        GROUP_SIZE = 200
    try:
        party = tribe.detect(
            stream=detection_st, threshold=10.0, threshold_type="MAD", trig_int=4,
            plot=False, daylong=False, parallel_process=False, xcorr_func=xcorr_func,
            concurrency="concurrent", cores=CORES_INNER, ignore_length=False,
            save_progress=False, process_cores=CORES_INNER, 
            cores_outer=CORES_OUTER, return_stream=False, group_size=GROUP_SIZE,
            pre_processed=True)
    except Exception as e:
        Logger.error(e)
        traceback.print_exc()
        with open("Failed_days.txt", "a") as f:
            f.write("Failed to detect between {0} and {1}\n".format(
                st[0].stats.starttime, st[0].stats.endtime))
            f.write("{0}\n".format(e))
    Logger.info("Finished detection")

    return party

def day_process(tribe, st, date, retries, skip_done: bool = False, gpu: bool = False):
    outfile = "{0}/{1}-{2}_party.tgz".format(
        DETECTION_PATH, date, (date + 86400))
    if skip_done and os.path.isfile(outfile):
        Logger.warning(f"Out file: {outfile} exists, skipping")
        return
    party = detect(tribe=tribe, st=st, gpu=gpu)
    Logger.info("Made {0} detections".format(len(party)))
    if len(party) == 0:
        with open(outfile, "w") as f:
            f.write("No detections")
        return
    # Just check that no weirdness has happened
    for family in party:
        _dets = []
        for detection in family:
            if abs(detection.detect_val) < detection.no_chans + 1:
                _dets.append(detection)
            else:
                Logger.error(
                    "Detection made at {0}, above no_chans {1}".format(
                        detection.detect_val, detection.no_chans))
                detection.write("Dodgy_detection.csv")
        family.detections = _dets
    if len(party) == 0:
        with open(outfile, "w") as f:
            f.write("No detections")
        return
    party.decluster(1) 
    Logger.info(
        "{0} detections left after 1s decluster".format(len(party)))
    party.rethreshold(
        new_threshold=0.15, new_threshold_type='av_chan_corr')
    Logger.info(
        "{0} detections left after 0.15 average threshold".format(
            len(party)))
    # Ensure no empty families remain
    if len(party) == 0:
        Logger.info("No detections, no output")
        with open(outfile, "w") as f:
            f.write("No detections")
        return
    _f = []
    for family in party:
        if len(family) > 0:
            _f.append(family)
    party.families = _f

    # Need to hack to add in chans not used for detection
    for fam in party:
        # TODO: there was a bug here - template.st rather than fam.template.st
        template_stachans = {
                (tr.stats.station, tr.stats.channel) 
                for tr in fam.template.st}
        stachans = template_stachans.intersection(
            {(tr.stats.station, tr.stats.channel) for tr in st})
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
    Logger.info(
        "Writing party of {0} detections to {1}/{2}-{3}_party.tgz".format(
            len(party), DETECTION_PATH, date.date, (date + 86400).date))
    party.write(outfile, overwrite=True)
    repicked_cat.write("{0}/{1}-{2}_repicked_catalog.xml".format(
        DETECTION_PATH, date, (date + 86400)), format="QUAKEML")


if __name__ == "__main__":
    """
    Allow to be called on an iterative basis from the shell to allow for lack
    of python garbage collection and associated memory issues
    """
    import subprocess
    from obspy import read

    import os


    #startdate = UTCDateTime(2009, 1, 1)
    # enddate = UTCDateTime(2020, 1, 1)
    parser = argparse.ArgumentParser(
        description="Detect Kaikoura earthquakes using the Kaikoura "
                    "templates")
    parser.add_argument("-s", "--startdate", type=UTCDateTime, required=True,
                        help="Start-date in UTCDateTime parsable format")
    parser.add_argument("-e", "--enddate", type=UTCDateTime, required=True,
                        help="End-date in UTCDateTime parsable format")
    parser.add_argument("-g", "--gpu", action="store_true", 
                        help="Use FMF on the GPU for correlations")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output")

    args = parser.parse_args()
    startdate = args.startdate
    enddate = args.enddate

    ndays = (enddate - startdate) / 86400
    
    
    date = None
    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level, stream=sys.stdout,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    Logger.info("Looping through {0} days".format(ndays))

    Logger.info("Reading in tribe: {0}".format(TRIBE_FILE))
    tribe = Tribe().read(TRIBE_FILE)
    Logger.info("Read in tribe of {0} templates".format(len(tribe)))

    min_stations = 5
    retries = 30
    # Check the tribe and remove bad channels
    _templates = []
    for template in tribe:
        template.process_length = 86400.0  # Some have slightly short process-len
        for tr in template.st:
            if tr.stats.npts != EXPECTED_LENGTH:
                Logger.warning("Trace for {0} is {1} long, not {2}".format(
                    tr.id, tr.stats.npts, EXPECTED_LENGTH))
                template.st.remove(tr)
        n_stations = len(set([tr.stats.station for tr in template.st]))
        if n_stations < min_stations:
            Logger.warning(
                "Template {0} has only {1} station, not using it".format(
                    template.name, n_stations))
        else:
            _templates.append(template)
    tribe.templates = _templates
    Logger.info("Will run {0} templates".format(len(tribe)))

    # To cope with moveout affects we need to overlap days.
    overlap = max(max(
        tr.stats.endtime - min(tr.stats.starttime for tr in template.st) 
        for tr in template.st) for template in tribe)
    overlap += 20

    date_step = 86400 - overlap
    # Use external looping for this long-running process
    date = date or startdate
    
    # Get the first stream
    template_params = TemplateParams.from_template(tribe[0])
    template_params.seed_ids = list(
        {tr.id for template in tribe for tr in template.st})

    template_params_filename = f"temp_waveforms/{startdate}_template_params.json"
    template_params.write(template_params_filename)

    # Open subprocess downloading the day of data
    st_filename_format = "temp_waveforms/{startdate}-{enddate}.ms"
    st_filename = st_filename_format.format(startdate=date, enddate=date + 86400)
    if os.path.isfile(st_filename):
        os.remove(st_filename)
    downloader_process = subprocess.Popen(
        ["python", "get_data.py", f"-s={date}", f"-l=86400", 
         f"-p={template_params_filename}", f"-o={st_filename}"])
    while date < enddate:
        # Get the days stream from the future
        ret_val = downloader_process.wait()  # Wait for process to finish
        if ret_val:
            raise IOError("Some error downloading data")
        st = read(st_filename).merge()
        os.remove(st_filename)  # cleanup
        next_date = date + date_step
        if next_date < enddate:
            st_filename = st_filename_format.format(
                startdate=next_date, enddate=next_date + 86400)
            if os.path.isfile(st_filename):
                os.remove(st_filename)
            # Submit the next days stream
            downloader_process = subprocess.Popen(
                ["python", "get_data.py", f"-s={next_date}", f"-l=86400", 
                 f"-p={template_params_filename}", f"-o={st_filename}"]) 
        # Submit job to main threads
        day_process(tribe=tribe, st=st, date=date, retries=3, skip_done=True, gpu=args.gpu)
        # Get the next days stream
        date = next_date

