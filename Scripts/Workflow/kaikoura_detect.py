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
import boto3

from typing import List
from tempfile import NamedTemporaryFile
from multiprocessing import cpu_count

from obspy import UTCDateTime, Catalog, Stream, read
from obspy.clients.fdsn import Client

sys.path.insert(0, "/nesi/project/nesi02337/EQcorrscan")
from eqcorrscan import Tribe

WORK_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIBE_FILE = "{0}/Templates/STREWN_merged_catalog_4s_1.5Hz-12Hz_2019.tgz".format(WORK_PATH)
DETECTION_PATH = "{0}/Detections".format(WORK_PATH)
EXPECTED_LENGTH = 4 * 30
TOTAL_CORES = min(cpu_count(), 16)
CORES_OUTER = 2
CORES_INNER = TOTAL_CORES // CORES_OUTER
GROUP_SIZE = 800
S3 = boto3.client('s3')
GEONET_BUCKET = "geonet-data"
GEONET_FORMATTER = (
    "miniseed/{year:04d}/{year:04d}.{julday:03d}/"
    "{station}.{network}/{year:04d}.{julday:03d}.{station}."
    "{location}-{channel}.{network}.D")

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)


class AWSClient(object):
    """ Basic get_waveforms and get_waveforms_bulk for GeoNet AWS bucket. """
    def __init__(
        self, 
        bucket_name: str = GEONET_BUCKET,
        bucket_formatter: str = GEONET_FORMATTER):
        self.bucket_name = bucket_name
        self.bucket_formatter = bucket_formatter

    def __repr__(self):
        return "AWSClient(bucket_name={0}, bucket_formatter={1})".format(
            self.bucket_name, self.bucket_formatter)
    
    def get_waveforms(
        self,
        network: str, 
        station: str, 
        location: str, 
        channel: str, 
        starttime: UTCDateTime, 
        endtime: UTCDateTime, 
        **kwargs
    ) -> Stream:
        if len(kwargs):
            Logger.warning("Ignoring kwargs, not implemented here")
        date = UTCDateTime(starttime.date) - 86400 
        # GeoNet bucket data don't start when they should :(
        enddate = UTCDateTime(endtime.date)
        temp_file = NamedTemporaryFile()
        st = Stream()
        while date <= enddate:
            filename = self.bucket_formatter.format(
                        year=date.year, julday=date.julday, network=network,
                        station=station, location=location, channel=channel)
            Logger.debug("Downloading: {0}".format(filename))
            try:
                S3.download_file(self.bucket_name, filename, temp_file.name)
            except Exception as e:
                Logger.error(
                    "No data for {network}.{station}.{location}.{channel} "
                    "between {date} and {enddate}".format(
                        network=network, station=station, location=location,
                        channel=channel, date=date, enddate=date + 86400))
                Logger.error(e)
                date += 86400
                continue
            st += read(temp_file.name)
            date += 86400
        st.merge().trim(starttime, endtime)
        return st.split()

    def get_waveforms_bulk(self, bulk: List, **kwargs) -> Stream:
        st = Stream()
        for query in bulk:
            st += self.get_waveforms(*query)
        return st


def detect(tribe, startdate, enddate):
    Logger.info("Running detections between {0} and {1}".format(
        startdate, enddate))
    try:
        party, st = tribe.client_detect(
            client=Client("GEONET"), starttime=startdate, endtime=enddate,
            threshold=10.0, threshold_type="MAD", trig_int=4, plot=False,
            daylong=True, parallel_process=False, xcorr_func="fftw",
            concurrency="concurrent", cores=CORES_INNER, ignore_length=False,
            save_progress=False, process_cores=CORES_INNER, 
            cores_outer=CORES_OUTER, return_stream=True, group_size=GROUP_SIZE)
    except Exception as e:
        Logger.error(e)
        traceback.print_exc()
        with open("Failed_days.txt", "a") as f:
            f.write("Failed to detect between {0} and {1}\n".format(startdate, enddate))
            f.write("{0}\n".format(e))
    Logger.info("Finished detection")
    return party, st


def day_process(tribe, date, retries):
    party, st = detect(tribe=tribe, startdate=date, enddate=date + 86400)
    Logger.info("Made {0} detections".format(len(party)))
    if len(party) == 0:
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
        return
    _f = []
    for family in party:
        if len(family) > 0:
            _f.append(family)
    party.families = _f
    for fam in party:
        for det in fam:
            det._calculate_event(template=fam.template)
    party.lag_calc(
        stream=st, pre_processed=False, shift_len=0.5, min_cc=0.4, 
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
    party.write("{0}/{1}-{2}_party".format(
        DETECTION_PATH, date.date, (date + 86400).date))
    repicked_cat.write("{0}/{1}-{2}_repicked_catalog.xml".format(
        DETECTION_PATH, date.date, (date + 86400).date), format="QUAKEML")


if __name__ == "__main__":
    """
    Allow to be called on an iterative basis from the shell to allow for lack
    of python garbage collection and associated memory issues
    """
    #startdate = UTCDateTime(2013, 1, 1)
    startdate = UTCDateTime(2016, 11, 3)
    # enddate = UTCDateTime(2019, 1, 1)
    enddate = UTCDateTime(2016, 11, 11)
    ndays = (enddate - startdate) / 86400
    parser = argparse.ArgumentParser(description="Detect Kaikoura earthquakes")
    parser.add_argument(
        '-d', '--day', type=int,
        help="The day index, see Python source for the date range")
    parser.add_argument(
        '-s', '--step_size', type=int,
        help="Number of days for this instance")
    args = vars(parser.parse_args())
    try:
        step_size = args['step_size']
    except KeyError:
        step_size = 1
    try:
        if args['day'] > ndays:
            raise IndexError("{0} is outside the range of {1}".format(
                args['day'], ndays))
        date = startdate + (86400 * args['day'])
        Logger.info("Working on {0}".format(date))
    except (KeyError, TypeError):
        Logger.info("Looping through {0} days".format(ndays))
        date = None

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

    # Use external looping for this long-running process
    if date is None:
        date = startdate
        while date < enddate:
            day_process(tribe=tribe, date=date, retries=retries)
            date += 86400
    else:
        for add_day in range(step_size):
            date += 86400 * add_day
            day_process(tribe=tribe, date=date, retries=retries)
