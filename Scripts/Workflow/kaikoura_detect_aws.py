"""
Run kaikoura_detect on AWS.

"""

import boto3
import logging
import sys

from obspy import UTCDateTime

Logger = logging.getLogger(__name__)
S3 = boto3.client('s3')
BUCKET_NAME = "matched-filter-data"
BUCKET_PATH = "Kaikoura"

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")


def get_from_bucket():
    """Get templates and detection script from bucket. """
    import os

    for folder in ["Templates", "Scripts", "Detections"]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    required_files = [
        "Templates/STREWN_merged_catalog_4s_1.5Hz-12Hz_2019.tgz",
        "Scripts/kaikoura_detect.py",]

    for required_file in required_files:
        S3.download_file(
            BUCKET_NAME, "{0}/{1}".format(BUCKET_PATH, required_file),
            required_file)
        Logger.info("Downloaded {0}".format(required_file))
    return


def put_detections_in_bucket(sent_list: list = None) -> list:
    """ Put all the detections into the bucket. """
    import glob

    sent_list = sent_list or []
    for detect_file in glob.glob("Detections/*"):
        if detect_file in sent_list:
            continue
        S3.upload_file(detect_file, BUCKET_NAME,
                       "{0}/{1}".format(BUCKET_PATH, detect_file))
        Logger.info("Uploaded {0}".format(detect_file))
        sent_list.append(detect_file)
    return sent_list


def run_detections(startdate: UTCDateTime=None, step_size: int=None, days: list=None):
    from eqcorrscan import Tribe
    import sys
    sys.path.insert(0, "Scripts")
    from kaikoura_detect import TRIBE_FILE, EXPECTED_LENGTH, day_process

    days = days or [startdate + (i * 86400) for i in range(step_size)]
    Logger.info("Will run for {0}".format(days))
    Logger.info("Reading in tribe: {0}".format(TRIBE_FILE))
    tribe = Tribe().read(TRIBE_FILE)
    Logger.info("Read in tribe of {0} templates".format(len(tribe)))

    min_stations = 5
    retries = 0
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

    date = startdate
    # Use external looping for this long-running process
    sent_list = []
    
    for date in days:
        day_process(tribe=tribe, date=date, retries=retries)
        sent_list = put_detections_in_bucket(sent_list=sent_list)


if __name__ == "__main__":
    import argparse

    startdate = UTCDateTime(2009, 1, 1)
    enddate = UTCDateTime(2010, 1, 1)
    ndays = (enddate - startdate) / 86400
    #with open("Missed_days.txt", "r") as f:
    #    days = [UTCDateTime(l.rstrip()) for l in f]
    #ndays = len(days)

    parser = argparse.ArgumentParser(description="Detect Kaikoura earthquakes")
    parser.add_argument(
        '-d', '--day', type=int,
        help="The day index, see Python source for the date range")
    parser.add_argument(
        '-s', '--step_size', type=int,
        help="Number of days for this instance")
    args = parser.parse_args()
    try:
        step_size = args.step_size
    except KeyError:
        step_size = 1
    try:
        if args.day > ndays:
            raise IndexError("{0} is outside the range of {1}".format(
                args.day, ndays))
        date = startdate + (86400 * args.day)
        Logger.info("Working on {0}".format(date))
    except (KeyError, TypeError):
        Logger.info("Looping through {0} days".format(ndays))
        date = None

    get_from_bucket()

    # Run detections
    run_detections(startdate=date, step_size=step_size)
    # run_detections(days=days[args.day: args.day + step_size])

