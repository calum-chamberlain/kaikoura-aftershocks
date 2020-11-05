"""
Script to test template parameters for a subset of templates.

This script will generate a subset of templates (the first 20 in the catalog)
and run them through two days of data, one day pre-Kaikoura, and one day
post-Kaikoura to.

Templates will be generated for a range of parameters and detections can then
be compared between parameter sets.

"""

import os
import shutil
import glob
import logging

from collections import Counter

from obspy import read_events, Inventory, UTCDateTime
from obspy.core.event import WaveformStreamID
from obspy.clients.fdsn import Client
from eqcorrscan import Tribe, Party
from eqcorrscan.utils.pre_processing import dayproc
from eqcorrscan.utils.catalog_utils import filter_picks
from eqcorrscan.utils.plotting import detection_multiplot

from generate_templates import generate_templates

WORKING_PATH = "/Volumes/GeoPhysics_09/users-data/chambeca/Kaik_afterslip"
TEMPLATE_PATH = ("{0}/Templates".format(WORKING_PATH))
CATALOG_FILE = "{0}/SIMUL_locations.xml".format(TEMPLATE_PATH)
BAD_STATIONS = ['RDCS', 'RCBS', 'MRZ', 'POWZ', 'WPHZ', 'PRWZ', 'QRZ']
TEST_PARAMS = {
    "2s_wide": {"lowcut": 4.0, "highcut": 15.0, "length": 2.0},
    "4s_wide": {"lowcut": 4.0, "highcut": 15.0, "length": 4.0},
    "6s_wide": {"lowcut": 4.0, "highcut": 15.0, "length": 6.0},
    "2s_low": {"lowcut": 4.0, "highcut": 10.0, "length": 2.0},
    "4s_low": {"lowcut": 4.0, "highcut": 10.0, "length": 4.0},
    "6s_low": {"lowcut": 4.0, "highcut": 10.0, "length": 6.0},
    "2s_high": {"lowcut": 8.0, "highcut": 15.0, "length": 2.0},
    "4s_high": {"lowcut": 8.0, "highcut": 15.0, "length": 4.0},
    "6s_high": {"lowcut": 8.0, "highcut": 15.0, "length": 6.0},
}
TEST_DAYS = [UTCDateTime(2016, 11, 10), UTCDateTime(2016, 11, 20)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger("generate_templates.__main__")


if __name__ == "__main__":
    plot = False
    Logger.info("Reading catalog from {0}".format(CATALOG_FILE))
    cat = read_events(CATALOG_FILE)
    cat.events.sort(key=lambda e: e.origins[0].time)

    Logger.info("Read in {0} events".format(len(cat)))
    cat = cat[0:20] # Used a short catalog for testing
    pre_plots = glob.glob("*.png")
    for key, test_params in TEST_PARAMS.items():
        outfile = "{0}_{1}".format(os.path.splitext(CATALOG_FILE)[0], key)
        tribe = generate_templates(cat=cat, plot=plot, **test_params)
        Logger.info("Generated tribe of {0} templates".format(len(tribe)))
        tribe.write(outfile)
        Logger.info("Written tribe to {0}.tgz".format(outfile))
        # Move plots
        if plot:
            if not os.path.isdir("{0}/Plots/{1}".format(WORKING_PATH, key)):
                os.makedirs("{0}/Plots/{1}".format(WORKING_PATH, key))
            plots = glob.glob("*.png")
            for _plot in plots:
                if _plot not in pre_plots:
                    shutil.move(_plot, "{0}/Plots/{1}/{2}".format(
                        WORKING_PATH, key, _plot))
    Logger.info('Finished making tribes')

    for day in TEST_DAYS:
        startdate = day
        enddate = startdate + 86400
        for key in TEST_PARAMS.keys():
            tribe = Tribe().read(
                "{0}_{1}.tgz".format(os.path.splitext(CATALOG_FILE)[0], key))
            party = tribe.client_detect(
                client=Client("GEONET"), starttime=startdate, endtime=enddate,
                threshold=8.0, threshold_type="MAD", trig_int=5, plotvar=False,
                daylong=True, parallel_process=True, xcorr_func="fftw",
                concurrency="concurrent", cores=8, ignore_length=False,
                save_progress=False, process_cores=2, cores_outer=1,
                return_stream=False)
            party.write("{0}/Detections/Test_days/{1}-{2}_party.tgz".format(
                WORKING_PATH, day.strftime("%Y-%m-%d"), key))
    
    Logger.info("Finished detections")

    # Plot all detections
    parties = {}
    for day in TEST_DAYS:
        for key in TEST_PARAMS.keys():
            party = Party().read(
                "{0}/Detections/Test_days/{1}-{2}_party.tgz.tgz".format(
                    WORKING_PATH, day.strftime("%Y-%m-%d"), key))
            if key in parties.keys():
                parties[key].update({day.strftime("%Y-%m-%d"): party})
            else:
                parties.update({key: {day.strftime("%Y-%m-%d"): party}})

    streams = {}
    client = Client("GEONET")
    for day in TEST_DAYS:
        bulk = []
        for key in parties.keys():
            for family in parties[key][day.strftime("%Y-%m-%d")]:
                for tr in family.template.st:
                    _bulk = (tr.stats.network, tr.stats.station,
                             tr.stats.location, tr.stats.channel,
                             day, day + 86400)
                    if _bulk not in bulk:
                        bulk.append(_bulk)
        Logger.info("Downloading data for {0}".format(day))
        st = client.get_waveforms_bulk(bulk)
        st.trim(day, day + 86400).merge()
        st = dayproc(
            st=st, lowcut=4, highcut=15, filt_order=4, samp_rate=50,
            starttime=day, num_cores=2)
        streams.update({day.strftime("%Y-%m-%d"): st})      

    for key in parties.keys():
        Logger.info("Plotting for {0}".format(key))
        for day_key in parties[key]:
            fig = parties[key][day_key].plot(
                plot_grouped=True, show=False, return_figure=True)
            fig.savefig("{0}/Plots/Test_days/{1}_{2}_detections.png".format(
                WORKING_PATH, day_key, key))
            fig.clear()
            del(fig)
            for family in parties[key][day_key]:
                Logger.info("Family: {0}".format(family.template.name))
                for detection in family.detections:
                    st = streams[day_key].slice(detection.detect_time - 5,
                                                detection.detect_time + 80)
                    if len(st) == 0:
                        Logger.warning(
                            "Skipped plot of detection at {0}".format(
                                detection.detect_time))
                        continue
                    detection_multiplot(
                        stream=st, template=family.template.st,
                        times=[detection.detect_time], save=True,
                        savefile="{0}/Plots/Test_days/detection_{1}_{2}.png".format(
                            WORKING_PATH, 
                            detection.detect_time.strftime("%Y-%m-%d-%H-%M-%S"),
                            key), title="Detected by: {0}".format(key))

    Logger.info("FINISHED")