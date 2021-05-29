"""
Script for detecting earthquakes using Kaikoura aftershock templates.

Written by: Calum J Chamberlain
Date:       22/06/2018
"""
import argparse
import os
import sys
import logging
import numpy as np
import json

from typing import List
from tempfile import NamedTemporaryFile
from multiprocessing import cpu_count

from obspy import UTCDateTime, Catalog, Stream, read
from obspy.clients.fdsn import Client

from eqcorrscan.core.match_filter.matched_filter import _group_process

Logger = logging.getLogger(__name__)


import numpy as np


GEONET_BUCKET = "geonet-data"
GEONET_FORMATTER = (
    "miniseed/{year:04d}/{year:04d}.{julday:03d}/"
    "{station}.{network}/{year:04d}.{julday:03d}.{station}."
    "{location}-{channel}.{network}.D")


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
        import boto3
        S3 = boto3.client('s3')
        
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


class TemplateParams:
    def __init__(self, filt_order, highcut, lowcut, samp_rate, process_length, seed_ids):
        self.filt_order = filt_order
        self.highcut = highcut
        self.lowcut = lowcut
        self.samp_rate = samp_rate
        self.process_length = process_length
        self.seed_ids = list(seed_ids)

    def __repr__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        params = ', '.join(params)
        return f"TemplateParams({params})"

    @classmethod
    def from_template(cls, template):
        tp = cls(
            filt_order=template.filt_order,
            highcut=template.highcut,
            lowcut=template.lowcut,
            samp_rate=template.samp_rate,
            process_length=template.process_length,
            seed_ids={tr.id for tr in template.st})
        return tp

    def write(self, filename):
        with open(filename, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def read(cls, filename):
        with open(filename, "r") as f:
            params = json.load(f)
        return cls(**params)


def get_waveforms_from_clients(
    starttime, 
    endtime, 
    template_params,
    return_stream=True
):
    from concurrent.futures import ThreadPoolExecutor

    st = Stream()
    iris_client = Client("IRIS")
    geonet_client = Client("GEONET")

    seed_ids = set(template_params.seed_ids)

    futures = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for seed_id in seed_ids:
            net, sta, loc, chan = seed_id.split('.')
            kwargs = dict(
                network=net, station=sta, location=loc, channel=chan,
                starttime=starttime, endtime=endtime)
            if net == "NZ":
                client = geonet_client
            else:
                client = iris_client
            futures.append(executor.submit(_get_waveform, client, kwargs))
        for future in futures:
            st += future.result()

    processed_streams = _group_process(
        [template_params], parallel=False, cores=1, stream=st.merge(),
        daylong=False, ignore_length=False, ignore_bad_data=True,
        overlap=0.0)  # Overlap handled elsewhere
    st = processed_streams[0]

    if not return_stream:
        filename = f"temp_waveforms/{starttime}_{len(st)}.ms"
        st.split().write(filename, format="MSEED")
        return filename

    return st.merge()


def _get_waveform(client, params):
    Logger.debug(f"Downloading for {params}")
    try:
        st = client.get_waveforms(**params).trim(
            starttime=params["starttime"], endtime=params["endtime"])
    except Exception as e:
        Logger.warning(f"Could not download for {params} due to {e}")
        return Stream()
    Logger.debug(f"Acquired stream for {params}")
    return st


if __name__ == "__main__":
    """
    Allow to be called on an iterative basis from the shell to allow for lack
    of python garbage collection and associated memory issues
    """
    parser = argparse.ArgumentParser(
        description="Download and process data templates")
    parser.add_argument("-s", "--startdate", type=UTCDateTime, required=True,
                        help="Start-date in UTCDateTime parsable format")
    parser.add_argument("-l", "--length", type=float, default=86400.0,
                        help="Length of data to download and process in seconds")
    parser.add_argument("-p", "--template-params", type=str, required=True,
                        help="Template parameter file")
    parser.add_argument("-o", "--outfile", type=str, required=True,
                        help="File to write to - must not exist")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output")

    args = parser.parse_args()
    startdate = args.startdate
    enddate = startdate + args.length

    if os.path.isfile(args.outfile):
        raise NotImplementedError(f"{args.outfile} exists")

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level, stream=sys.stdout,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    # Get the stream
    template_params = TemplateParams.read(args.template_params)

    st = get_waveforms_from_clients(
        starttime=startdate, endtime=enddate, 
        template_params=template_params, return_stream=True)
    st.write(args.outfile, format="MSEED")
    Logger.info(f"Written stream of {len(st)} traces to {args.outfile}")

