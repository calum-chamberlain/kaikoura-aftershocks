import os
import logging

from obspy import read, Catalog

from eqcorrscan import Party


TOTAL_CORES = 4

Logger = logging.getLogger(__name__)


def lag_calc(party_file: str, st_file: str):
    st = read(st_file).merge()
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

    import argparse

    parser = argparse.ArgumentParser(
        description="Re-do lag-cal for a specific party")
    parser.add_argument("-s", "--st_file", type=str, required=True,
                        help="Stream filename")
    parser.add_argument("-p", "--party_file", type=str, required=True,
                        help="Party filename")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    lag_calc(party_file=args.party_file, st_file=args.st_file)