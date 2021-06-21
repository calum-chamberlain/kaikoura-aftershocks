"""
Script to iteratively remove picks and locate using Hypocentre until some
quality criteria are met.

:author: Calum Chamberlain
:date: 15/11/2019
"""
import logging
from typing import List

from obspy import read_events, Catalog, read_inventory
from obspy.core.event import Event
from obspy.core.inventory import Inventory, Station


Logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")


class Velocity(object):
    """
    Simple class holding velocity layer.

    Parameters
    ----------
    top:
        Top of layer in km
    vp:
        P-wave velocity oif layer in km/s
    moho:
        Mark whether this layer is the moho.
    """
    def __init__(self, top: float, vp: float, moho: bool = False):
        self.top = top
        self.vp = vp
        self.moho = moho

    def __repr__(self):
        return "Velocity(top={0}, vp={1}, moho={2})".format(
            self.top, self.vp, self.moho)

    def __str__(self):
        if self.moho:
            return "{0:7.3f}   {1:7.3f}    N     ".format(self.vp, self.top)
        return "{0:7.3f}   {1:7.3f}".format(self.vp, self.top)


############################## GLOBALS #########################################

VELOCITIES = [  # Kaikoura rom Table 2 of Okada et al., 2019, Tectonophysics
    Velocity(-1.0, 3.464),
    Velocity(3.0, 4.380),
    Velocity(8.0, 5.181),
    Velocity(15.0, 5.638),
    Velocity(23.0, 6.095),
    Velocity(30.0, 6.603, moho=True),  # Inferred from Figure 1
    Velocity(38.0, 7.309),
    Velocity(48.0, 7.876),
    Velocity(65.0, 7.942),
    Velocity(85.0, 7.965),
    Velocity(105.0, 7.851),
    Velocity(130.0, 8.652),
]
VPVS = 1.73
MAX_RMS = 1.5
MIN_STATIONS = 5

################################# END ##########################################


def seisan_hyp(
    event: Event,
    inventory: Inventory,
    velocities: List[Velocity],
    vpvs: float,
    remodel: bool = True,
    clean: bool = True
) -> Event:
    """
    Use SEISAN's Hypocentre program to locate an event.
    """
    import warnings
    import subprocess
    from obspy.core.event import Origin
    from obspy.io.nordic.core import write_select, read_nordic

    # Write STATION0.HYP file
    _write_station0(inventory, velocities, vpvs)

    if remodel:
        subprocess.call(['remodl'])
        subprocess.call(['setbrn'])

    event_out = event.copy()
    try:
        old_origin = event.preferred_origin() or event.origins[0]
        origin = Origin(time=old_origin.time)
    except IndexError:
        origin = Origin(
            time=sorted(event.picks, key=lambda p: p.time)[0].time)
    event_out.origins = [origin]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        event_out.write(format="NORDIC", filename="to_be_located")
    subprocess.call(['hyp', "to_be_located"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            event_back = read_nordic("hyp.out")
        except Exception as e:
            Logger.error(e)
            return None
    # We lose some info in the round-trip to nordic
    event_out.origins[0] = event_back[0].origins[0]
    event_out.magnitudes = event_back[0].magnitudes
    event_out.picks = event_back[0].picks
    if clean:
        _cleanup()
    return event_out


def _cleanup():
    import os

    # Clean up
    files_to_remove = [
        "hyp.out", "to_be_located", "remodl.tbl", "remodl1.lis", "remodl2.lis",
        "print.out", "gmap.cur.kml", "hypmag.out", "hypsum.out", "remodl.hed",
        "IASP91_linux.HED", "IASP91_linux.TBL", "setbrn1.lis", "setbrn2.lis",
        "setbrn3.lis", "STATION0.HYP", "focmec.dat", "focmec.inp", "fort.17",
        "fps.out", "hash_seisan.out", "pspolar.inp", "scratch1.out",
        "scratch2.out", "scratch3.out"]
    for f in files_to_remove:
        if os.path.isfile(f):
            os.remove(f)


def _stationtoseisan(station: Station) -> str:
    """
    Convert obspy inventory to string formatted for Seisan STATION0.HYP file.

    :param station: Inventory containing a single station.

    .. note::
        Only works to the low-precision level at the moment (see seisan
        manual for explanation).
    """

    if station.latitude < 0:
        lat_str = 'S'
    else:
        lat_str = 'N'
    if station.longitude < 0:  # Stored in =/- 180, not 0-360
        lon_str = 'W'
    else:
        lon_str = 'E'
    if len(station.code) > 4:
        sta_str = station.code[0:4]
    else:
        sta_str = station.code.ljust(4)
    if len(station.channels) > 0:
        depth = station.channels[0].depth
    else:
        msg = 'No depth found in station.channels, have you set the level ' +\
              'of stationXML download to channel if using obspy.get_stations?'
        raise IOError(msg)
    elev = str(int(round(station.elevation - depth))).rjust(4)
    # lat and long are written in STATION0.HYP in deg,decimal mins
    lat = abs(station.latitude)
    lat_degree = int(lat)
    lat_decimal_minute = (lat - lat_degree) * 60
    lon = abs(station.longitude)
    lon_degree = int(lon)
    lon_decimal_minute = (lon - lon_degree) * 60
    lat = ''.join([str(int(abs(lat_degree))),
                   '{0:.2f}'.format(lat_decimal_minute).rjust(5)])
    lon = ''.join([str(int(abs(lon_degree))),
                   '{0:.2f}'.format(lon_decimal_minute).rjust(5)])
    station_str = ''.join(['  ', sta_str, lat, lat_str, lon, lon_str, elev])
    return station_str


def _write_station0(
    inventory: Inventory,
    velocities: List[Velocity],
    vpvs: float
):
    out = (
        "RESET TEST(02)=500.0\nRESET TEST(07)=-3.0\nRESET TEST(08)=2.6\n"
        "RESET TEST(09)=0.001\nRESET TEST(11)=99.0\nRESET TEST(13)=5.0\n"
        "RESET TEST(34)=1.5\nRESET TEST(35)=2.5\nRESET TEST(36)=0.0\n"
        "RESET TEST(41)=20000.0\nRESET TEST(43)=5.0\nRESET TEST(51)=3.6\n"
        "RESET TEST(50)=1.0\nRESET TEST(56)= 1.0\nRESET TEST(58)= 99990.0\n"
        "RESET TEST(40)=0.0\nRESET TEST(60)=0.0\nRESET TEST(71)=1.0\n"
        "RESET TEST(75)=1.0\nRESET TEST(76)=0.910\nRESET TEST(77)=0.00087\n"
        "RESET TEST(78)=-1.67\nRESET TEST(79)=1.0\nRESET TEST(80)=3.0\n"
        "RESET TEST(81)=1.0\nRESET TEST(82)=1.0\nRESET TEST(83)=1.0\n"
        "RESET TEST(88)=1.0\nRESET TEST(85)=0.1\nRESET TEST(91)=0.1\n")
    for network in inventory:
        for station in network:
            out += "\n" + _stationtoseisan(station)
    out += "\n\n"
    # Add velocity model
    for layer in velocities:
        out += "{0}\n".format(layer)
    out += "\n15.0 1100.2200. {0:.2f} \nTES\n".format(vpvs)
    with open("STATION0.HYP", "w") as f:
        f.write(out)
    return


def iterate_over_event(
    event: Event,
    inventory: Inventory,
    velocities: List[Velocity],
    vpvs: float,
    max_rms: float = 1.0,
    min_stations: int = 5,
    clean: bool = True,
    remodel: bool = True,
) -> Event:
    n_stations = len({p.waveform_id.station_code for p in event.picks})
    if n_stations < min_stations:
        Logger.info("{0} stations picked, returning None".format(n_stations))
        return None
    event_to_be_located = event.copy()  # Keep the original event safe
    while n_stations >= min_stations:
        event_located = seisan_hyp(
            event=event_to_be_located, inventory=inventory,
            velocities=velocities, vpvs=vpvs, clean=clean, remodel=remodel)
        if event_located is None:
            Logger.error("Issue locating event, returning None")
            return None
        if len(event_located.origins) == 0 or event_located.origins[0].latitude == None:
            Logger.warning("Issue locating event, returning None")
            return None
        if event_located.origins[0].quality.standard_error <= max_rms:
            Logger.info("RMS below max_RMS, good enough!")
            event_to_be_located.origins.append(event_located.origins[0])
            return event_to_be_located
        # Remove least well fit pick and go again
        worst_arrival = event_located.origins[0].arrivals[0]
        for arr in event_located.origins[0].arrivals[1:]:
            try:
                if arr.time_residual > worst_arrival.time_residual:
                    worst_arrival = arr
            except TypeError:
                Logger.error(arr)
        for pick in event_located.picks:
            if pick.resource_id == worst_arrival.pick_id:
                worst_pick = pick
                Logger.info("Removing pick at {0} on {1}".format(
                    pick.time, pick.waveform_id.get_seed_string()))
                break
        _picks = []
        for pick in event_to_be_located.picks:
            if pick.waveform_id.station_code == worst_pick.waveform_id.station_code and abs(pick.time - worst_pick.time) < 0.01:
                continue
            _picks.append(pick)
        assert len(_picks) < len(event_to_be_located.picks)
        event_to_be_located.picks = _picks
        n_stations = len({p.waveform_id.station_code
                          for p in event_located.picks})

    Logger.info("{0} stations picked, returning None".format(n_stations))
    return None


def locate_and_remove_picks(
    catalog: Catalog,
    inventory: Inventory,
    velocities: List[Velocity],
    vpvs: float,
    max_rms: float = 1.0,
    min_stations: int = 5,
) -> Catalog:
    cat_out = Catalog()
    clean, remodel = False, True
    for i, event in enumerate(catalog):
        event_back = iterate_over_event(
            event=event, inventory=inventory, velocities=velocities,
            vpvs=vpvs, max_rms=max_rms, min_stations=min_stations,
            clean=clean, remodel=remodel)
        remodel = False
        if i == len(catalog) - 1:
            clean = True
        if event_back:
            cat_out.events.append(event_back)
    return cat_out


if __name__ == "__main__":
    """
    Kaikoura input:
    python locate_remove_picks.py -c ../../Detections_2021/declustered_2sminimum_5_stations_2009-01-02-2020-01-01_detections.xml\
        -i Kaikoura_stations.xml -o ../../Detections_2021/bad_picks_removed_2009-01-01-2020-01-01_repicked_catalog.xml
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Locate and QC picks for Kaikoura detections")
    parser.add_argument(
        '-c', '--catalog', type=str, required=True,
        help="The file containing the input catalog")
    parser.add_argument(
        '-i', '--inventory', type=str, required=True,
        help="The file containing the station inventory")
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help="The file to save the output catalog to")

    args = parser.parse_args()

    cat = read_events(args.catalog)
    inv = read_inventory(args.inventory)

    Logger.info("Read in {0} events for tuning".format(len(cat)))
    cat_back = locate_and_remove_picks(
        catalog=cat, inventory=inv, velocities=VELOCITIES, vpvs=VPVS,
        max_rms=MAX_RMS, min_stations=MIN_STATIONS)

    Logger.info("After tuning, {0} events remain".format(len(cat_back)))
    cat_back.write(args.output, format='QUAKEML')
