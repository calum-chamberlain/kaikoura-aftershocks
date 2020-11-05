"""
Get GPS daily solutions from GeoNet.

"""


import requests
import numpy as np

import datetime as dt
from typing import Iterable, List
from collections import Counter

import matplotlib.pyplot as plt

from obspy import Trace, Stream
from obspy.signal.rotate import rotate_ne_rt


STATIONS = {"MRBL", "HANM", "KAIK", "CLRR", "LOOK", "GLOK", "CMBL", "SEDD"}

GEONET_FITS = "http://fits.geonet.org.nz/observation"


class GPSData():
    def __init__(self, reciever: str, component: str, times: np.ndarray, 
                 observations: np.ndarray, errors: np.ndarray):
        assert times.shape == observations.shape, "Times is not shaped like observations"
        assert observations.shape == errors.shape, "Observations is not shaped like errors"
        self.reciever = reciever
        self.component = component
        # Sort everything
        sort_mask = np.argsort(times)
        self.times = times[sort_mask]
        self.observations = observations[sort_mask]
        self.errors = errors[sort_mask]

    def __repr__(self):
        return (
            f"GPSData(reciever={self.reciever}, component={self.component},"
            f" <starttime: {self.times[0]}> <endtime: {self.times[-1]}>)")

    @property
    def trace(self):
        """ Make an ObsPy Trace from the data. """
        # Work out most common sampling-rate
        sample_intervals = [
            t.total_seconds() for t in self.times[1:] - self.times[0:-1]]
        sample_interval = Counter(sample_intervals).most_common(1)[0][0]
        slice_points = np.where(np.array(sample_intervals) > sample_interval)[0]
        header = dict(
            delta=sample_interval, station=self.reciever, 
            channel=self.component)
        stream = Stream()
        chunk_start = 0
        for chunk_end in slice_points:
            header.update({"starttime": self.times[chunk_start]})
            trace = Trace(
                header=header, data=self.observations[chunk_start:chunk_end])
            chunk_start = chunk_end
            stream += trace
        return stream.merge()[0]

    def trim(self, starttime: dt.datetime = None, endtime: dt.datetime = None):
        starttime = starttime or self.times[0]
        endtime = endtime or self.times[-1]
        mask = np.where(
            np.logical_and(self.times >= starttime, self.times <= endtime))
        self.times = self.times[mask]
        self.observations = self.observations[mask]
        self.errors = self.errors[mask]
        return self

    def zero_start(self, index: int = 0):
        self.observations -= self.observations[index]
        return self

    def detrend(
        self, 
        trend_starttime: dt.datetime = None,
        trend_endtime: dt.datetime = None,
        gradient: float = None
    ):
        """
        Linear detrend using fit between trend_starttime and trend_endtime.

        Parameters
        ----------
        trend_starttime:
            Starttime to calculate gradient for detrending
        trend_endtime:
            Endtime to calculate gradient for detrending
        gradient:
            Gradient to remove - will use this if given, otherwise will 
            calculate gradient between trend_starttime and trend_endtime.
            Must be in observations-units per second.
        """
        if gradient:
            return self._detrend(gradient)
        trend_starttime = trend_starttime or self.times[0]
        trend_endtime = trend_endtime or self.times[-1]
        mask = np.where(
            np.logical_and(self.times >= trend_starttime, 
                           self.times <= trend_endtime))
        if len(mask) == 0:
            print(f"Could not detrend, no data between {trend_starttime} and "
                  f"{trend_endtime}")
            return self
        x, y = self.times[mask], self.observations[mask]
        x = np.array([(t - x[0]).total_seconds() for t in x])
        try:
            gradient, intercept = np.polyfit(x, y, 1)
        except TypeError as e:
            print(e)
            return self
        return self._detrend(gradient)


    def _detrend(self, gradient: float):
        x = np.array([(t - self.times[0]).total_seconds() 
                        for t in self.times])
        self.observations -= x * gradient
        return self

    def fft(
        self, 
        strict_length: bool = False, 
        plot: bool = False,
        period_unit: str = "day",
    ) -> np.ndarray:
        """
        Compute the fourier transform of the data.
        
        Note that data should be detrended first. 
        """
        from scipy import fftpack
        if plot:
            assert period_unit in ("hour", "day", "second")
        # Fill gaps
        data = self.trace.split().detrend().merge(
            fill_value="interpolate")[0].data
        N = len(data)
        if strict_length:
            fft_len = N
        else:
            # Find the next fast length for the FFT
            fft_len = fftpack.next_fast_len(N)
        dt = self.trace.stats.delta
        yf = fftpack.fft(data, n=fft_len)
        if not plot:
            return yf
        # Make an array of frequencies that the FFT has been computed for
        xf = np.linspace(0.0, 1.0 / (2. * dt), int(N / 2))
        # Get the positive and real component of the FFT - this is the amplitude spectra
        amplitude_spectra = np.abs(yf[:N // 2])
        # Multiply to normalise amplitude spectra to 1.
        amplitude_spectra *= 2. / N

        # Convert xf to period
        if period_unit == "second":
            xf = 1 / xf
        elif period_unit == "hour":
            xf = (1 / xf) / 3600.
        elif period_unit == "day":
            xf = (1 / xf) / 86400.

        fig, ax = plt.subplots()
        ax.loglog(xf, amplitude_spectra)
        ax.set_xlabel(f"Period ({period_unit})")
        ax.set_ylabel("Nomalized Amplitude")
        ax.autoscale(enable=True, axis='both', tight=True)
        plt.show()
        return yf



class GPSStation():
    def __init__(
        self, 
        components: List[GPSData],
    ):
        station = self._check_components(components)
        self._station = station
        self._components = components

    def __repr__(self):
        component_strings = [c.__repr__() for c in self]
        component_strings = "\n\t".join(component_strings)
        return f"GPSStation: {self._station}:\n\t{component_strings}"

    def __iter__(self):
        return list(self._components).__iter__()

    def __len__(self):
        return len(self._components)

    def __getitem__(self, val):
        return self._components[val]

    @staticmethod
    def _check_components(components):
        for _component in ("u", "n", "e", "1", "2", "r", "t"):
            count = [c for c in components if c.component == _component]
            assert len(count) <= 1, f"Multiple instances of {_component} component"
        # check that all come from the same station
        station = {c.reciever for c in components}
        assert len(station) == 1, "Only one station allowed"
        return station.pop()

    @classmethod
    def from_geonet(
        cls, 
        reciever: str, 
        components: Iterable[str] = ("u", "n", "e")
    ):
        _components = []
        for channel in components:
            parameters = {"typeID": channel[0], "siteID": reciever}
            response = requests.get(GEONET_FITS, params=parameters)
            assert response.status_code == 200, "Bad request"
            payload = response.content.decode("utf-8").split("\n")
            # payload is a csv with header
            payload = [p.split(',') for p in payload]
            # Check that this is what we expect
            assert payload[0][0] == 'date-time', "Unkown format"
            assert len(payload[0]) == 3, "Unknown format"
            times, displacements, errors = zip(*[
                (dt.datetime.strptime(p[0], '%Y-%m-%dT%H:%M:%S.%fZ'),
                float(p[1]), float(p[2])) for p in payload[1:-1]])
            _data =  GPSData(
                reciever=reciever, component=channel, times=np.array(times),
                observations=np.array(displacements), 
                errors=np.array(errors))
            _components.append(_data)
        return cls(_components)

    @property
    def stream(self) -> Stream:
        return Stream([component.trace for component in self])

    @property
    def component_ids(self) -> set:
        return {component.component for component in self}

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, components: List[GPSData]):
        station = self._check_components(components)
        if self._station:
            assert self._station == station, "Cannot change components to a different station"
        self._components = components

    def select(self, component: str):
        if component == "vertical":
            component = "u"
        elif component == "north":
            component = "n"
        elif component == "east":
            component = "e"
        return GPSStation([c for c in self if c.component == component])

    @property
    def vertical(self):
        u = self.select("u")
        if len(u) == 0:
            return None
        return u[0]

    @property
    def north(self):
        n = self.select("n")
        if len(n) == 0:
            return None
        return n[0]

    @property
    def east(self):
        e = self.select("e")
        if len(e) == 0:
            return None
        return e[0]

    @property
    def _horizontals(self):
        n, e = self.north, self.east
        assert n and e, "Requires North (n) and East (e) components"
        assert np.all(n.times == e.times), "Requires the same sampling of n and e"
        return n, e

    def zero_start(self, index: int = 0):
        for component in self:
            component.zero_start(index)
        return self

    def rotate(self, bearing: float):
        n, e = self._horizontals
        r, t = rotate_ne_rt(
            n.observations, e.observations, (bearing + 180) % 360)
        r_errors, t_errors = rotate_ne_rt(
            n.errors, e.errors, (bearing + 180) % 360)
        r = GPSData(reciever=self._station, component="r", times=n.times,
                    observations=r, errors=r_errors)
        t = GPSData(reciever=self._station, component="t", times=n.times,
                    observations=t, errors=t_errors)
        if "r" in self.component_ids:
            print("Overwriting component r")
            self._components = [
                c for c in self.components if c.component != "r"]
        if "t" in self.component_ids:
            print("Overwriting component t")
            self._components = [
                c for c in self.components if c.component != "t"]
        self.components = self.components + [r, t]
        return self

    @property
    def bearing(self) -> float:
        from math import atan, degrees

        n, e = self._horizontals
        n_disp = n.observations[-1] - n.observations[0]
        e_disp = e.observations[-1] - e.observations[0]
        if n_disp == 0 and e_disp == 0:
            return 0.0
        elif n_disp == 0 and e_disp > 0:
            return 90.0
        elif n_disp == 0 and e_disp < 0:
            return 270.0
        elif e_disp == 0 and n_disp > 0:
            return 0.0
        elif e_disp == 0 and n_disp < 0:
            return 180.0
        elif n_disp > 0 and e_disp > 0:
            return degrees(atan(e_disp / n_disp))
        elif n_disp < 0 and e_disp > 0:
            return 90.0 + degrees(atan(abs(n_disp) / e_disp))
        elif n_disp < 0 and e_disp < 0:
            return 180 + degrees(atan(abs(e_disp)/ abs(n_disp)))
        elif n_disp > 0 and e_disp < 0:
            return 270 + degrees(atan(n_disp / abs(e_disp)))
        else:
            NotImplementedError(f"N: {n_disp}, E: {e_disp}")

    def detrend(
        self, 
        trend_starttime: dt.datetime = None,
        trend_endtime: dt.datetime = None,
        gradient: float = None
    ):
        """
        Linear detrend using fit between trend_starttime and trend_endtime.

        Parameters
        ----------
        trend_starttime:
            Starttime to calculate gradient for detrending
        trend_endtime:
            Endtime to calculate gradient for detrending
        gradient:
            Gradient to remove - will use this if given, otherwise will 
            calculate gradient between trend_starttime and trend_endtime.
            Must be in observations-units per second.
        """
        for component in self:
            component.detrend(
                trend_starttime=trend_starttime, trend_endtime=trend_endtime, 
                gradient=gradient)
        return self

    def trim(self, starttime: dt.datetime = None, endtime: dt.datetime = None):
        for component in self:
            component.trim(starttime=starttime, endtime=endtime)
        return self

    def plot(self, fig: plt.Figure = None, show: bool = True) -> plt.Figure:
        if fig:
            ax = fig.gca()
        else:
            fig, ax = plt.subplots()

        for component in self:
            ax.plot(component.times, component.observations, 
                    label=component.component.upper())
            ax.fill_between(
                component.times, component.observations + component.errors,
                component.observations - component.errors, alpha=0.3)
        ax.legend()
        ax.set_title(self._station)
        if show:
            plt.show()
        return fig


def get_data(
    reciever: str, 
    components: Iterable = ("u", "n", "e"),
    starttime: dt = dt.datetime(2009, 1, 1), 
    endtime: dt = dt.datetime(2020, 1, 1),
    detrend: bool = False,
    trend_starttime: dt.datetime = dt.datetime(2015, 11, 1),
    trend_endtime: dt.datetime = dt.datetime(2016, 11, 1),
) -> dict:
    """ Get data from the GeoNet FITS service. """
    data = dict()
    for channel in components:
        parameters = {"typeID": channel[0], "siteID": reciever}
        response = requests.get(GEONET_FITS, params=parameters)
        assert response.status_code == 200, "Bad request"
        payload = response.content.decode("utf-8").split("\n")
        # payload is a csv with header
        payload = [p.split(',') for p in payload]
        # Check that this is what we expect
        assert payload[0][0] == 'date-time', "Unkown format"
        assert len(payload[0]) == 3, "Unknown format"
        times, displacements, errors = zip(*[
            (dt.strptime(p[0], '%Y-%m-%dT%H:%M:%S.%fZ'),
            float(p[1]), float(p[2])) for p in payload[1:-1]])
        _data =  GPSData(
            reciever=reciever, component=channel, times=np.array(times),
            observations=np.array(displacements), 
            errors=np.array(errors))
        if detrend:
            _data.detrend(trend_starttime=trend_starttime, 
                          trend_endtime=trend_endtime)
        _data.trim(starttime=starttime, endtime=endtime)
        data.update({channel: _data})
    return data


def main():
    plt.style.use("ggplot")
    starttime = dt.datetime(2009, 1, 1)
    endtime = dt.datetime(2020, 1, 1)
    fig, axes = plt.subplots(nrows=len(STATIONS), sharex=True)
    for station, ax in zip(STATIONS, axes):
        data = get_data(reciever=station, starttime=starttime, endtime=endtime)
        for component in ("u", "n", "e"):
            ax.plot(data[component].times, data[component].observations, 
                    label=component.upper())
        ax.set_xlim(starttime, endtime)
        ax.set_ylabel(f"{station} (mm)")

    ax.set_xlabel("Date (UTC)")
    ax.legend()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.show()
