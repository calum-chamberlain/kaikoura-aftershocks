"""
Compare my magnitudes to GeoNets.

"""
import requests
import datetime
import os
import numpy as np
import pandas as pd

from obspy import UTCDateTime
from obspy.clients.fdsn import Client


def get_geonet_quakes(
    min_latitude=-49.0, max_latitude=-40.0,
    min_longitude=164.0, max_longitude=182.0,
    min_magnitude=0.0, max_magnitude=9.0,
    min_depth=0.0, max_depth=100.0,
    start_time=datetime.datetime(1960, 1, 1),
    end_time=datetime.datetime(2020, 1, 1),
):
    """
    Get a dataframe of the eatrhquakes in the GeoNet catalogue.
    
    Parameters
    ----------
    min_latitude
        Minimum latitude in degrees for search
    max_latitude
        Maximum latitude in degrees for search
    min_longitude
        Minimum longitude in degrees for search
    max_longitude
        Maximum longitude in degrees for search
    min_depth
        Minimum depth in km for search
    max_depth
        Maximum depth in km for search
    min_magnitude
        Minimum magnitude for search
    max_magnitude
        Maximum magnitude for search
    start_time
        Start date and time for search
    end_time
        End date and time for search
        
    Returns
    -------
    pandas.DateFrame of resulting events
    """
    quakes = []
    max_chunk_size = 365 * 86400
    _starttime = start_time
    _endtime = _starttime + max_chunk_size
    kwargs = dict(min_latitude=min_latitude, min_longitude=min_longitude, 
            max_latitude=max_latitude, max_longitude=max_longitude,
            min_depth=min_depth, max_depth=max_depth, 
            min_magnitude=min_magnitude, max_magnitude=max_magnitude)
    while _endtime < end_time:
        quakes.append(_get_geonet_quakes(           
            start_time=_starttime, end_time=_endtime, **kwargs))
        _starttime += max_chunk_size
        _endtime += max_chunk_size
    quakes.append(_get_geonet_quakes(
        start_time=_starttime, end_time=end_time, **kwargs))

    earthquakes = quakes[0]
    for df in quakes[1:]:
        earthquakes = earthquakes.append(df, ignore_index=True)
    return earthquakes


def _get_geonet_quakes(
    min_latitude, max_latitude, min_longitude, max_longitude, min_magnitude, 
    max_magnitude, min_depth, max_depth, start_time, end_time):
    # Convert start_time and end_time to strings
    start_time = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end_time = end_time.strftime("%Y-%m-%dT%H:%M:%S")
    # Use the more efficient f-string formatting
    query_string = (
        "https://quakesearch.geonet.org.nz/csv?bbox="
        f"{min_longitude},{min_latitude},{max_longitude},"
        f"{max_latitude}&minmag={min_magnitude}"
        f"&maxmag={max_magnitude}&mindepth={min_depth}"
        f"&maxdepth={max_depth}&startdate={start_time}"
        f"&enddate={end_time}")
    print(f"Using query: {query_string}")
    response = requests.get(query_string)
    if not response.ok:
        print(response.content)
        return
    with open(".earthquakes.csv", "wb") as f:
        f.write(response.content)
    earthquakes = pd.read_csv(
        ".earthquakes.csv", 
        parse_dates=["origintime", "modificationtime"],
        dtype={"publicid": str})
    earthquakes = earthquakes.rename(
        columns={" magnitude": "magnitude",
                " latitude": "latitude",
                " depth": "depth"})
    earthquakes = earthquakes.sort_values(by=["origintime"], ignore_index=True)
    os.remove(".earthquakes.csv")
    return earthquakes


def compare_to_geonet():
    from obspy.geodetics import gps2dist_azimuth
    import matplotlib.pyplot as plt
    from progressbar import ProgressBar


    my_quakes = pd.read_csv(
        "../../Locations/GrowClust_located_magnitudes_callibrated_focal_mechanisms.csv", parse_dates=["time"])
    my_quakes = my_quakes.rename(columns={'time': 'origintime'})

    startdate, enddate = (
        UTCDateTime(min(my_quakes.origintime)), 
        UTCDateTime(max(my_quakes.origintime)))

    min_lat, max_lat = min(my_quakes.latitude), max(my_quakes.latitude)
    min_lon, max_lon = min(my_quakes.longitude), max(my_quakes.longitude)

    geonet_quakes = get_geonet_quakes(
        min_latitude=min_lat - 0.1, max_latitude=max_lat + 0.1, 
        min_longitude=min_lon - 0.1, max_longitude=max_lon + 0.1, 
        start_time=startdate - 3600, end_time=enddate + 3600)
    geonet_origin_times = [UTCDateTime(t) for t in geonet_quakes.origintime]
    geonet_quakes.depth *= 1000  # Convert to m.

    # convert geonet origin times to seconds from some timestamp
    timestamp = startdate - 3600
    geonet_origin_times = np.array([t - timestamp for t in geonet_origin_times])

    bar = ProgressBar(max_value=len(my_quakes))
    matched_quakes = dict()
    for i in range(len(my_quakes)):
        origin_time = UTCDateTime(my_quakes.origintime[i])
        origin_seconds = origin_time - timestamp
        deltas = np.abs(geonet_origin_times - origin_seconds)
        index = np.argmin(deltas)
        delta = deltas[index]
        if delta <= 5.0:
            # distance check
            dist, _, _ = gps2dist_azimuth(
                lat1=geonet_quakes.latitude[index],
                lon1=geonet_quakes.longitude[index],
                lat2=my_quakes.latitude[i],
                lon2=my_quakes.longitude[i])
            dist = ((dist ** 2) + (
                (geonet_quakes.depth[index] - my_quakes.depth[i]) ** 2)) ** .5
            if dist <= 10000:  # within 10km
                geonet_id = geonet_quakes.publicid[index]
                if geonet_id in matched_quakes.keys():
                    # Check whether this is a better match
                    if delta > matched_quakes[geonet_id]["delta"] or dist > matched_quakes[geonet_id]["dist"]:
                        continue
                matched_quakes.update(
                    {geonet_id: dict(delta=delta, dist=dist, my_id=i)})
        bar.update(i)
    bar.finish()



    geonet_mags = [
        float(geonet_quakes.magnitude[geonet_quakes.publicid == geonet_ev_id]) 
        for geonet_ev_id in matched_quakes.keys()]
    my_mags = [float(my_quakes.magnitude[my_index["my_id"]]) 
               for my_index in matched_quakes.values()]
    magnitude_plot(geonet_mags=geonet_mags, my_mags=my_mags, 
                   geonet_ids=list(matched_quakes.keys()), 
                   my_ids=[item["my_id"] for item in matched_quakes.values()])
    matplot_mag_plot(geonet_mags=geonet_mags, my_mags=my_mags)


def matplot_mag_plot(geonet_mags, my_mags):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.scatter(geonet_mags, my_mags)
    ax.set_xlabel("GeoNet preferred_magnitude")
    ax.set_ylabel("Local Magnitude")
    ax.grid("on")
    plt.show()


def magnitude_plot(geonet_mags: list, my_mags: list, geonet_ids: list, my_ids: list):
    from bokeh.plotting import figure, ColumnDataSource, show
    
    plot_data = ColumnDataSource(data=dict(
        my_mags=my_mags, geonet_mags=geonet_mags, geonet_ids=geonet_ids,
        index=my_ids))

    tooltips = [
        ("My magnitude", "@my_mags"),
        ("GeoNet magnitude", "@geonet_mags"),
        ("My index", "@index"), 
        ("PublicID", "@geonet_ids")
    ]

    p = figure(
        tools="pan,box_zoom,reset,save", 
        x_axis_label="GeoNet preferred magnitude", 
        y_axis_label="Relative magnitude", tooltips=tooltips, 
        plot_width=1200, plot_height=1200)
    p.circle('geonet_mags', 'my_mags', source=plot_data)

    show(p)


if __name__ == "__main__":
    compare_to_geonet()