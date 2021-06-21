"""
Compute magnitude inversion - required because GeoNet magnitudes are rubbish.
"""
import json
import pandas as pd
import requests

from obspy.clients.fdsn import Client
from obspy import read_events, UTCDateTime
from obspy.core.event import Magnitude, ResourceIdentifier, Catalog

from cjc_utilities.magnitude_inversion.magnitude_inversion import (
    magnitude_inversion, _get_origin_attrib)


def get_ristau_moment_tensor_db(
    min_latitude: float = -49.0, max_latitude: float = -40.0,
    min_longitude: float = 164.0, max_longitude: float = 182.0,
    min_magnitude: float = 0.0, max_magnitude: float = 9.0,
    min_depth: float = 0.0, max_depth: float = 500.0,
    start_time: UTCDateTime = UTCDateTime(1960, 1, 1),
    end_time: UTCDateTime = UTCDateTime(2020, 1, 1),
):
    """
    Get a dataframe of the earthquakes in the GeoNet catalogue.
    
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
    raw_path = (
        "https://raw.githubusercontent.com/GeoNet/data/master/moment-tensor/"
        "GeoNet_CMT_solutions.csv")
    print(f"Using query: {raw_path}")
    response = requests.get(raw_path)
    with open("Ristau_earthquakes.csv", "wb") as f:
        f.write(response.content)
    earthquakes = pd.read_csv(
        "Ristau_earthquakes.csv", 
        dtype={"PublicID": str})
    # Convert "Date" column
    earthquakes.Date = pd.to_datetime(earthquakes.Date, format="%Y%m%d%H%M%S")
    ##### Filters.

    # Date
    earthquakes = earthquakes[
        (earthquakes.Date <= end_time.datetime) & 
        (earthquakes.Date >= start_time.datetime)]
    
    # Latitude
    earthquakes = earthquakes[
        (earthquakes.Latitude <= max_latitude) &
        (earthquakes.Latitude >= min_latitude)]
    
    # Longitude
    earthquakes = earthquakes[
        (earthquakes.Longitude <= max_longitude) &
        (earthquakes.Longitude >= min_longitude)]
    
    # Depth
    earthquakes = earthquakes[
        (earthquakes.CD <= max_depth) &
        (earthquakes.CD >= min_depth)]
    
    # Magnitude
    earthquakes = earthquakes[
        (earthquakes.Mw <= max_magnitude) &
        (earthquakes.Mw >= min_magnitude)]
    return earthquakes


def main():
    new_catalog = read_events(
        "../../Locations/NonLinLoc_NZW3D_2.2/NLL_located_magnitudes.xml")
    
    # Get comparison events - just use those in the Kaikoura region.
    min_lat, max_lat, min_lon, max_lon = -42.9, -41.3, 172.3, 175.0
    starttime = min(_get_origin_attrib(ev, "time") for ev in new_catalog) - 3600
    endtime = max(_get_origin_attrib(ev, "time") for ev in new_catalog) + 3600
    client = Client("GEONET")

    # Get John Ristau's Moment Tensor catalog
    ristau_quakes = get_ristau_moment_tensor_db(
        min_latitude=min_lat, min_longitude=min_lon, max_latitude=max_lat,
        max_longitude=max_lon, start_time=starttime, end_time=endtime)

    comparison_cat = Catalog()
    for row in ristau_quakes.itertuples():
        event = client.get_events(eventid=row.PublicID)[0]
        # Add Ristau moment magnitude
        if "Mw" not in {mag.magnitude_type for mag in event.magnitudes}:
            ristau_mag = Magnitude(
                magnitude_type="Mw", mag=row.Mw, 
                method_id=ResourceIdentifier("Ristau"))
            event.magnitudes.append(ristau_mag)
            event.preferred_magnitude_id = ristau_mag.resource_id
        comparison_cat += event
    
    # Note: inversion too large for KEA, run on SGEES001
    output, gamma, station_corrections = magnitude_inversion(
        new_catalog=new_catalog, callibration_catalog=comparison_cat,
        time_difference=5.0, epicentral_difference=20.0, depth_difference=30.0,
        magnitude_type="Mw", only_matched=False, in_place=True, plot=False)
    # output, gamma, station_corrections = magnitude_inversion(
    #     new_catalog=new_catalog, callibration_catalog=comparison_cat,
    #     magnitude_type="Mw", only_matched=True, in_place=True)

    output.write(
        "../../Locations/NonLinLoc_NZW3D_2.2/NLL_located_magnitudes_callibrated.xml",
        format="QUAKEML")
    print(f"Written callibrated catalog")

    out_parameters = {
        "gamma": gamma,
        "station_corrections": station_corrections}
    
    print("######## RESULTS #########")
    print(f"gamma:\t{gamma}")
    print("STATION CORRECTIONS")
    for station, correction in station_corrections.items():
        print(f"{station}:\t{correction}")
    print("\n\n")

    with open("magnitude_inversion.json", "w") as f:
        json.dump(out_parameters, f)    
    print("Written gamma and station corrections to magnitude_inversion.json")


if __name__ == "__main__":
    main()