"""
Select possible repeaters from catalogue.

"""
import numpy as np

from obspy import Catalog, UTCDateTime
from obsplus.events.json import dict_to_cat

def get_repeats(
    cat_dict: dict,
    threshold: float = 0.90,
    min_stations: int = 2,
) -> Catalog:
    """ Uses the dictionary of events - this is lighter in memory than a full Catalog """

    templates = {
        [comment["text"].split("Template: ")[-1] for comment in event["comments"]
         if "Template:" in comment["text"]][0] for event in cat_dict["events"]}

    possible_repeats, self_detections = [], dict()
    for event in cat_dict["events"]: 
        event_time = UTCDateTime(event["origins"][-1]["time"]) 
        template = [comment["text"].split("Template: ")[-1] 
                    for comment in event["comments"] if "Template:" in comment["text"]][0] 
        template_time = UTCDateTime.strptime(template, "%Y_%m_%dt%H_%M_%S") 
        if abs(template_time - event_time) <= 120: 
            self_detections.update({template: event})
            continue 
        sta_corrs = [(pick["waveform_id"]["station_code"], float(pick["comments"][0]["text"].split("=")[-1])) for pick in event["picks"] if len(pick["comments"])] 
        sta_corrs = [_ for _ in sta_corrs if _[-1] >= threshold] 
        if len(sta_corrs) >= min_stations: 
            possible_repeats.append(event)
    
    templates_included, templates_to_add = [], []
    for event in possible_repeats:
        template = [comment["text"].split("Template: ")[-1] 
                    for comment in event["comments"] if "Template:" in comment["text"]][0]
        if template in templates_included:
            continue
        template_event = self_detections.get(template, None)
        if template_event is None:
            print(f"Template {template} not found")
            continue
        templates_to_add.append(template_event)
        templates_included.append(template)
    possible_repeats.extend(templates_to_add)
    # Make a catalog out of these
    possible_repeats = dict(
        events=possible_repeats, resource_id="local_possible_repeats",
        description="Possible repeating earthquakes around Kaikoura",
        comments=f"Threshold of {threshold} on {min_stations} stations",
        creation_info="")
    possible_repeats = dict_to_cat(possible_repeats)
    return possible_repeats


def _distance(lat1, lon1, depth1, lat2, lon2, depth2):
    from obspy.geodetics import degrees2kilometers, locations2degrees

    epi_dist = degrees2kilometers(locations2degrees(lat1, lon1, lat2, lon2))
    d_dist = abs(depth1 - depth2)
    return ((epi_dist ** 2) + (d_dist ** 2)) ** 0.5


def get_near_interface_events(
    cat_dict: dict, 
    distance_threshold: float = 5.0
) -> Catalog:
    from progressbar import ProgressBar
    from kaikoura_csv_visualisations import get_williams_contours

    subd_lats, subd_lons, subd_depths = get_williams_contours()
    subd_depths *= -1

    possible_interface_events = []
    bar = ProgressBar(max_value=len(cat_dict["events"]))
    for k, event in enumerate(cat_dict["events"]):
        lat = float(event["origins"][-1]["latitude"])
        lon = float(event["origins"][-1]["longitude"])
        depth = float(event["origins"][-1]["depth"]) / 1000
        lat_mask = np.where(np.abs(subd_lats - lat) < 0.2)
        lon_mask = np.where(np.abs(subd_lons - lon) < 0.2)
        _subd_lats = subd_lats[lat_mask]
        _subd_lons = subd_lons[lon_mask]
        _subd_depths = subd_depths[lat_mask][:, lon_mask]
        dists = [_distance(lat, lon, depth, _subd_lats[i], 
                           _subd_lons[j], _subd_depths[i][0][j]) 
                 for i in range(len(_subd_lats)) for j in range(len(_subd_lons))]
        dists = [d for d in dists]
        if len(dists) and min(dists) < distance_threshold:
            possible_interface_events.append(event)
        bar.update(k)

    possible_interface_events = dict(
        events=possible_interface_events, resource_id="possible_interface_events",
        description=f"Events within {distance_threshold} km of the interface",
        comments="", creation_info="")
    possible_interface_events = dict_to_cat(possible_interface_events)
    return possible_interface_events