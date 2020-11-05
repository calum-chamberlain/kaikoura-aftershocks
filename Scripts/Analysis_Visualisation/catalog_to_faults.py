"""
Separate the Kaikoura catalog into subcatalogs associated with Faults
modelled in Hamling et al., (2017).

The fault model is broken down into small chunks, we will use some functions
Calum wrote for extracting earthquakes close to fault planes
(https://bitbucket.org/calum-chamberlain/utilities/src/master/cjc_utilities)
to get earthquakes for each fault plane segment then chunk them into separate
overall fault segments.  Whatever is left is unassociated.

Calum J. Chamberlain
November 2018
"""
from itertools import cycle
import numpy as np

from obspy import read_events, Catalog
from cjc_utilities.coordinates.coordinates import Geographic, Location

fault_model = "Hamling_etal_supplement/aam7194_Data_S3.dat"
# KAIK_CATALOG = "/home/calumch/Dropbox/Manuscripts/STREWN_initial/catalogs/final_catalog_REST_KPICK_ev632_pol.xml"
DIP_PERP_OFFSET = 5.0  # Dip-perpendicular offset for an earthquake to be 
# considered associated with a fault in kilometers.


class FaultPlane():
    """
    A Fault plane defined by a Geographic mid-point, half-width (horizontal, km)
    and half-height (vertical, km), strike and dip (degrees).
    """
    def __init__(self, middle, half_width, half_height, strike, dip):
        self.middle = middle
        self.half_width = half_width
        self.half_height = half_height
        self.strike = strike
        self.dip = dip

    def point_nearby(self, location, dip_perpendicular_distance):
        """
        Calculate whether a Geographic point is within 
        dip-perpendicular-distance of the FaultPlane.

        :type location: Geographic
        :param location: The location of the point to test
        :type dip_perpendicular_distance: float
        :param dip_perpendicular_distance: 
            Distance in km for location to be within away from the FaultPlane.
        """
        projected_location = location.to_xyz(
            origin=self.middle, strike=self.strike, dip=self.dip)
        if abs(projected_location.z) > self.half_height:
            return False
        if abs(projected_location.y) > self.half_width:
            return False
        if abs(projected_location.x) > dip_perpendicular_distance:
            return False
        return True

    @property
    def corners(self):
        corners = [
            Location(
                x=0, y=self.half_width, z=self.half_height, origin=self.middle,
                strike=self.strike, dip=self.dip),
            Location(
                x=0, y=self.half_width, z=-self.half_height, origin=self.middle,
                strike=self.strike, dip=self.dip),
            Location(
                x=0, y=-self.half_width, z=-self.half_height, origin=self.middle,
                strike=self.strike, dip=self.dip),
            Location(
                x=0, y=-self.half_width, z=self.half_height, origin=self.middle,
                strike=self.strike, dip=self.dip)]
        return [corner.to_geographic() for corner in corners]
        

def hamling_fault_mesh(in_file):
    """
    Read in Hamling's fault mesh and return a series of faults.

    :type in_file: str
    :param in_file: The data file to read from.
    
    :rtype: dict
    """
    faults = {}
    with open(in_file, "rb") as f:
        faults_raw = f.read()
    fault_lines = faults_raw.decode("UTF-8").split("\n")
    segment_indexes = {}
    for header, line in enumerate(fault_lines):
        if line.startswith("long"):
            break
        index, names = tuple(line.lstrip("Fault index ").split(" = "))
        if "-" in index:
            _start, _end = tuple(index.split("-"))
            for index in range(int(_start), int(_end) + 1):
                segment_indexes.update({index: names})
        else:
            segment_indexes.update({int(index): names})
        faults.update({names: []})
    for patch_line in fault_lines[header + 1:]:
        patch_line = patch_line.split()
        if len(patch_line) == 0:
            continue

        half_width = float(patch_line[5]) / 2
        top = float(patch_line[6])
        bottom = float(patch_line[7])
        half_height = (bottom - top) / 2
        mid_depth = top + half_height
        middle = Geographic(
            latitude=float(patch_line[1]), longitude=float(patch_line[0]),
            depth=mid_depth)
        patch = FaultPlane(
            middle=middle, half_width=half_width, half_height=half_height,
            strike=float(patch_line[2]) - 180, dip=float(patch_line[3]))
        name = segment_indexes[int(float(patch_line[-1]))]
        faults[name].append(patch)

    return faults


def catalog_to_geographic(catalog):
    """
    Convert a catalog of events to Geographic
    
    :type catalog: `obspy.core.event.Catalog`

    :rtype: list
    """
    locations = []
    for event in catalog:
        try:
            magnitude = (
                event.preferred_magnitude or event.magnitudes[0])
            magnitude = magnitude.mag
        except:
            magnitude = None
        origin = event.preferred_origin() or event.origins[0]
        try:
            origin_time = origin.time.datetime
        except AttributeError:
            origin_time = None
        location = Geographic(
            latitude=origin.latitude, longitude=origin.longitude,
            depth=origin.depth / 1000, magnitude=magnitude,
            time=origin_time)
        locations.append(location)
    return locations


def plot_events(associated_catalogs, unassociated_catalog, faults=None):
    """
    3D-scatter plot of events associated with faults and those not associated.

    :type associated_catalogs: dict
    :type unassociated_catalog: `obspy.core.event.Catalog`
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unassociated_locations = catalog_to_geographic(unassociated_catalog)
    x, y, z = zip(*[(l.longitude, l.latitude, l.depth) 
                    for l in unassociated_locations])
    ax.scatter(x, y, z, c="gray", marker="o", label="Unassociated", s=0.6)

    colorlist = [
        "red", "blue", "green", "purple", "cyan", "orange", "darksalmon",
        "gold", "navy", "fuchsia", "darkslategrey", "aquamarine", "cyan",
        "saddlebrown", "orchid", "lightpink", "darkseagreen", "lime",
        "skyblue", "plum"]

    colors = cycle(colorlist)
        
    for name, cat in associated_catalogs.items():
        if len(cat) == 0:
            continue
        locs = catalog_to_geographic(cat)
        x, y, z = zip(*[(l.longitude, l.latitude, l.depth) 
                    for l in locs])
        ax.scatter(x, y, z, c=next(colors), marker="o", label=name)
    if faults:
        colors = cycle(colorlist)
        for name, fault_planes in faults.items():
            color = next(colors)
            cx, cy, cz = ([], [], [])
            for fault_plane in fault_planes:
                _cx, _cy, _cz = zip(*[(c.longitude, c.latitude, c.depth) 
                                    for c in fault_plane.corners])
                _cx, _cy, _cz = (list(_cx), list(_cy), list(_cz))
                _cx.append(_cx[0])
                _cy.append(_cy[0])
                _cz.append(_cz[0])
                cx.extend(_cx)
                cy.extend(_cy)
                cz.extend(_cz)
            ax.plot(cx, cy, cz, alpha=0.4, c=color)
    ax.invert_zaxis()
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_zlabel("Depth(km)")
    fig.legend()
    return fig    


def _split_on_position(s, splits, formats=None):
    """
    Split a string at given positions, optionally convert those sections.

    :type s: str
    :param s: String to split
    :type splits: list
    :param splits: Widths of splits
    :type formats: list
    :param formats: List of formats to convert using

    :rtype: list
    """
    out = []
    position = 0
    for i, split in enumerate(splits):
        value = s[position:position + split]
        position += split
        if formats is not None:
            try:
                value = formats[i](value)
            except IndexError:
                raise IndexError("Format not given for {0}".format(value))
            except Exception as e:
                print("Could not work out what {0} means in position "
                      "{1}".format(value, position))
                print(s)
                raise(e)
        yield(value)


def make_fault_catalogs(earthquakes, faults=None, fault_points=None):
    """
    Split catalog of events into events associated with separate faults.

    :type earthquakes: `obspy.core.event.Catalog`
    :type faults: dict of FaultPlanes
    :type fault_points: dict of numpy arrays.

    :returns: dict
    """
    assert faults or fault_points, "Requires either faults of fault_points"
    fault_catalogs = {}
    if faults is not None:
        for key, patches in faults.items():
            fault_catalog = Catalog()
            for earthquake in earthquakes:
                origin = earthquake.preferred_origin() or earthquake.origins[0]
                location = Geographic(
                    latitude=origin.latitude, longitude=origin.longitude,
                    depth=origin.depth / 1000)
                for patch in patches:
                    if patch.point_nearby(
                       location=location, 
                       dip_perpendicular_distance=DIP_PERP_OFFSET):
                        fault_catalog += earthquake
                        break
            fault_catalogs.update({key: fault_catalog})
    else:
        # Project all
        earthquake_locations = catalog_to_geographic(earthquakes)
        projection_origin = Geographic(
            latitude=min([e.latitude for e in earthquake_locations]),
            longitude=min([e.longitude for e in earthquake_locations]),
            depth=0)
        # Project fault points
        fault_point_locations = {key: [
            Geographic(latitude=fp[0], longitude=fp[1], depth=fp[2]).to_xyz(
                origin=projection_origin, strike=0, dip=90)
            for fp in value] for key, value in fault_points.items()}

        fault_catalogs = {fault_name: Catalog() 
                          for fault_name in fault_points.keys()}
        for eq in earthquakes:
            origin = eq.preferred_origin() or eq.origins[0]
            eq_location = Geographic(
                latitude=origin.latitude, longitude=origin.longitude, 
                depth=origin.depth / 1000).to_xyz(
                    origin=projection_origin, strike=0, dip=90)
            min_dist = 9999
            nearest_fault = None
            for fault_name, _fault_points in fault_point_locations.items():
                x, y, z = zip(*[(fp.x, fp.y, fp.z) for fp in _fault_points])
                x, y, z = (
                    np.abs(np.array(x) - eq_location.x), 
                    np.abs(np.array(y) - eq_location.y),
                    np.abs(np.array(z) - eq_location.z))
                distances = np.sqrt(np.square(x) + np.square(y) + np.square(z))
                _min_dist = distances.min()
                if _min_dist < min_dist:
                    min_dist = _min_dist
                    nearest_fault = fault_name
            if nearest_fault is None or min_dist > DIP_PERP_OFFSET:
                continue
            fault_catalogs.update(
                {nearest_fault: fault_catalogs[nearest_fault] + eq})
    # Get the unassociated events
    unassoc = Catalog()
    for earthquake in earthquakes:
        for catalog in fault_catalogs.values():
            if earthquake in catalog:
                break
        else:
            unassoc += earthquake
    return fault_catalogs, unassoc


def read_hypodd(filename):
    from obspy.core.event import (
        Catalog, Event, Magnitude, Origin, ResourceIdentifier)
    with open(filename, 'rb') as f:
        lines = f.read().decode("UTF8").splitlines()
    catalog = Catalog()
    for line in lines:
        line = line.split()
        if len(line) != 5:
            print("Could not read event from line:\n\t{0}".format(line))
            continue
        event = Event(
            origins=[Origin(longitude=float(line[0]), latitude=float(line[1]), 
                            depth=float(line[2]) * 1000)],
            magnitudes=[Magnitude(mag=float(line[3]))],
            resource_id=ResourceIdentifier(line[4]))
        catalog.append(event)
    return catalog


def read_dsaa_grd(filename, null_value=1.7014100000000001e+38,
                  return_grid=False, projection='epsg:2193'):
    """
    Read a Surfer format grid file.

    Returns x, y, z values as numpy arrays
    """
    import numpy as np
    import pyproj

    with open(filename, 'rb') as f:
        lines = f.read().decode('UTF8').splitlines()

    header = lines[0:5]
    assert header[0] == 'DSAA'
    nx, ny = [int(_) for _ in header[1].split()]
    x_min, x_max = [float(_) for _ in header[2].split()]
    y_min, y_max = [float(_) for _ in header[3].split()]
    z_min, z_max = [float(_) for _ in header[4].split()]
    x = np.linspace(start=x_min, stop=x_max, num=nx)
    y = np.linspace(start=y_min, stop=y_max, num=ny)

    data = lines[5:]

    _data = []
    _data_line = []
    for data_line in data:
        if len(data_line) > 0:
            bob = [float(dp) for dp in data_line.split()]
            for i in range(len(bob)):
                if bob[i] == null_value:
                    bob[i] = np.nan
            _data_line.extend(bob)
        else:
            if len(_data_line) > 0:
                _data.append(np.asarray(_data_line))
                _data_line = []
    if len(_data_line) > 0:
        _data.append(np.asarray(_data_line))
    data = np.asarray(_data)

    assert data.shape == (ny, nx)

    if not return_grid:
        return x, y, data

    grid = np.empty((nx * ny, 3))
    for i, _x in enumerate(x):
        for j, _y in enumerate(y):
            index = (i * ny) + j
            lon, lat = pyproj.transform(
                pyproj.Proj("+init={0}".format(projection)),
                pyproj.Proj("+init=EPSG:4326"), _x, _y)
            grid[index, 0] = lat
            grid[index, 1] = lon
            grid[index, 2] = data[j, i] / -1000  # convert to km depth
    # Remove nan-values from grid
    return grid[~np.isnan(grid[:, 2])]


if __name__ == '__main__':
    import glob
    # faults = hamling_fault_mesh(in_file=fault_model)
    litchfield_faults = glob.glob("grid_files_Litchfield_faults/*.grd")
    fault_points = {}
    for f in litchfield_faults:
        name = f.split("to model 3D - ")[-1].split(".grd")[0]
        if len(name.split(' - ')) > 1:
            name = name.split(' - ')[1]
        fault_points.update({
            name: read_dsaa_grd(f, return_grid=True)})
    # earthquakes = read_events(KAIK_CATALOG)
    # earthquakes = read_events("../Detections/all_detections.xml")
    earthquakes = read_hypodd("hypoDD_reloc_ID.gmt")
    
    fault_catalogs, unassoc = make_fault_catalogs(
        earthquakes=earthquakes, fault_points=fault_points)
        # earthquakes=earthquakes, faults=faults)

    for name, catalog in fault_catalogs.items():
        _name = name.replace("/", "_").replace(" ", "-")
        catalog.write("{0}_catalog.xml".format(_name), format="QUAKEML")
    

    unassoc.write("Unassociated_events.xml", format="QUAKEML")

    fig = plot_events(fault_catalogs, unassoc)  #, faults)
    fig.show()