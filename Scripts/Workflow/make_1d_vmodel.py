"""
Script to extract an average 1D model from Donna's NZ3D model.

"""
import numpy as np
import pandas as pd


MODEL_FILE = "vlnzw2p2dnxyzltln.tbl.txt"  # NZ3D 2.2 from https://zenodo.org/record/3779523#.YCRFaOrRVhF


def extract_one_d(
    min_x: float = 72.0, 
    max_x: float = 110.0, 
    min_y: float = -100.0, 
    max_y: float = 80.0,
) -> pd.DataFrame:
    """
    Extract a one-d spatial average velocity model from NZ3D.

    Parameters
    ----------
    min_x:
        Minimum X value in NZ3D co-ordinate system
    max_x:
        Maximum X value in NZ3D co-ordinate system
    min_y:
        Minimim Y value in NZ3D co-ordinate system
    max_y:
        Maximum Y value in NZ3D co-ordinate system

    """
    v_model = pd.read_csv(MODEL_FILE, header=1, delim_whitespace=True)

    x_mask = np.logical_and(
        v_model["x(km)"] <= max_x, v_model["x(km)"] >= min_x)
    y_mask = np.logical_and(
        v_model["y(km)"] <= max_y, v_model["y(km)"] >= min_y)
    
    mask = np.logical_and(x_mask, y_mask)

    region = v_model[mask]
    # Make a quick plot showing the region
    bl = region[np.logical_and(region["x(km)"] == region["x(km)"].min(),
                               region["y(km)"] == region["y(km)"].min())]
    br = region[np.logical_and(region["x(km)"] == region["x(km)"].min(),
                               region["y(km)"] == region["y(km)"].max())]
    tl = region[np.logical_and(region["x(km)"] == region["x(km)"].max(),
                               region["y(km)"] == region["y(km)"].min())]
    tr = region[np.logical_and(region["x(km)"] == region["x(km)"].max(),
                               region["y(km)"] == region["y(km)"].max())]
    bl = (bl.Latitude.to_list()[0], bl.Longitude.to_list()[0])
    br = (br.Latitude.to_list()[0], br.Longitude.to_list()[0])
    tl = (tl.Latitude.to_list()[0], tl.Longitude.to_list()[0])
    tr = (tr.Latitude.to_list()[0], tr.Longitude.to_list()[0])
    plot_region(corners=[bl, tl, tr, br])

    depths = sorted(list(set(region["Depth(km_BSL)"])))

    # Get average vp and vs for each depth
    vp, vs = [], []
    for depth in depths:
        vp.append((region[region["Depth(km_BSL)"] == depth]).Vp.mean())
        vs.append((region[region["Depth(km_BSL)"] == depth]).Vs.mean())
    out = pd.DataFrame(data={"Depth": depths, "vp": vp, "vs": vs})
    return out


def plot_region(corners):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    fig = plt.figure()
    ax = fig.add_subplot(projection=ccrs.PlateCarree())

    lats, lons = zip(*corners)
    lats, lons = list(lats), list(lons)
    ax.set_extent((min(lons) - 1, max(lons) + 1, min(lats) - 1, max(lats) + 1), 
                  crs=ccrs.PlateCarree())
    ax.coastlines()
    lons.append(lons[0])
    lats.append(lats[0])
    ax.plot(lons, lats, transform=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)

    plt.show()



if __name__ == "__main__":
    vmodel = extract_one_d()

    # Write to GrowClust format
    with open("vzmodel.txt", "w") as f:
        for _, row in vmodel.iterrows():
            f.write(f"{row.Depth:5.1f} {row.vp:.2f} {row.vs:.2f}\n")
    