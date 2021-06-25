"""
Script to add in the focal mechanism solutions from the Bayesianess.

ONLY WORKS ON KEA!
"""

import pandas as pd
import numpy as np
import datetime as dt
import glob

from math import sin, cos, degrees, radians, asin

from obspy import read_events


FM_FILE = "/home/chambeca/Desktop/Kaikoura_Bayesian_focmech/projects/Kaikoura_templates/outdata/Kaikoura_templates_scalar_err_degrees.csv"
NLL_FILES = glob.glob(
    "/home/chambeca/Desktop/Kaikoura_Bayesian_focmech/projects/Kaikoura_templates/loc/*.hyp")


def slip(s, d, r): 
    """ Compute slip vector. """
    _slip = [0, 0, 0] 
    _slip[0] = (cos(radians(r)) * cos(radians(s))) + ( 
        sin(radians(r)) * cos(radians(d)) * sin(radians(s))) 
    _slip[1] = (-1 * cos(radians(r)) * sin(radians(s))) + ( 
        sin(radians(r)) * cos(radians(d)) * cos(radians(s))) 
    _slip[2] = sin(radians(r)) * sin(radians(d)) 
    return np.array(_slip)


def correct_fm(s, d, r): 
    """ Correct FMs with dips out of range. """
    if d <= 90 and d >= 0: 
        return s, d, r 
    # Dip is not as expected... 
    print(f"Correcting {s}/{d}/{r}") 
    s_out = (s - 180) % 360 
    d_out = d % 180  # Make sure it sits within 0-180 
    d_out = 180 - d_out
    r_out = (180 - r) + 180
    slip_3 = sin(radians(r)) * sin(radians(d)) 
    # Flip - we are switching hanging-wall 
    slip_3 *= -1 
    # r_out = degrees(asin(slip_3 / sin(radians(d_out)))) 
    print(f"Corrected to {s_out}/{d_out}/{r_out}") 
    slip_in = slip(s, d, r) 
    slip_out = slip(s_out, d_out, r_out) 
    slip_out = slip_out * -1 
    assert np.allclose(slip_in, slip_out), f"Slip_in:\n{slip_in}\nSlip_out:\n{slip_out}" 
    return s_out, d_out, r_out


def main(
    infile: str = "../../Locations/GrowClust_located_magnitudes_callibrated.csv",
    max_err: float = 30.0,
):
    
    earthquakes = pd.read_csv(infile, parse_dates=["time"])
    print("Read in quakes")
    fm_db = pd.read_csv(FM_FILE, dtype={"eq": str})

    nll_files = {
        f.split("Kaikoura_templates.")[-1].split(".grid0")[0]: f 
        for f in NLL_FILES}

    strike, dip, rake, kappa, err = np.ones((5, len(earthquakes))) * np.nan
    for i in range(len(fm_db)):
        nll_id = fm_db["eq"][i]
        nll_file = nll_files.get(nll_id, None)
        if nll_file is None:
            raise NotImplementedError(f"No file for {nll_id}")
        nll_event = read_events(nll_file)[0]
        eq_time = nll_event.preferred_origin().time.datetime

        diffs = [abs(t.total_seconds()) for t in earthquakes.time - eq_time]
        eq_no = np.argmin(diffs)
        diff = diffs[eq_no]
        assert diff < 10.0, f"Difference of {diff} out of range"
        if fm_db.err[i] > max_err:
            print(f"Error: {fm_db.err[i]} is too big!")
        strike[eq_no], dip[eq_no], rake[eq_no], kappa[eq_no], err[eq_no] = (
            fm_db.strike[i], fm_db.dip[i], fm_db.rake[i], fm_db.kappa[i],
            fm_db.err[i])

    for i in range(len(strike)):
        s, d, r = strike[i], dip[i], rake[i]
        if np.isnan(s):
            continue
        strike[i], dip[i], rake[i] = correct_fm(s, d, r)
        # Migrate rake to +/- 180
        if rake[i] > 180:
            rake[i] = -1 * (360 - rake[i])

    earthquakes["strike"], earthquakes["dip"], earthquakes["rake"] = (
        strike, dip, rake)
    earthquakes["kappa"], earthquakes["scalar error"] = kappa, err

    # Add in template id column
    template_id = ["_".join(eid.split("_")[0:-1]).split('/')[-1] 
                   for eid in earthquakes.event_id]
    earthquakes["template-id"] = template_id

    template_slip_styles = dict()
    keep_indexes = []
    for tid in set(template_id):
        dets = earthquakes[earthquakes["template-id"] == tid]
        template_fm = dets[~pd.isna(dets.strike)]
        if len(template_fm) > 1:
            print("More than 1 mechanism for this template, taking the best")
            # Remove the other mechanism(s)!
            template_fm = template_fm[
                template_fm["scalar error"] == template_fm["scalar error"].min()]       
        elif len(template_fm) == 0:
            template_slip_styles.update({tid: "Unknown"})
            continue
        keep_indexes.append(template_fm.index[0])
        s, d, r = (template_fm.strike.to_list()[0], 
                   template_fm.dip.to_list()[0],
                   template_fm.rake.to_list()[0])
        if -135 <= r < -45:
            slip_style = "Normal"
        elif 45 <= r < 135:
            slip_style = "Reverse"
        else:
            slip_style = "Strike-slip"
        template_slip_styles.update({tid: slip_style})

    # Remove unneeded, duplicate mechanisms
    keep_indexes.sort()
    strike, dip, rake, kappa, err = np.ones((5, len(earthquakes))) * np.nan
    for i in keep_indexes:
        strike[i] = earthquakes.strike[i]
        dip[i] = earthquakes.dip[i]
        rake[i] = earthquakes.rake[i]
        kappa[i] = earthquakes.kappa[i]
        err[i] = earthquakes["scalar error"][i]
    
    # Overwrite the old columns
    earthquakes["strike"] = strike
    earthquakes["dip"] = dip
    earthquakes["rake"] = rake
    earthquakes["kappa"] = kappa
    earthquakes["scalar error"] = err

    # Add in slip-style column
    slip_style = [template_slip_styles.get(tid, "Unknown") 
                  for tid in earthquakes["template-id"]]
    
    earthquakes["Slip style"] = slip_style

    earthquakes.to_csv(
        "../../Locations/GrowClust_located_magnitudes_callibrated_focal_mechanisms.csv")
    print("Written FM file")


if __name__ == "__main__":
    main()
