# Kaikoura afterslip

These files were used to generate a matched-filter catalog of Kaikoura aftershocks
using the catalog of 2,655 earthquakes re-picked by the STREWN team (led by
Federica Lanza, University of Wisconcin Madison) as templates. Calum Chamberlain
used EQcorrscan to generate templates and run the correlations.

## Directory structure

- Kaik_afterslip/
    - README.md - This file
    - Templates/
        - SIMUL_locations.xml - Locations and picks from STREWN as relocated using SIMUL
        - *.tgz - Tribe archives from EQcorrscan
    - Detections/
        - *.tgz - Party archives from EQcorrscan
    - Scripts/
        - Workflow/
            - generate_templates.py - Script for generating templates
            - kaikoura_detect.py - Script for running the matched-filter routines
            - simul_locate.py - Locate detections using SIMUL
        - Analysis_Visualisation/  - Scripts and functions for analysing the catalogue
        

## Workflow:

1. Scripts/generate_templates.py
2. Scripts/kaikoura_detect.py
3. Scripts/post_process.py
4. Scripts/simul_locate.py
5. Scripts/pick_magnitudes.py
6. Run magnitude inversion code over catalogue - cjc_utilities/magnitude_inversion/magnitude_inversion.py
7. Scripts/prepare_relocations.py
