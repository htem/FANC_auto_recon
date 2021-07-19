# Transform points between different VNC coordinate frames

### realignment.py
Functions for transforming coordinates between FANCv3 (FANC alignment version 3, produced by the Lee lab using [AlignTK](https://mmbios.pitt.edu/aligntk-home), published in [Phelps, Hildebrand, Graham et al. 2021](https://www.lee.hms.harvard.edu/phelps-hildebrand-graham-et-al-2021), and is the alignment version on [FANC CATMAID](https://radagast.hms.harvard.edu/catmaidvnc/?pid=13&zp=54000&yp=541313&xp=168658&tool=tracingtool&sid0=10&s0=8)) and FANCv4 (FANC alignment version 4, produced by Zetta AI using Seamless, not yet published, and is the alignment version used for automated reconstruction and so is the version visible when proofreading in neuroglancer). This module has no prerequisites other than a python environment with the packages imported in the script's first few lines.

### warp_points_between_FANC_and_template.py

Functions for transforming coordinates between FANCv3 and the `JRC2018_VNC_FEMALE` template available [here from Janelia](https://www.janelia.org/open-science/jrc-2018-brain-templates). PREREQUISITES: [Download elastix](https://elastix.lumc.nl/download.php) and add its `bin/` folder to your shell `PATH` so that `elastix` and `transformix` can be called from the command line.
