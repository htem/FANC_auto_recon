# FANC_auto_recon

FANC (pronounced "fancy") is the Female Adult Nerve Cord, a GridTape-TEM dataset of an adult _Drosophila melanogaster_'s ventral nerve cord. The dataset was first published in [Phelps, Hildebrand, Graham et al. 2021 _Cell_](https://www.lee.hms.harvard.edu/phelps-hildebrand-graham-et-al-2021), after which we applied automated methods for reconstructing neurons, synapses, and nuclei to accelerate reconstruction of the ventral nerve cord connectome. This repository contains a python package for interacting with the connectome (see the folder [`fanc`](fanc)) as well as other supporting files. 

Have any questions? Please [open an issue](https://github.com/htem/FANC_auto_recon/issues/new) or contact [Jasper Phelps (jasper.s.phelps@gmail.com)](https://github.com/jasper-tms).

## Installing and configuring the `fanc` python package

### Before you start

As is always the case in python, consider making a virtual environment (using your preference of virtualenv/virtualenvwrapper or conda) before installing.

### Installation option 1: pip install directly from GitHub

    pip install git+https://github.com/htem/FANC_auto_recon.git

### Installation option 2: Clone then install

    cd ~/repos  # or wherever you keep your repos
    git clone https://github.com/htem/FANC_auto_recon.git
    cd FANC_auto_recon
    pip install -e .

### Additional installation steps for mesh manipulation and transform

The mesh manipulation and coordinate transform code requires `pytransformix`, which is itself a Python wrapper for Elastix. Therefore, Elastix must be installed and its lib and bin paths must be appended to the `LD_LIBRARY_PATH` and `PATH` environment variables. See [`pytransformix` documentation](https://github.com/jasper-tms/pytransformix#installation) for details.


### Provide credentials

Access to the latest reconstruction of FANC is restricted to authorized users. If you are a member of the FANC community and have been granted access, you can generate an API key by visiting [https://global.daf-apis.com/auth/api/v1/create_token](https://global.daf-apis.com/auth/api/v1/create_token) and logging in with your FANC-authorized google account. Copy the key that is displayed, then create a text file that contains:

    {
      "token": "THE API KEY YOU COPIED"
    }

and save that file to `~/.cloudvolume/secrets/cave-secret.json`. You could, for example, create such a file by running the following commands from a terminal:

    mkdir -p ~/.cloudvolume/secrets  # Create this folder if it doesn't exist
    echo '{ "token": "THE API KEY YOU COPIED" }' >> ~/.cloudvolume/secrets/cave-secret.json
