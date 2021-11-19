# FANC_auto_recon

Code for interacting with the automated neuron and synapse reconstructions of FANC, the Female Adult Nerve Cord GridTape-TEM dataset. See [here](https://www.lee.hms.harvard.edu/phelps-hildebrand-graham-et-al-2021) for more information about the dataset.

This repo can't be `pip install`ed yet (we'll make this possible at some point) so you may want to add it to your `PYTHONPATH` so you can import its modules regardless of what folder you're in, e.g. if you cloned this repo to `~/repos/FANC_auto_recon`, add `export PYTHONPATH=~/repos/:$PYTHONPATH` to your shell config file (`~/.bashrc` or `~/.bash_profile` or `~/.zshrc`)

Folder descriptions:

- `annotations` : Code for interacting with the annotation framework.
- `example_notebooks`: Examples of how to use the code in this repo. `FANC_Connectomics_General_Intro.ipynb` is a good place to start.
- `images`: Code and info related to the EM image data.
- `lm_em_comparisons`: Code for comparing EM reconstructions with light microscopy data, e.g. color-depth MIP mask searching.
- `segmentation`: Code for interacting with chunkedgraph to access the neuron segmentation data.
- `skeletonization`: Code related to skeletonization.
- `synapses`: Code related to connectivity.
- `transforms`: Code for transforming points between different alignment versions of FANC or between FANC and the Janelia VNC template.
- `volume_meshes`: Neuropil compartment mesh files (work in progress).


Generate an API key via [https://api.zetta.ai/auth/google/login](https://api.zetta.ai/auth/google/login), then create a text file that contains:

    {
      "token": "YOUR API KEY"
    }

and save that file to `~/.cloudvolume/secrets/cave-secret.json`
