# FANC_auto_recon

FANC (pronounced "fancy") is the Female Adult Nerve Cord, a GridTape-TEM dataset of an adult _Drosophila melanogaster_'s ventral nerve cord. The dataset was first published in [Phelps, Hildebrand, Graham et al. 2021 _Cell_](https://www.lee.hms.harvard.edu/phelps-hildebrand-graham-et-al-2021), after which we applied automated methods for reconstructing neurons, synapses, and nuclei to accelerate reconstruction of the ventral nerve cord connectome. This repository contains a python package for interacting with the connectome (see the folder [`fanc`](fanc)) as well as other supporting files. 

Have any questions? Please [open an issue](https://github.com/htem/FANC_auto_recon/issues/new) or contact [Jasper Phelps (jasper.s.phelps@gmail.com)](https://github.com/jasper-tms).

## Installing and configuring the `fanc` python package

### Before you start

As is always the case in python, consider making a virtual environment (using your preference of virtualenv/virtualenvwrapper or conda) before installing.

### Installation option 1: pip install from PyPI

    pip install fanc-fly

### Installation option 2: pip install directly from GitHub
The code on GitHub will sometimes be slightly more up to date than the version on PyPI

    pip install git+https://github.com/htem/FANC_auto_recon.git

### Installation option 3: Clone then install
This is the best option if you want to make changes yourself to the code

    cd ~/repos  # or wherever you keep your repos
    git clone https://github.com/htem/FANC_auto_recon.git
    cd FANC_auto_recon
    pip install -e .

### Provide credentials

Access to the latest reconstruction of FANC is restricted to authorized users. If you are a member of the FANC community (see [Collaborative community](../../wiki#collaborative-community) on this repo's wiki) and have been granted access, you can generate an API key by visiting [https://global.daf-apis.com/auth/api/v1/create_token](https://global.daf-apis.com/auth/api/v1/create_token) and logging in with your FANC-authorized google account. Copy the key that is displayed, then run the following commands in python to save your key to the appropriate file:
```python
import fanc
fanc.save_cave_credentials("THE API KEY YOU COPIED")
```

Alternatively, you can manually do what the command above accomplishes, which is to create a text file at `~/.cloudvolume/secrets/cave-secret.json` with these contents:

    {
      "token": "THE API KEY YOU COPIED",
      "fanc_production_mar2021": "THE API KEY YOU COPIED"
    }

You can verify that your API key has been saved successfully by running:
```python
import fanc
client = fanc.get_caveclient()
```

### Optional installation steps for additional functionality

#### Install Elastix to transform neurons into alignment with the VNC template
The mesh manipulation and coordinate transform code requires `pytransformix`, which is itself a Python wrapper for Elastix. Therefore, Elastix must be installed and its lib and bin paths must be appended to the `LD_LIBRARY_PATH` and `PATH` environment variables. See [`pytransformix` documentation](https://github.com/jasper-tms/pytransformix#installation) for specific instructions.

#### Provide CATMAID credentials to pull data from CATMAID
You can get your CATMAID API key by logging into https://radagast.hms.harvard.edu/catmaidvnc then hovering over "You are [Your Name]" in the top-right corner, then clicking "Get API token".

Save your CATMAID API key by running:
```python
import fanc
fanc.catmaid.save_catmaid_credentials("YOUR CATMAID API KEY")
```

You can verify that your API key has been saved successfully by running:
```python
import fanc
fanc.catmaid.connect()
```

## Documentation
- First go through [`fanc_python_package_examples.ipynb`](https://github.com/htem/FANC_auto_recon/blob/main/example_notebooks/fanc_python_package_examples.ipynb)
- Then check out other notebooks in [`example_notebooks/`](https://github.com/htem/FANC_auto_recon/tree/main/example_notebooks)
- Finally you can [browse the code](https://github.com/htem/FANC_auto_recon/tree/main/fanc), check out modules that have names that interest you, and read the docstrings.
