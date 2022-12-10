### Generate a color-depth MIP from a FANC neuron

---

##### Prerequisites

Download and install the Color-MIP Generator Fiji plugin by following the instructions in [this README](https://github.com/JaneliaSciComp/ColorMIP_Mask_Search/tree/master/ColorDepthMIP_Generator).

Download VNC template transforms by launching python using an environment with `navis-flybrains` installed (`pip install flybrains` if needed) and running

    import flybrains
    flybrains.download_jrc_vnc_transforms()

Then check that you now have two `.h5` files located at `~/flybrain-data`.

---

##### Make a color MIP for a specific neuron

1. Run `./render_neuron_into_template_space.py [your neuron's segID]`
2. Take the output file, `[your neuron's segID].nrrd`, and open it in Fiji.
3. Follow the instructions in the [Color-MIP generator's README](https://github.com/JaneliaSciComp/ColorMIP_Mask_Search/tree/master/ColorDepthMIP_Generator) to turn your `.nrrd` into a color-depth MIP. The default parameters appear to work well enough, though I haven't tried tuning them at all so perhaps some improvements can be found.
