You can download neuropil region meshes and tract meshes in .obj format through VFB by going [here](https://v2.virtualflybrain.org/org.geppetto.frontend/geppetto?id=Court2020&i=VFB_00200000) using the left and right arrows to scroll through the available meshes, then clicking on the one I want to load it into the 3D viewer, then scrolling down to the `Downloads` section and downloading the `Mesh/Pointcloud(OBJ)`.

(Unfortunately, these are aligned to the unisex template, not the female template. I'm asking Rob at VFB to try to get ones aligned to the Female template.)

Once downloaded, I opened the .obj file in meshlab, deleted any connected components that weren't attached to the main mesh, scaled things up by 1000x to go from microns to nm, then saved as an .stl with binary encoding. (I only know how to open .stl files in python, via the numpy-stl package, but if you know of ways to open other types, feel free to work with a format you're most familiar with.) Then, I warped the mesh from `JRC2018_VNC_UNISEX` space to FANC space by running:

    ./warp_mesh_to_FANC.py "JRC2018_VNC_UNISEX/wing tectulum_cleaned_1000x.stl" JRC2018_VNC_UNISEX_to_FANC
