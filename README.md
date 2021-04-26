# FANC_auto_recon
- `annotations` : For code related to interacting with the annotation framework.
- `images`: For code related to image upload/download.
- `segmentation`: For code related to interacting with/authenticating chunkedgraphs.
- `skeletonization`: For code related to skeletonization.
- `synapses`: For code related to connectivity.
- `transforms`: For code related to rootID lookup and alignment transforms.
- `syn_and_seg`: For code that combines automated neuron segmentation with automated synaptic partner predictions to build connectomes.

Generate an API key via [https://api.zetta.ai/auth/google/login](https://api.zetta.ai/auth/google/login), then create a text file that contains:

    {
      "token": "YOUR API KEY"
    }

and save that file to `~/.cloudvolume/secrets/cave-secret.json`
