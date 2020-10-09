from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox
from tifffile import imsave

path = 'https://storage.googleapis.com/zetta_lee_fly_vnc_001_precomputed/vnc1_full_v3align_2/realigned_v1'
mip = 0
vol = CloudVolume(path, mip=mip, fill_missing=True)
bounds       = vol.bounds                   # Bbox of volume extent
offset       = vol.voxel_offset             # bounds.minpt
chunk_size   = vol.underlying               # Vec of chunk dimensions
chunk_counts = bounds.size3() // chunk_size # Vec with no. of chunks

for z in range(bounds.minpt.z, bounds.maxpt.z+1):
    for x in range(chunk_counts.x):
        for y in range(chunk_counts.y):
            chunk_start = chunk_size*Vec(x,y,1) + offset
            bbox = Bbox(chunk_start, chunk_start+chunk_size)
            img = vol[bbox.to_slices()][:,:,0,0]
            fn = '{}_{}_{}.tif'.format(x,y,z)
            imsave(fn, img)
