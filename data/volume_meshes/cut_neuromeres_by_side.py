"""
Run source mesh_env_setup.sh first!
"""

import stl
import navis
import trimesh
import flybrains
from pathlib import Path


# Configs
midline = 329 * 0.4    # 0.4um per voxel
unisex_template_dir = Path(
    'JRC2018_VNC_UNISEX/'
)
output_dir = Path(
    'JRC2018_VNC_UNISEX_to_FANC/meshes_by_side'
)
output_dir.mkdir(parents=True, exist_ok=True)
uncut_neuromere_mesh_paths = {
    f'{path.name.split()[0]}': path
    for path in unisex_template_dir.glob('*.stl')
}
whole_vnc_mesh = 'VFB_00200000'

# Download template mapping transform
flybrains.download_jrc_vnc_transforms()


# Load meshes with trimesh, split in halves, save
uncut_neuromere_meshes = {
    k: trimesh.load_mesh(v)
    for k, v in uncut_neuromere_mesh_paths.items()
}
vnc_bounds = uncut_neuromere_meshes[whole_vnc_mesh].bounds[1, :]
_, y_size, z_size = vnc_bounds.astype('int') + 1

# arguments are: edge lengths, and homogeneous transformation matrix (4x4)
# for the box center. See link below under "Translation" for how this works
# http://www.it.hiof.no/~borres/j3d/math/threed/p-threed.html
left_box = trimesh.creation.box(
    [midline, y_size, z_size],
    [[1, 0, 0, midline * 0.5],
     [0, 1, 0, y_size * 0.5],
     [0, 0, 1, z_size * 0.5],
     [0, 0, 0, 1]]
)
right_box = trimesh.creation.box(
    [midline, y_size, z_size],
    [[1, 0, 0, midline * 1.5],
     [0, 1, 0, y_size * 0.5],
     [0, 0, 1, z_size * 0.5],
     [0, 0, 0, 1]]
)

cut_meshes = {}
for k, mesh in uncut_neuromere_meshes.items():
    if k == whole_vnc_mesh:
        continue    # this is the whole VNC
    cut_meshes[f'{k}_L'] = mesh.intersection(left_box)
    cut_meshes[f'{k}_R'] = mesh.intersection(right_box)

for k, mesh in cut_meshes.items():
    mesh.export(output_dir / f'unisex_template_{k}.stl')


# Reload with numpy-stl, transform to FANC space, and save
for k in cut_meshes:
    template_space_mesh_path = output_dir / f'unisex_template_{k}.stl'
    mesh = stl.mesh.Mesh.from_file(template_space_mesh_path)
    mesh.v0 = navis.xform_brain(mesh.v0, source='JRCVNC2018U', target='FANC')
    mesh.v1 = navis.xform_brain(mesh.v1, source='JRCVNC2018U', target='FANC')
    mesh.v2 = navis.xform_brain(mesh.v2, source='JRCVNC2018U', target='FANC')
    # swap left/right because FANC and the template have different z orders
    assert k[-2:] in ('_R', '_L'), 'unexpected mesh name'
    if k[-2:] == '_R':
        k = k[:-2] + '_L'
    else:
        k = k[:-2] + '_R'
    mesh.save(output_dir / f'fanc_{k}.stl')
    template_space_mesh_path.unlink()
