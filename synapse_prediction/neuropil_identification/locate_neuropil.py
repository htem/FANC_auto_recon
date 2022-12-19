import numpy as np
import pandas as pd
import trimesh
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from time import time
from tempfile import mkdtemp
from shutil import rmtree
from multiprocessing import Pool


# Config
sv_size = np.array([4.3, 4.3, 45])    # supervoxel size in nm
synapse_csv_cols = ['id', 'created', 'deleted', 'superceded_id', 'valid',
                    'pre_pt_position', 'post_pt_position', 'score']


def is_in_neuropil(pos_arr, mesh, pos_units=[1, 1, 1]):
    """
    Given an array of xyz coordinates and a mesh, check whether the
    points are inside the mesh.

    Parameters
    ----------
    pos_arr : np.NDArray[N, 3]
        Positions to check whether they're in the neuropil.
    mesh : trimesh.Trimesh
        Mesh of the neuropil.
    pos_units : list, length 3
         Each 1 unit in pos_arr is this many units in the mesh (in xyz).

    Returns
    -------
    np.NDArray[N]
        Mask indicating whether each point is within mesh.
    """
    pos_arr = pos_arr * np.array(pos_units).reshape([1, 3])
    res_mask = np.zeros((pos_arr.shape[0],), dtype='bool')
    
    # select only points that are at least within the mesh's bounding box
    is_within_bbox = (
        (pos_arr[:, 0] >= mesh.bounds[0, 0]) &
        (pos_arr[:, 0] <= mesh.bounds[1, 0]) &
        (pos_arr[:, 1] >= mesh.bounds[0, 1]) &
        (pos_arr[:, 1] <= mesh.bounds[1, 1]) &
        (pos_arr[:, 2] >= mesh.bounds[0, 2]) &
        (pos_arr[:, 2] <= mesh.bounds[1, 2])
    )
    #print(f'checking only {is_within_bbox.sum()}/{pos_arr.shape[0]} points')

    inner_mask = mesh.contains(pos_arr[is_within_bbox, :])
    res_mask[is_within_bbox] = inner_mask
    return res_mask


def _wrapper(chunk_id, chunk, neuropil_name, mesh, temp_dir):
        """Worker function"""
        start_time = time()
        res = is_in_neuropil(chunk, mesh, sv_size)
        walltime = time() - start_time
        
        tag = f'chunk_{chunk_id:06d}_{neuropil_name}'
        print(f'Finished processing chunk {chunk_id}, {neuropil_name} '
              f'in {walltime:.2f} sec')
        
        with open(temp_dir / f'{tag}.txt', 'w') as f:
            f.write(str(walltime))
        np.save(temp_dir / f'{tag}.npy', res)
        return res


def locate_neuropil(input_file, mesh_dir, output_file=None,
                    chunk_size=10000, procs=1):
    """Given a synapse table dump and a set of meshes, check which
    neuropil(s)/mesh(es) each synapse is in and save the result as
    a parquet file. This is the intended main entrance point.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the CSV synapse table dump (eg. 20221117_fanc_syn.csv).
    mesh_dir : pathlib.Path
        Directory under which meshes are saved. For each synapse,
        for each mesh "<name>.stl" under this directory, this function
        will check whether the synapse is in this region/neuropil.
    output_file : pathlib.Path or None
        If given, the resulting localized synapse table will be saved
        here. Should end with ".parquet".
    chunk_size : int
        The synapse table will be split into chunks, each no more than
        this number of synapses in size.
    procs : int
        Number of processes to spawn.
    
    Returns
    -------
    pandas.DataFrame
        Same as the input CSV synapse table dump, but with extra
        boolean columns "is_in_<name>". The "<name>"s correspond to 
        the filenames (excluding ".stl" at the end) of mesh files
        under `mesh_dir`.
    """
    # Make temporary directory
    temp_dir = Path(mkdtemp(prefix='neuropil_localization_'))

    # Load meshes
    meshes = {}
    for path in sorted(list(Path(mesh_dir).glob('fanc_*.stl'))):
        neuropil = path.name.replace('fanc_', '').replace('.stl', '')
        meshes[neuropil] = trimesh.load_mesh(path)
    
    # Load synapse table
    synapse_df = pd.read_csv(input_file, names=synapse_csv_cols).iloc[:50000]
    # synapse locations are stored as strs, format: "POINTZ(32771 117311 1941)"
    pos_arr = np.array(
        [[int(x) for x in string[7:-1].split()]
         for string in synapse_df['post_pt_position']]
    ).astype('int32')
    del synapse_df    # free some RAM during the execution

    # Define payloads
    # each payload checks, for a chunk of locations, whether each of the
    # points is in one neuropil. Returns a boolean array.
    num_chunks = 1 + (pos_arr.shape[0] - 1) // chunk_size
    print(num_chunks, len(meshes))
    pos_arr_chunk = np.array_split(pos_arr, num_chunks)
    jobs = []
    for chunk_id, chunk in enumerate(pos_arr_chunk):
        for neuropil_name in sorted(meshes):
            spec = (chunk_id, chunk, neuropil_name, meshes[neuropil_name],
                    temp_dir)
            jobs.append(spec)
    print(f'Defined {len(jobs)} jobs')

    # Run in parallel
    with Pool(int(procs)) as p:
        all_res = p.starmap(_wrapper, jobs)

    # Merge & save results
    res_by_chunk = {
        neuropil_name: [None for _ in pos_arr_chunk]
        for neuropil_name in meshes
    }
    for job_config, is_in_neuropil_mask in zip(jobs, all_res):
        chunk_id, _, neuropil_name, _, _ = job_config
        res_by_chunk[neuropil_name][chunk_id] = is_in_neuropil_mask
    synapse_df = pd.read_csv(input_file, names=synapse_csv_cols).iloc[:50000]
    for neuropil_name in meshes:
        concat_mask = np.concatenate(res_by_chunk[neuropil_name])
        synapse_df[f'is_in_{neuropil_name}'] = concat_mask
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        synapse_df.to_parquet(output_file)

    # Cleanup
    rmtree(temp_dir)

    return synapse_df


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(prog='locate_neuropil',
                            description='Identify which neuropil/tract each '
                                        'synapse is in')
    parser.add_argument('input_file',
                        help='Input CSV file listing all synapses')
    parser.add_argument('output_file',
                        help='Output Parquet file identifying the regions')
    parser.add_argument('mesh_dir',
                        help='Path to mesh files')
    parser.add_argument('-c', '--chunk_size', type=int, default=10000,
                        help='Synapses are localized in small chunks. '
                             'Set the chunk size here.')
    parser.add_argument('-p', '--procs', type=int, default=1,
                        help='Number of worker processes')
    args = parser.parse_args()

    locate_neuropil(args.input_file, args.mesh_dir, args.output_file,
                    args.chunk_size, args.procs)