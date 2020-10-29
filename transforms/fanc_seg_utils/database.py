import numpy as np
import csv
import os
import pandas as pd
import csv   
import os
import sys
from . import skeletonization
from . import skeleton_manipulations
from . import neuroglancer_utilities
from . import catmaid_utilities
import numpy as np
import pymaid
from cloudvolume import CloudVolume
from meshparty import trimesh_vtk
from matplotlib import pyplot as plt
import math
import collections
import six
import json
import git
from pathlib import Path

class neuron_database:
    
    ''' Class for keeping track of neurons from the autosegmenatation.'''
    def __init__(self,filename):

        self.filename = Path(filename)
        self.__initialize_database()
        self.target_instance = None
        self.v4_url = 'https://storage.googleapis.com/zetta_lee_fly_vnc_001_segmentation/vnc1_full_v3align_2/realigned_v1/seg/full_run_v1'
        self.dynamic_seg_url = 'https://standalone.poyntr.co/segmentation/table/vnc1_full_v3align_2'  
        self.cloudvolume = None
        self.repo = git.repo.Repo(self.filename.parent)
        
        # For now, hardcode all resolutions.  If you initialize a catmaid instance, it will at least set it programatically. 
        self.target_resolution = np.array([4.3,4.3,45])
        self.dynamic_seg_res = np.array([17.2,17.2,45])
        self.v4_res = np.array([4.3,4.3,45])





    def __initialize_database(self):
        fileEmpty =  os.path.exists(self.filename)
        if not fileEmpty:
            df = pd.DataFrame(data = None, columns=['Segment_ID','Name','V4_Soma','V3_Soma','Annotations','Extra_Segment_IDs','Catmaid_Skids'])
            df.to_csv(self.filename,index=False)
            print('Database created')
        else:
            print('Database active')

    
    def __serialize_coords(self,x):
        if isinstance(x,list):
            return(json.dumps(x))
        else:
            return(json.dumps(np.ndarray.tolist(x)))
            
    def __serialize_annotations(self,x):
        return(json.dumps(x))
    
    def __deserialize_cols(self,x):
        try:
            x_d = json.loads(x)
        except:
            x_d = x
        
        return(x_d)
        
    
    
    

    def __check_seg_id(self,seg_id):
        self.get_database()
        df = self.neurons
        for i in df.Segment_ID:
            if seg_id == i:
                return(True)

        return(False)     


    def add_entry(self,
              soma_coord, 
              Name = None, 
              Annotations = None, 
              override = False,
              extra_segids = None):
        ''' Add a database entry.
        Parameters
        -----------
        soma_coord:    np.array, mip0 pixel coordinates.
        Name:          str, name for neuron.
        Annotations:   list,str, annotations to add to neuron.
        override:      Bypass check for existing entry. Default=False
        extra_segids:  int,list extra segment_ids in case the neuron is in pieces. This should be unnecessary once the dynamic seg layer works.'''

        updated = self.__update_db(soma_coord,
                                 Name=Name, 
                                 Annotations=Annotations,
                                 override=override,
                                 extra_segids = extra_segids)   
        if updated is True:
            print('entry added')
        else:
            print('entry already exists')
    

    def __update_db(self,
          v4_pt, 
          Name=None, 
          Annotations = None, 
          override=False,
          extra_segids = None):
        
        filename = self.filename
        seg_id = neuroglancer_utilities.seg_from_pt(v4_pt)
        v3_transform = neuroglancer_utilities.fanc4_to_3(v4_pt,scale=2)
        v3_pt = [v3_transform['x'],v3_transform['y'],v3_transform['z']]
        
        v4_pt_s = self.__serialize_coords(v4_pt)
        v3_pt_s = self.__serialize_coords(v3_pt)
        
        if Annotations is not None:
            a_s = self.__serialize_annotations(Annotations)
        else:
            a_s = None
        
        if extra_segids is not None:
            e_s = self.__serialize_coords(extra_segids)
        else:
            e_s = self.__serialize_annotations([''])
        

        if self.__check_seg_id(seg_id) is False or override is True:
            df = pd.DataFrame([{'Segment_ID':seg_id, 'Name': Name, 'V4_Soma':v4_pt_s, 'V3_Soma':v3_pt_s,'Annotations':a_s,'Extra_Segment_IDs':e_s,'Catmaid_Skids':''}])
            df.to_csv(filename, mode='a', header=False,index=False, encoding = 'utf-8')

            
            self.repo.index.add(self.filename.as_posix())
            self.repo.index.commit('Added neuron:{}'.format(seg_id))
            return(True) 
        else:
            return(False)   


        
        
        
    def __is_iter(self,x):
        
        if isinstance(x, collections.Iterable) and not isinstance(x, (six.string_types, pd.DataFrame)):
            return True
        else:
            return False



        
        
    def get_cloudvolume(self,vol_url = None):
        if vol_url is None:
            vol_url = self.v4_url

        self.cloud_volume = CloudVolume(vol_url)





    def get_database(self):
        ## TODO: Add update seg_ids to this. 
        self.neurons = pd.read_csv(self.filename,converters={'V3_Soma':self.__deserialize_cols,
                                                            'Annotations':self.__deserialize_cols,
                                                            'Extra_Segment_IDs':self.__deserialize_cols,
                                                            'Catmaid_Skids':self.__deserialize_cols})
    
    
    
    def save_database(self,comment=None):

        if not hasattr(self,'neurons'):
            self.get_database()

        df = self.neurons
        for i in range(len(df.Annotations)):
            df.loc[i,'Annotations'] = json.dumps(df.loc[i,'Annotations'])

        df.to_csv(self.filename,index=False, encoding = 'utf-8')

    
        self.repo.index.add(self.filename.as_posix())
        if self.repo.is_dirty(): 
            self.repo.index.commit(comment)
            print('Database Updated')
        else:
            print('No change made')






    def get_annotations(self,x):
        ''' Get annotations for a given entry.
        Parameters
        ----------
        x: int,str, Either a segment_id or a name.
        
        Returns
        ----------
        annotations: dict, Dictionary of annotations with key equal to input (name or seg_id).'''
        
        if not hasattr(self,'neurons'):
            self.get_database()
    
        annotations = {}
        if not self.__is_iter(x):
            x = [x]
            
        for i in x:
            if isinstance(i,(int,np.int64)):
                column = 'Segment_ID'
            elif isinstance(i,str):
                column = 'Name'
        
            df = self.neurons
            for index, row in df.iterrows():
                if str(i) in str(row[column]):
                    annotations[int(i)] = row.Annotations

        return(annotations)
    
    
    
    
    def add_annotations(self,x,annotations):
        ''' Add annotations to an entry
        Parameters
        ----------
        x:            int, Segment ID 
        annotations:  list or str, annotations to add. Will not add redundant annoations.'''


        self.get_database()
        df = self.neurons
        if not self.__is_iter(annotations):
            annotations = [annotations]

        if isinstance(x,(int,np.int64)):

            [df.loc[df.Segment_ID == x,'Annotations'].values[0].append(i) for i in annotations if i not in df.loc[df.Segment_ID == x,'Annotations'].values[0]]
            print('Annotations added')

        self.save_database(comment = str(df.Name[df.Segment_ID == x].values[0]) + ':' + 'annotation_update')





    def update_segIDs(self,seg_id=None):
        ## Doesn't work. Need to figure out how to interact with dynamic seg layer. 
        ##TODO: Make this work. 
        new_seg_ids = []

        self.get_database()
        df = self.neurons

        for index, row in df.iterrows():
            row.Segment_ID = neuroglancer_utilities.seg_from_pt(row.V4_Soma,vol_url=self.dynamic_seg_url,seg_mip = self.dynamic_seg_res)

        self.save_database(comment = 'Segment_ID_Update')
        print('Segment IDs updated')





    def add_segIDs(self,x,extra_IDs):
        # Work around until proofreading is online. If neuron is in pieces, add all the segment IDs, and .get_skeletons will fuse them. 
        self.get_database()
        df = self.neurons

        if isinstance(x,(int,np.int64)):
            comment_base = str(df.Name[df.Segment_ID == x].values[0])
            df.loc[df.Segment_ID == x,'Extra_Segment_IDs'] = self.__serialize_coords(extra_IDs)
            print('Extra IDs added')

        elif isinstance(x,str):
            comment_base = x
            df.loc[[x in n for n in df.Name],'Extra_Segment_IDs'] = self.__serialize_coords(extra_IDs)
            print('Extra IDs added')

        self.save_database(comment = comment_base + ':' + 'annotation_update')

        




    def get_mesh(self,x,vol_url=None):

        if vol_url is None:
            vol_url = self.v4_url

        vol = CloudVolume(vol_url)
        self.get_database()
        df = self.neurons
        
        meshes = []
        for i in range(len(x)):
            if isinstance(x[i],(int,np.int64)):
                seg_id = x[i]
                extra_segids = df.loc[df.Segment_ID == x[i],'Extra_Segment_IDs'].values[0]

            elif isinstance(x[i],str):
                seg_id = [df.loc[[x[i] in n for n in df.Name],'Segment_ID']]
                extra_segids = df.loc[[x[i] in n for n in df.Name],'Extra_Segment_IDs'].values[0]

            if len(extra_segids) > 0:
                mesh = vol.mesh.get(seg_id+extra_segids,remove_duplicate_vertices = True, fuse = True)
            else:
                mesh = vol.mesh.get(seg_id,remove_duplicate_vertices=True,fuse=True)
            
            meshes.append(mesh)

        return meshes





    def plot_mesh(self,x,vol_url=None,save=False,output_dir=None):
        if vol_url is None:
                vol_url = self.v4_url
        
        if not self.__is_iter(x):
            x = [x]

        mesh = self.get_mesh(x,vol_url = vol_url)
        
        colors = plt.cm.Set1(range(len(mesh)))
        
        tmeshes = []
        
        
        for i in range(len(mesh)):
            tmeshes.append(trimesh_vtk.mesh_actor(mesh[i],vertex_colors=colors[i,0:3],face_colors=colors[i,0:3]))
        
   
        
        if save is True:
            trimesh_vtk.render_actors(tmeshes,do_save=True,filename=output_dir + str(x) + '.png')
        else:
            trimesh_vtk.render_actors(tmeshes)   

        



    def get_skeletons(self,x,
                     vol_url = None,
                     method = 'kimimaro',
                     transform = True,
                     cache_path = None,
                     output = 'pymaid',
                     save = False,
                     save_path = None):
        
        if not self.__is_iter(x):
            x = [x]
        
  
        
        if len(x) > 1:
            nlist = []
            for i in range(len(x)):
                nlist.append(self.__get_skeleton(x[i], 
                         vol_url = vol_url,
                         method = method,
                         transform = transform,
                         cache_path = cache_path,
                         output = output))
            
        
        else:
            nlist =  self.__get_skeleton(x[0], 
                         vol_url = vol_url,
                         method = method,
                         transform = transform,
                         cache_path = cache_path,
                         output = output)
            
        neuron_list = pymaid.CatmaidNeuronList(nlist)
        if save is True:
            for i in neuron_list:
                pymaid.to_swc(i,filename = save_path + i.neuron_name[0][0])
            
            
        return(neuron_list) ## TODO Add Navis output 
            
                                 
            



    def __get_skeleton(self,x,
                     vol_url = None,
                     method = 'Kimimaro',
                     transform = True,
                     cache_path = None,
                     output = 'pymaid'):

        if vol_url is None:
            vol_url = self.v4_url
        
        self.get_database()
        df = self.neurons
        # If input is a segment id:
        if isinstance(x,(int,np.int64)):
            seg_id = x
            extra_segids = df.loc[df.Segment_ID == x,'Extra_Segment_IDs'].values[0]
            name = df.loc[df.Segment_ID == x,'Name']
            soma_coords = df.V3_Soma[df.Segment_ID == x]
        
        # If input is a name:
        elif isinstance(x,str):
            name = x
            seg_id = [df.loc[[x in n for n in df.Name],'Segment_ID']]
            extra_segids = df.loc[[x in n for n in df.Name],'Extra_Segment_IDs'].values[0]
            soma_coords = df.V3_Soma[df.Segment_ID == x]
        
        #TODO: Add get by annotations
        
        # If there are extra segment IDs attached to the entry, append them, and fuse them together. 
        # This really should be unnecessary once proofreading is online (read: I know how to interact with it)
        if len(extra_segids) > 0: 
            seg_id = [seg_id] + extra_segids
            fuse = True
        else:
            fuse = False

        ## TODO: Update this for use with dynamic seg. Check segment IDs before appending to anntations, also add a timestamp. 
        annotations = self.get_annotations(x)
        annotations[x] = annotations[x] + ['FANC4_ID: ' + str(seg_id)]



        skeleton = skeletonization.get_skeleton(seg_id,
                                     vol_url,
                                     method=method,
                                     transform=transform,
                                     cache_path=cache_path,
                                     annotations=annotations[x],
                                     name=name,
                                     output=output,
                                     fuse = fuse)
        
        skeleton = skeleton_manipulations.set_soma(skeleton,np.array(soma_coords.iloc[0])* self.target_res) 
        pymaid.downsample_neuron(skeleton,resampling_factor=15,inplace=True)
        skeleton = skeleton_manipulations.diameter_smoothing(skeleton,smooth_method='smooth',smooth_bandwidth=1000)
        return(skeleton)

    
    
    
    
    
    
    def initialize_catmaid_instance(self,keys_path, project_id):
        
        target_instance = catmaid_utilities.catmaid_login('fanc',project_id,keys_path)
        self.target_instnace = target_instance
        xyz = [self.target_instnace.image_stacks.resolution.values[0][k] for k in self.target_instnace.image_stacks.resolution.values[0].keys()]
        self.target_res = np.array(xyz)
        
    
    
    
    
    
    def upload_to_catmaid(self, x = None, target_instance = None, only_good = True ):
        
        ''' Upload neurons in database to a CATMAID Instance. Will update the database with a serialized dictionary with key:value as project_id:skid.
        
            Parameters
            ----------
            x :                 int, segment ID to upload
            target_instance:    a pymaid catmaid instance. Default is instnace associated with the database, but you need to run .initialize_catmaid_instnace first.        
            only_good:          bool. Flag for only uploading neurons that have been annotated with 'good' indicating they are more or less complete.
            
            ##TODO: Add params for adjusting skeletonization 
            ##TODO: Add inputs other than segment_id
            ##TODO: Make a flexible way to update catmaid skeletons. Implement Philip's code here probably. 
            ##TODO: Change order so that multiple skeleton uploads happen one at a time rather than download all skeletons, then upload all skeletons. Right now the check for a skeleton existing in a catmaid instance is after it is downloaded. Need to fix this. 
            ##TODO: Make better commit comment. 
            '''
        if target_instance is None:
            target_instance = self.target_instnace
    
            
        
        if x is None:
            
            if not hasattr(self,'neurons'):
                self.get_database()
            
            all_neurons = self.neurons.Segment_ID.to_list()    
            if only_good is True:
                to_upload = []
                for i in all_neurons:
                    an = self.get_annotations(i)
                    if 'good' in an[i]:
                        to_upload.append(i)
            else:
                to_upload = all_neurons
        else:
            to_upload = [x]
            
 
        
        
        skeletons = self.get_skeletons(to_upload)
        
        if not 'Catmaid_Skids' in self.neurons:
            self.neurons['Catmaid_Skids'] = ''
            
        idx = 0
        updated = 0
        for i in skeletons:
            potential_sk = self.neurons.loc[self.neurons.Segment_ID == to_upload[idx],'Catmaid_Skids']
            if isinstance(potential_sk.iloc[0],dict):
                skids = potential_sk.iloc[0]
                
                if str(self.target_instnace.project_id) in skids.keys():
                    print('Neuron Exists in This Project')
                else:
                    i.annotations.append(i.meta_data['skeleton_type'])
                    i.annotations.append(self.filename.name[0:self.filename.name.rfind('.')])
                    upload_data = catmaid_utilities.upload_to_CATMAID(i,target_project=target_instance)
                    self.neurons.loc[self.neurons.Segment_ID == to_upload[idx],'Catmaid_Skids'] = json.dumps({target_instance.project_id:upload_data['skeleton_id']})
                    updated = 1
            else:
                i.annotations.append(i.meta_data['skeleton_type'])
                upload_data = catmaid_utilities.upload_to_CATMAID(i,target_project=target_instance)
                self.neurons.loc[self.neurons.Segment_ID == to_upload[idx],'Catmaid_Skids'] = json.dumps({target_instance.project_id:upload_data['skeleton_id']})
                updated = 1
             
            idx+=1
            
        if updated > 0:
            ## TODO: Fix this to be more specific. 
            self.save_database(comment = 'catmaid_skids_added')
            
        
        
        
