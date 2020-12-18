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

class Neuron_database:
    
    ''' Class for keeping track of neurons from the autosegmenatation.'''
    def __init__(self,filename,segmentation_version = 'V4_dynamic'):
        
        self.filename = Path(filename)
        self.Name = self.filename.name[0:self.filename.name.rfind('.')]
        self.target_instance = None
        self.cloud_volume = None
        self.target_resolution = np.array([4.3,4.3,45])
        self.segmentation_resolution = None
        self.segmentation_resolutions = {'V3': np.array([4,4,40]), 'V4': np.array([4.3,4.3,45]), 'V4_dynamic': np.array([17.2,17.2,45])}
        self.segmentations = {'V3':'https://storage.googleapis.com/zetta_lee_fly_vnc_001_segmentation_temp/vnc1_full_v3align_2/37674-69768_41600-134885_430-4334/seg/v3',                                 'V4':'https://storage.googleapis.com/zetta_lee_fly_vnc_001_segmentation/vnc1_full_v3align_2/realigned_v1/seg/full_run_v1',
                              'V4_dynamic': 'graphene://https://standalone.poyntr.co/segmentation/table/vnc1_full_v3align_2',
                              'V4_brain_regions': 'https://storage.googleapis.com/zetta_lee_fly_vnc_001_precomputed/vnc1_full_v3align_2/brain_regions'}  
       
    
        self.segmentation_version = segmentation_version
        self.__initialize_database()
 
        if self.__check_repo() is True:
            self.repo = git.repo.Repo(self.filename.parent)
        
        
    def __initialize_database(self):
        fileEmpty =  os.path.exists(self.filename)
        if not fileEmpty:
            df = pd.DataFrame(data = None, 
                              columns =  ['Segment_ID',
                                          'Name',
                                          'V4_Soma',
                                          'V3_Soma',
                                          'Annotations',
                                          'Catmaid_Skids'])
            df.to_csv(self.filename,index=False)
            print('Database created')            
            self.get_cloudvolume(self.segmentations[self.segmentation_version])

        else:
            self.get_cloudvolume(self.segmentations[self.segmentation_version])
            self.get_database()
            print('Database active')
    
    
    
    # Version control things    
    def __check_repo(self):
        try:
            git.Repo(self.filename.parent)
            return(True)
        except:
            print('Warning: Database is not in a repo, changes not logged')
            return(False)

    #Set which segmentation to use
    def set_segmentation(self,version):
        self.segmentation_version = version
        self.get_cloudvolume(vol_url=self.segmentations[self.segmentation_version])
        self.segmentation_resolution = self.segmentation_resolutions[self.segmentation_version]
        self.update_segIDs()
        return('Segmentation version updated')
    
    
    # Serialize / Deserialize lists for coords/annotations
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
        
    
    
    # Check if things are iterable. 
    def __is_iter(self,x):

        if isinstance(x, collections.Iterable) and not isinstance(x, (six.string_types, pd.DataFrame)):
            if isinstance(x,dict) and len(x) == 1:
                return False
            else:
                return True
        else:
            return False 

        
    def get_database(self):
         self.neurons  = pd.read_csv(self.filename,converters={'V4_Soma':self.__deserialize_cols,
                                                            'V3_Soma':self.__deserialize_cols,
                                                            'Annotations':self.__deserialize_cols,
                                                            'Catmaid_Skids':self.__deserialize_cols})

        

    def save_database(self,comment='Update'):

        if not hasattr(self,'neurons'):
            self.get_database()

        df = self.neurons.copy(deep=True)
        for index, row in df.iterrows():
            df.loc[index,'Annotations'] = self.__serialize_annotations(row.Annotations)
            df.loc[index,'V4_Soma'] = self.__serialize_coords(row.V4_Soma)
            df.loc[index,'V3_Soma'] = self.__serialize_coords(row.V3_Soma)

        
        df.to_csv(self.filename,index=False, encoding = 'utf-8')

    
        if self.__check_repo() is True:
            self.repo.index.add(self.filename.as_posix())
            if self.repo.is_dirty(): 
                self.repo.index.commit(comment)
                print('Database Updated')
            else:
                print('No change made')
        else:
            print('Database Updated')

    
    
    def get_entries(self,input_var):
        ''' Get entries by seg id or by name.
        Args: 
            input_var: int, str, list   Either a seg_id (int), name (str), or list of either.
        Returns:
            subset: pd.DataFrame.    Rows corresponding to input criteria '''
        df = self.neurons.copy(deep=True)
        
        if self.__is_iter(input_var):
            input_type = type(input_var[0])
        else:
            input_type = type(input_var)
            input_var = [input_var]

        if input_type==int:
            df.set_index('Segment_ID',inplace=True)
            subset = df.loc[input_var]


        elif input_type == str:
            df.set_index('Name',inplace=True)
            subset = df.loc[input_var]

        else:
            raise ValueError('Incorrect input type')
        
        
        subset.reset_index(inplace=True)

        return(subset)
    
    
    def add_entry(self,
              soma_coord, 
              Name = None, 
              Annotations = None,
              override = False):
        ''' Add a database entry.
            Args
            -----------
            soma_coord:    np.array, mip0 voxel coordinates, or in downsampled voxel coords for the dynamic segmentation.
            Name:          str, name for neuron.
            Annotations:   list,str, annotations to add to neuron.
            override:      Bypass check for existing entry. Default=False

            ## TODO: Use class specific get point with more flexible call to seg_from_pt'''

        updated = self.__update_db(soma_coord,
                                 Name=Name, 
                                 Annotations=Annotations,
                                 override=override)   
        if updated is True:
            print('entry added')
        else:
            print('entry already exists')
    

    def __update_db(self,
          v4_pt, 
          Name=None, 
          Annotations = None, 
          override=False,
          ):
        
        filename = self.filename
        
        seg_id = self.seg_from_pt([v4_pt])
        seg_id = int(seg_id[0])
       
        # Make sure that the v4 pt is at the correct resolution. The dynamic segmetation 4x downsampled, so need to adjust. 
        scale_factor = self.segmentation_resolution / self.target_resolution
        v4_pt_scaled = v4_pt #* scale_factor
        
        v3_transform = neuroglancer_utilities.fanc4_to_3(v4_pt_scaled,scale=2)
        v3_pt = [v3_transform['x'],v3_transform['y'],v3_transform['z']]
        
        v4_pt_s = self.__serialize_coords(v4_pt)
        v3_pt_s = self.__serialize_coords(v3_pt)
        
        if Annotations is not None:
            a_s = self.__serialize_annotations(Annotations)
        else:
            a_s = None
        

        

        if self.__check_seg_id(seg_id) is False or override is True:
            df = pd.DataFrame([{'Segment_ID':seg_id, 
                                'Name': Name, 
                                'V4_Soma':v4_pt_s, 
                                'V3_Soma':v3_pt_s,
                                'Annotations':a_s,
                                'Catmaid_Skids':''}])
            
            df.to_csv(filename, mode='a', header=False,index=False, encoding = 'utf-8')

            if self.__check_repo() is True:
                self.repo.index.add(self.filename.as_posix())
                self.repo.index.commit('Added neuron:{}'.format(seg_id))
                
            return(True) 
        else:
            return(False)   
        
    
    ## Dealing with annotations
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
        
        subset = self.get_entries(x)
        
        return(dict(zip(subset.Segment_ID.values,subset.Annotations.values)))
            
    
     
    def add_annotations(self,x,annotations):
        ''' Add annotations to an entry
        Parameters
        ----------
        x:            int,str,list Segment ID or name or list of either.
        annotations:  list or str, annotations to add. Will not add redundant annoations.'''


        self.get_database()
        df = self.neurons
        
        if not self.__is_iter(annotations):
            annotations = [annotations]
            
        subset = self.get_entries(x)
        seg_ids = subset.Segment_ID
        
        for i in seg_ids:
            [df.loc[df.Segment_ID == i,'Annotations'].values[0].append(j) for j in annotations if j not in df.loc[df.Segment_ID == i,'Annotations'].values[0]]
            self.save_database(comment = str(df.Name[df.Segment_ID == i].values[0]) + ':' + 'annotation_update')
            

        print('Annotations added')

        
                   
    def remove_annotations(self,x,annotations):
        self.get_database()
        df = self.neurons
        
        if not self.__is_iter(annotations):
            annotations = [annotations]
               
        current_annotations_dict = self.get_annotations(x)
        
        subset = self.get_entries(x)
        seg_ids = subset.Segment_ID
        
        change = 0
        affected = []
        for entry in seg_ids:
            current_annotations = current_annotations_dict[entry]
            for annotation in annotations:
                if annotation in current_annotations:
                    self.neurons.loc[self.neurons.Segment_ID==entry,'Annotations'].values[0].remove(annotation)
                    affected.append(subset.loc[subset.Segment_ID == entry,'Name'])
                    change += 1
            
        if change > 0:
            self.save_database(comment = str( str(affected) + ':' + 'annotation_update'))
            print('Annotation Removed')
        else:
            print('Annotation does not exist')
        
    
     
    
    # Check if ID is in the database. 
    def __check_seg_id(self,seg_id):
        self.get_database()
        df = self.neurons
        for i in df.Segment_ID:
            if seg_id == i:
                return(True)

        return(False)     

    def get_cloudvolume(self,vol_url = None):
        if vol_url is None:
            vol_url = self.segmentations[self.segmentation_version]

        if 'graphene' in vol_url:
            print('Dynamic Segmentation Enabled')
            self.cloud_volume = CloudVolume(vol_url,use_https=True,agglomerate=True)
        else:
            self.cloud_volume = CloudVolume(vol_url)
        self.segmentation_resolution = self.cloud_volume.scale['resolution']


    
    # Get a rootID / segmentID from a mip0 coordinate
    def seg_from_pt(self,pt,segmentation_version=None):            
        
        if segmentation_version is None:
            segmentation_version = self.segmentation_version
            
     
        return(neuroglancer_utilities.seg_from_pt(pt,
                                                  vol = self.cloud_volume,
                                                  image_res = np.array([4.3,4.3,45])))
        


        
    def update_segIDs(self):
        
        if 'dynamic' in self.segmentation_version:
            self.get_database()
            
            for index,row in self.neurons.iterrows():
                self.neurons.loc[index,'Segment_ID'] = self.seg_from_pt(row.V4_Soma)
            
            self.save_database(comment='SEG_ID_UPDATE')
            print('Segment IDs updated')
        
        else:
             for index,row in self.neurons.iterrows():
                self.neurons.loc[index,'Segment_ID'] = self.seg_from_pt(row.V4_Soma)
            
             print('dynamic segmentation not enabled, retreived flat seg_ids')
        
        




    def get_mesh(self,x,vol_url=None):
        ## TODO: UPDATE input parsing
        
        if vol_url is None:
            vol_url = self.segmentations[self.segmentation_version]
            
        if 'graphene' in vol_url:  
            vol = CloudVolume(vol_url,use_https=True,agglomerate=True)
        else:
            vol = CloudVolume(vol_url)
            

        df = self.neurons
        
        meshes = []
        if not self.__is_iter(x):
            x = [x]
        
        for i in range(len(x)):
            if isinstance(x[i],(int,np.int64)):
                seg_id = x[i]
                
                

            elif isinstance(x[i],str):
                seg_id = [df.loc[[x[i] in n for n in df.Name],'Segment_ID']]
                
                
     
            mesh = vol.mesh.get(seg_id,remove_duplicate_vertices=True,fuse=True)
            
            meshes.append(mesh)
            
        if len(meshes) == 1:
            return(meshes[0])
        else:
            return(meshes)






    def plot_mesh(self,x,vol_url=None,plot_neuropil=False,neuropil_url=None,save=False,output_dir=None,opacity=.5):
        ## TODO UPDATE INPUT PARSING
        if vol_url is None:
                vol_url = self.segmentations[self.segmentation_version]
        
        # Currently not functional
        if plot_neuropil is True and neuropil_url is None:
                np_url = self.segmentations['V4_brain_regions']
        
        
        if not self.__is_iter(x):
            x = [x]

        mesh = []
        for i in x:
            mesh.append(self.get_mesh(i,vol_url = vol_url))
        
        #np_mesh = self.get_mesh(2,vol_url = np_url)
        
        colors = plt.cm.Set1(range(len(mesh)))
        
        tmeshes = []
        
        
        for i in range(len(mesh)):
            tmeshes.append(trimesh_vtk.mesh_actor(mesh[i],vertex_colors=colors[i,0:3],face_colors=colors[i,0:3],opacity=opacity))
        
        if plot_neuropil is True:
            tmeshes.append(trimesh_vtk.mesh_actor(np_mesh,vertex_colors=[0,0,1],face_colors=[0,0,1],opacity=0.1))
        
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
            vol_url = self.segmentations[self.segmentation_version]
        

        df = self.neurons
        # If input is a segment id:
        if isinstance(x,(int,np.int64)):
            seg_id = x
            name = df.loc[df.Segment_ID == x,'Name']
            soma_coords = df.V3_Soma[df.Segment_ID == x]
        
        # If input is a name:
        elif isinstance(x,str):
            name = x
            seg_id = [df.loc[[x in n for n in df.Name],'Segment_ID']]
            soma_coords = df.V3_Soma[df.Segment_ID == x]
        
        #TODO: Add get by annotations
        
        # If there are extra segment IDs attached to the entry, append them, and fuse them together. 
        # This really should be unnecessary once proofreading is online (read: I know how to interact with it)
        
        fuse = False
        annotations = self.get_annotations(x)
        annotations[x] = annotations[x] + ['{}_ID: '.format(self.segmentation_version) + str(seg_id) ] 

        ## TODO: Update this for use with dynamic seg. Check segment IDs before appending to anntations, also add a timestamp. 
        

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
            if not self.__is_iter(x):
                x = [x]
                
            to_upload = x
            
 
        
        
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
                    i.annotations.append(self.name)
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
            
            
            import pandas as pd


    def soma_annotation_table(self,path=None):
        if path is None:
            path = self.filename.parent

        self.get_database()
        entry = []
        for index,row in self.neurons.iterrows():

            p1 = str(row.V4_Soma)
            p1 = p1.replace('[','(')
            p1 = p1.replace(']',')')

            p2 = str(str(list(row.V4_Soma + np.array([100,100,10]))))
            p2 = p2.replace('[','(')
            p2 = p2.replace(']',')')


            entry.append( {'Coordinate 1':p1, 
                     'Coordinate 2': p2, 
                     'Ellipsoid Dimensions': None, 
                     'Tags':None, 
                     'Description': row.Name, 
                     'Segment IDs': row.Segment_ID,
                     'Parent ID':None,
                     'Type':'AABB',
                     'ID':None} )

        ng_annotations = pd.DataFrame(entry)
        ng_annotations.to_csv(path / (self.filename.name[0:self.filename.name.rfind('.')] + '_NG_Soma_Annotation.csv'),index=False)



            
        
        
        
