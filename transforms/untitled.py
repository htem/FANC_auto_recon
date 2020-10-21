class neuron_database:
    
    def __init__(self,filename):

        self.filename = filename
        self.initialize_database()
        self.v4_url = 'https://storage.googleapis.com/zetta_lee_fly_vnc_001_segmentation/vnc1_full_v3align_2/realigned_v1/seg/full_run_v1'
        self.dynamic_seg_url = 'https://standalone.poyntr.co/segmentation/table/vnc1_full_v3align_2'
        self.dynamic_seg_res = np.array([17.2,17.2,45])
        self.v4_res = np.array([4.3,4.3,45])

    def initialize_database(self):
        fileEmpty =  os.path.exists(self.filename)
        if not fileEmpty:
            df = pd.DataFrame(data = None, columns=['Segment_ID','Name','V4_Soma','V3_Soma','Annotations','Extra_Segment_IDs'])
            df.to_csv(self.filename)
        else:
            print('Database active')

    def get_cloudvolume(self,vol_url = None):
        if vol_url is None:
            vol_url = self.v4_url
            
        self.cloud_volume = CloudVolume(vol_url)

    def get_database(self):
        self.database = pd.read_csv(self.filename)



    def check_seg_id(self,seg_id):
        df = pd.read_csv(self.filename)
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

        updated = self.update_db(soma_coord,
                                 Name=Name, 
                                 Annotations=Annotations,
                                 override=override,
                                 extra_segids = extra_segids)   
        if updated is True:
            print('entry added')
        else:
            print('entry already exists')


    def update_db(self,
              v4_pt, 
              Name=None, 
              Annotations = None, 
              override=False,
              extra_segids = None):

        filename = self.filename
        seg_id = neuroglancer_utilities.seg_from_pt(v4_pt)
        v3_transform = neuroglancer_utilities.fanc4_to_3(v4_pt,scale=2)
        v3_pt = [v3_transform['x'],v3_transform['y'],v3_transform['z']]

        if self.check_seg_id(seg_id) is False or override is True:
            df = pd.DataFrame({'Segment_ID':seg_id, 'Name': Name, 'V4_Soma':v4_pt, 'V3_Soma':v3_pt,'Annotations':Annotations,'Extra_Segment_IDs':extra_segids})
            df.to_csv(filename, mode='a', header=False)
            return(True) 
        else:
            return(False)


    def get_annotations(self,x):
        filename = self.filename
        annotations = {}

        if isinstance(x,int):
            column = 'Segment_ID'
        elif isinstance(x,str):
            column = 'Name'
        df = pd.read_csv(self.filename)
        for index, row in df.iterrows():
            if str(x) in str(row[column]):
                annotations[x] = row.Annotations

        return(annotations)


    def update_segIDs(self,seg_id=None):

        new_seg_ids = []

        df = pd.read_csv(self.filename)

        for index, row in df.iterrows():
            row.Segment_ID = neuroglancer_utilities.seg_from_pt(row.V4_Soma,vol_url=dynamic_seg_url,seg_mip = dynamic_seg_res)

        print('Segment IDs updated')



    def add_segIDs(self,x,extra_IDs):

        filename = self.filename
        df = pd.read_csv(self.filename)

        if isinstance(x,int):
            df.loc[df.Segment_ID == x,'Extra_Segment_IDs'] = extra_IDs

        elif isinstance(x,str):
            df.loc[[x in n for n in df.Name],'Extra_Segment_IDs'] = extra_IDs

        df.to_csv(self.filename)


    def get_mesh(self,x,vol_url=None):
        
        if vol_url is None:
            vol_url = self.v4_url

        vol = CloudVolume(vol_url)
        df = pd.read_csv(self.filename)

        if isinstance(x,int):
            seg_id = x
            extra_segids = df.loc[df.Segment_ID == x,'Extra_Segment_IDs']

        elif isinstance(x,str):
            seg_id = [df.loc[[x in n for n in df.Name],'Segment_ID']]
            extra_segids = df.loc[[x in n for n in df.Name],'Extra_Segment_IDs']

        if extra_segids is not None:
            mesh = vol.mesh.get(seg_id+extra_segids,remove_duplicate_vertices = True, fuse = True)

        return mesh
    
    
    def plot_mesh(self,x,vol_url=None)
        if vol_url is None:
                vol_url = self.v4_url
        
        mesh = self.get_mesh(x,vol_url = vol_url)
        
        
        for i in range(len(meshes)):
    trimesh_vtk.render_actors([meshes[i]],do_save=True,filename='/Users/brandon/Documents/MN_Analysis/T1_Leg_Meshes/' + mesh_names[i] + '.png')
    
        
        

    
    def get_skeleton(self,x,
                     vol_url = None,
                     method = 'Kimimaro',
                     transform = True,
                     cache_path = None,
                     output = 'pymaid'):
        
        if vol_url is None:
            vol_url = self.v4_url

        skeleton = skeletonization.get_skeleton(seg_id,
                                     vol_url,
                                     method=method,
                                     transform=transform,
                                     cache_path=cache_path,
                                     annotations=annotations,
                                     name=name,
                                     output=output)
        






        
        
        
        
        
    

        
        
        
 
                    
        
        
            
        
