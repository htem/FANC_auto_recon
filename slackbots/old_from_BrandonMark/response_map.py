import response_methods as rsp

response_dict = {
    'get upstream partners': [
        rsp.get_upstream_partners
    ],
    'get downstream partners': [
        rsp.get_downstream_partners
    ],
    'get top upstream partners': [
        rsp.get_top_upstream_partners
    ],
    'get top downstream partners': [
        rsp.get_top_downstream_partners
    ],
    'get all annotation tables': [
        rsp.get_annotation_tables
    ],
    'download annotation table': [
        rsp.download_annotation_table,
    ],
    'get user tables': [
        rsp.get_user_tables
    ],
    'find neuron': [
        rsp.find_neuron
    ],
    'update roots': [
        rsp.update_roots
    ],
    'skel2seg': [
        rsp.getskel2seg
    ],
    'empty link':[
        rsp.empty_link
    ]
        
    
}

