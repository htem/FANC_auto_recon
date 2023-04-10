#!/usr/bin/env python3

import anytree

help_url = 'https://github.com/htem/FANC_auto_recon/wiki/Neuron-annotations#neuron_information'

annotation_hierarchy = {
    'primary class': {
        'sensory neuron': {
            'chordotonal neuron': {},
            'bristle neuron': {},
            'hair plate neuron': {},
            'campaniform sensillum neuron': {},
            'descending neuron': {},
            'ascending neuron': {}},
        'central neuron': {
            'descending neuron': {},
            'ascending neuron': {},
            'VNC interneuron': {}},
        'motor neuron': {
            'T1 leg motor neuron': {},
            'T2 leg motor neuron': {},
            'T3 leg motor neuron': {},
            'neck motor neuron': {},
            'wing motor neuron': {},
            'haltere motor neuron': {},
            'abdominal motor neuron': {}},
        'efferent non-motor neuron': {
            'UM neuron': {
                'T1 leg UM neuron': {},
                'T2 leg UM neuron': {},
                'T3 leg UM neuron': {},
                'neck UM neuron': {},
                'wing UM neuron': {},
                'haltere UM neuron': {},
                'abdominal UM neuron': {}}}
    },
    'projection pattern': {
        'local': {},
        'intersegmental': {},
        'unilateral': {},
        'bilateral': {}
    }
}


def _dict_to_anytree(dict):
    """
    Given a dictionary containing a hierarchy of strings, return a dictionary
    with each string as a key and the corresponding anytree.Node as the value.
    """
    def _build_tree(annotations: dict, parent: dict = None, nodes: dict = {}):
        for annotation in annotations.keys():
            node = anytree.Node(annotation, parent=parent)
            nodes[annotation] = nodes.get(annotation, []) + [node]
            _build_tree(annotations[annotation], parent=node, nodes=nodes)
        return nodes
    
    return _build_tree(dict)

annotation_tree = _dict_to_anytree(annotation_hierarchy)


def print_annotation_tree():
    """
    Print the annotation hierarchy
    """
    def print_one_tree(root: str):
        for prefix, _, node in anytree.RenderTree(annotation_tree[root][0]):
            print(f'{prefix}{node.name}')
    print_one_tree('primary class')
    print_one_tree('projection pattern')


def guess_class(annotation: 'str') -> 'str':
    """
    Given an annotation, return the annotation class to which it belongs.
    """
    try:
        annotation_nodes = annotation_tree[annotation]
    except:
        raise ValueError(f'Class of "{annotation}" could not be guessed.'
                         f' See valid annotations at {help_url}')

    if len(annotation_nodes) > 1:
        raise ValueError(f'Class of "{annotation}" could not be guessed'
                         ' because it has multiple possible classes.'
                         f' See the annotation scheme at {help_url}')


    return annotation_nodes[0].parent.name



def is_valid_pair(annotation: str, annotation_class: str, raise_errors=True) -> bool:
    """
    Determine whether `annotation` is a valid annotation for the given
    `annotation_class`, according to the rules described at
    https://github.com/htem/FANC_auto_recon/wiki/Neuron-annotations#neuron_information
    """
    if annotation_class == 'neuron identity':
        return annotation not in annotation_tree
    
    try:
        class_nodes = annotation_tree[annotation_class]
    except:
        if raise_errors:
            raise ValueError(f'Annotation class "{annotation_class}" not'
                             f' recognized. See valid classes at {help_url}')
        else:
            return False
    try:
        annotation_nodes = annotation_tree[annotation]
    except:
        if raise_errors:
            raise ValueError(f'Annotation "{annotation}" not recognized.'
                             f' See valid annotations at {help_url}')
        else:
            return False

    for class_node in class_nodes:
        for annotation_node in annotation_nodes:
            if annotation_node in class_node.children:
                return True

    if raise_errors:
        parent_names = [node.parent.name
                        if node.parent is not None else '<no class>'
                        for node in annotation_nodes]
        if len(annotation_nodes) == 1:
            raise ValueError(f'Annotation "{annotation}" belongs to class'
                             f' "{parent_names[0]}" but you specified class'
                             f' "{annotation_class}". See the annotation'
                             f' scheme described at {help_url}')
        else:
            raise ValueError(f'Annotation "{annotation}" belongs to classes'
                             f' {parent_names} but you specified class'
                             f' "{annotation_class}". See the annotation'
                             f' scheme described at {help_url}')

    else:
        return False
    
    return True
