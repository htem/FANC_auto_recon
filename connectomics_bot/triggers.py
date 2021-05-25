import re

def get_response_key(command):
    regex = re.search
    lookup = search_triggers
    for key, value in lookup:
        if regex(key, command):
            return value
    return None


search_triggers = (
    (re.compile("get upstream partners"), 'get upstream partners'),
    (re.compile("get downstream partners"), "get downstream partners"),
    (re.compile("get all annotation tables"), "get all annotation tables"),
    (re.compile("get annotation table:"), "download annotation table"),
    (re.compile("get top [0-9]+ upstream partners"), "get top upstream partners"),
    (re.compile("get top [0-9]+ downstream partners"), "get top downstream partners"),
    (re.compile("get annotation tables by"), "get user tables"),
    (re.compile("find neuron"), "find neuron"),
    (re.compile("_update_roots"), "update roots"),
    (re.compile("skeleton [IiDd]+"),  "skel2seg"),
    (re.compile("[Gg]et empty"), "empty link")
    )

