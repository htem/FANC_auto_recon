#!/usr/bin/env python3
"""
Slack API info:
    https://api.slack.com/messaging/retrieving
    https://api.slack.com/messaging/sending
Install the slack python package with `pip install slack_sdk`

View your slack apps:
    https://api.slack.com/apps
Save your app's auth token to your shell environment by adding a line like this
to your shell startup file (e.g. ~/.bashrc, ~/.zshrc):
    export SLACK_BOT_TOKEN=xoxb-123456789012-...
"""

import os
import sys
import time
from datetime import datetime

import slack_sdk
import numpy as np
import pandas as pd
from caveclient import CAVEclient


caveclient = CAVEclient('fanc_production_mar2021')
info = caveclient.info.get_datastack_info()

if len(sys.argv) > 1 and sys.argv[1] in os.environ:
    token = os.environ[sys.argv[1]]
else:
    token = os.environ['SLACK_TOKEN_FANC_SOMABOT']
slackclient = slack_sdk.WebClient(token=token)

# Build some dicts that make it easy to look up IDs for users and channels
#all_users = client.users_list()['members']
#userid_to_username = {
#    user['id']: user['profile']['display_name'].lower()
#    if user['profile']['display_name'] else user['profile']['real_name'].lower()
#    for user in all_users
#}
#username_to_userid = {v: k for k, v in userid_to_username.items()}
all_conversations = slackclient.conversations_list()['channels']
channelname_to_channelid = {x['name']: x['id'] for x in all_conversations}
channelid_to_channelname = {x['id']: x['name'] for x in all_conversations}

channel_id = channelname_to_channelid['connect-the-somas']  # returns 'C04EYRQCVS8'


def fetch_orphaned_somas(y_range=[0, 160000], query_size=20, synapse_count_threshold=50):
    """
    Get a list of somas that have few postsynaptic sites associated with
    them, which likely means they are falsely split from their arbor and
    they are in need of a few merge operations.
    query_size determines how many somas to select randomly from the
    full soma table for inspection. Then only the subset of those somas
    that have a small enough synapse count will be returned. As such,
    different calls to this function will return different numbers of
    somas. As of Dec 2022, calling this function with the default
    parameters typically returns between 2 and 6 somas.

    --- Arguments ---
    y_range : list of ints (default [0, 160000]) or str
      If str, must be 'T1', 'T2', or 'T3' to select these coordinate ranges:
        T1 -> [     0, 120000]
        T2 -> [120000, 160000]
        T3 -> [160000, 223000]
      Only inspect somas with y coordinate between y_range[0] and
      y_range[1]. This allows us to prioritize connecting somas to
      arbors in specific regions of the dataset. The default parameters
      currently specify to look in the anterior half of the VNC (T1 &
      wing & T2 neuropils), where most people in the community are
      currently focused.
      Some useful landmark values:
        120000 : Just below T1
        160000 : Just below T2
        223000 : The bottom of the dataset

    query_size : int (default 20)
      Request postsynaptic site counts for this many somas. Larger queries take
      more time.

    synapse_count_threshold : int (default 50)
      Among the queried somas, only return ones that have fewer than this
      number of postsynaptic sites. 50 seems to be a pretty reasonable cutoff,
      so it's unlikely users would find it useful to change this.

    --- Returns ---
    A DataFrame with 2 columns named 'pt_root_id' and 'synapse_counts'. Each row
    represents the segment ID and number of synapses for a different object.
    """

    if isinstance(y_range, str):
        y_range = {
            'T1': [     0, 120000],
            'T2': [120000, 160000],
            'T3': [160000, 223000]
        }[y_range]

    caveclient.materialize.version = caveclient.materialize.most_recent_version()
    somas = caveclient.materialize.query_table(info['soma_table'])
    somas = somas.loc[np.vstack(somas.pt_position)[:, 1] > y_range[0]]
    somas = somas.loc[np.vstack(somas.pt_position)[:, 1] < y_range[1]]
    soma_ids = somas.pt_root_id

    # Get synapses for a random n somas
    soma_ids_sample = soma_ids.sample(query_size)
    synapses = caveclient.materialize.synapse_query(post_ids=soma_ids_sample)

    synapse_counts = pd.Series(
        data=synapses.post_pt_root_id.value_counts(),
        index=soma_ids_sample,
        name='synapse_counts'
    ).fillna(0).astype('int64')
    synapse_counts.sort_values(inplace=True)

    return synapse_counts[synapse_counts <= synapse_count_threshold].reset_index()


def serve_somas_to_eligible_messages(verbosity=1, fake=False):
    channel_data = slackclient.conversations_history(channel=channel_id)
    for message in channel_data['messages']:
        if message.get('subtype', None): #in ['channel_join', 'channel_topic', 'channel_purpose']:
            continue
        if message.get('thread_ts', None): #message['ts']) != message['ts']:
            # If this message has a reply already
            continue
        if '<@U04EW9C2MEX>' not in message['text']:
            continue

        if verbosity >= 1:
            print('Serving somas to message with timestamp', message['ts'])
        kwargs = dict()
        if 'T1' in message['text'] and 'T2' not in message['text']:
            kwargs['y_range'] = 'T1'
        elif 'T2' in message['text'] and 'T1' not in message['text']:
            kwargs['y_range'] = 'T2'
        elif 'T3' in message['text']:
            kwargs['y_range'] = 'T3'

        orphaned_somas = pd.Series(dtype='int64')
        while orphaned_somas.shape == (0,):
            orphaned_somas = fetch_orphaned_somas(**kwargs).values
        text = "Segment IDs: {}\nSynapse counts: {}".format(
                      list(orphaned_somas[:, 0]),
                      list(orphaned_somas[:, 1])
                  )
        if verbosity >= 2:
            print('Slack user post:', message)
            print('Slack bot post:', text)
        if fake:
            print('fake=True, not posting message')
            return
        slackclient.chat_postMessage(
            channel=channel_id,
            thread_ts=message['ts'],
            text=text
        )

if __name__ == '__main__':
    while True:
        print(datetime.now().strftime('%A %Y-%h-%d %H:%M:%S'))
        try:
            serve_somas_to_eligible_messages(verbosity=2, fake=False)
        except Exception as e:
            print('Encountered exception: {} {}'.format(type(e), e))
            logfn = os.path.join('exceptions', datetime.now().strftime('%Y-%h-%d_%H-%M-%S') + '.txt')
            with open(logfn, 'w') as f:
                f.write('{}\n{}'.format(type(e), e))
            time.sleep(50)
        time.sleep(10)
