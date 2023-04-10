#!/usr/bin/env python3
"""
Install the slack python package with `pip install slack_sdk`
Some useful Slack API info pages:
  - https://api.slack.com/messaging/retrieving
  - https://api.slack.com/messaging/sending

View and configure your slack apps: https://api.slack.com/apps
Through Features > App Home > Show Tabs, select "Allow users to send
  Slash commands and messages from the messages tab" to enable DMs.
Through Features > OAuth & Permissions > Scopes > Bot Token Scopes,
give your bot these permissions:
  chat:write
  im:read
  im:history
From Features > OAuth & Permissions > OAuth Tokens for Your Workspace,
  copy your app's auth token to your shell environment by adding a line
  like this to your shell startup file (e.g. ~/.bashrc, ~/.zshrc):
    export SLACK_BOT_TOKEN=xoxb-123456789012-...

TODO:
- Parse argument into segid, tag, and tag2
- Validate that tag+tag2 is a valid pair
- Post
- Query
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Union

import requests
import slack_sdk
import numpy as np
import pandas as pd
from caveclient import CAVEclient

import fanc


caveclient = CAVEclient('fanc_production_mar2021')
table_name = 'neuron_information'

if len(sys.argv) > 1 and sys.argv[1] in os.environ:
    token = os.environ[sys.argv[1]]
else:
    token = os.environ['SLACK_TOKEN_FANC_NEURONINFORMATIONBOT']
slackclient = slack_sdk.WebClient(token=token)

with open('slack_user_permissions.json', 'r') as f:
    permissions = json.load(f)

# Build some dicts that make it easy to look up IDs for users and channels
#all_users = slackclient.users_list()['members']
#userid_to_username = {
#    user['id']: user['profile']['display_name'].lower()
#    if user['profile']['display_name'] else user['profile']['real_name'].lower()
#    for user in all_users
#}
#username_to_userid = {v: k for k, v in userid_to_username.items()}
#all_conversations = slackclient.conversations_list()['channels']
#channelname_to_channelid = {x['name']: x['id'] for x in all_conversations}
#channelid_to_channelname = {x['id']: x['name'] for x in all_conversations}


def show_help():
    return (
"""
Send me a message that looks like one of the `example messages below` to get or upload information about neurons in FANC.

Get information:
- `648518346481082458?` -> get information about a segment
- `648518346481082458? all` -> get extended information about a segment

Upload annotations:
(Before contributing your own annotations, you are REQUIRED to read the description of the "neuron_information" table at https://cave.fanc-fly.com/annotation/views/aligned_volume/fanc_v4/table/neuron_information AND read the list of valid annotations at https://github.com/htem/FANC_auto_recon/wiki/Neuron-annotations )
- `648518346481082458! primary class > central neuron` -> annotate that the indicated segment's "primary class" is "central neuron".
- `648518346481082458! central neuron > ascending neuron` -> annotate that the indicated segment's "central neuron" subtype is "ascending neuron".
- `648518346481082458! projection pattern > intersegmental` -> annotate that the indicated segment projects to multiple segments (T1 / T2 / T3 / abdominal ganglion) of the VNC.

Uploading currently only works on neurons with somas. Will work on sensory and descending neurons soon.
Feel free to contact Jasper with any questions or bug reports.
""")


def process_message(message: str, user: str, fake=False) -> str:
    """
    Process a slack message posted by a user, and return a text response.

    See the `show_help()` function in this module for a description of
    valid message formats and how they will be processed.

    Arguments
    ---------
    message : str
        The user's Slack message.
    user : str
        The user's Slack ID. This is a string that looks like 'ULH2UM0H4'
        and is provided by Slack for each user.

    Returns
    -------
    response : str
        A message to tell the user the information they requested, or to
        tell them the result of the upload operation their message
        triggered, or to describe an error that was encountered when
        processing their message.
    """
    tokens = message.strip(' ').split(' ')
    if len(tokens) == 0:
        return ("NO ACTION: Your message is empty or I couldn't understand"
                " it. Make a post containing the word 'help' if needed.")
    #if tokens[0] not in all_tables and tokens[0][-1] not in ['?', '!']:
    #    return (f"ERROR: Could not understand first word `{tokens[0]}`. Make "
    #            "a post containing the word 'help' if you need instructions.")

    #if tokens[0] in all_tables:
    #    table_name = tokens.pop(0)
    #else:
    #    table_name = default_proofreading_table
    try:
        segid = int(tokens[0][:-1])
    except ValueError:
        return (f"ERROR: Could not convert the first word"
                f" `{tokens[0][:-1]}` to int. Is it a segID?")

    caveclient.materialize.version = caveclient.materialize.most_recent_version()

    if tokens[0].endswith('?'):  # Query
        return_as = 'list'
        if len(tokens) > 1 and tokens[1].lower() in ['all', 'everything', 'verbose']:
            return_as = 'dataframe'
        try:
            info = fanc.lookup.get_annotations(segid,
                                               table_name=table_name,
                                               return_as=return_as)
        except Exception as e:
            return f"`{type(e)}`\n```{e}```"
        if return_as == 'dataframe':
            info.drop(columns=['id', 'valid', 'pt_supervoxel_id', 'pt_root_id',
                               'pt_position'], inplace=True)
            info.rename(columns={'tag': 'annotation',
                                 'tag2': 'annotation_class'}, inplace=True)
            return ('```' + info.to_string(index=False) + '```')
        else:
            return ('```' + '\n'.join(info) + '```')

    elif tokens[0].endswith('!'):  # Upload
        # Permissions
        table_permissions = permissions.get(table_name, None)
        if table_permissions is None:
            return f"ERROR: `{table_name}` not listed in permissions file."
        cave_user_id = table_permissions.get(user, None)
        if cave_user_id is None:
            return ("You have not yet been given permissions to post to"
                    f" `{table_name}`. Please send Jasper a DM on slack"
                    " to request permissions.")

        # Sanity checks
        if not caveclient.chunkedgraph.is_latest_roots(segid):
            return (f"ERROR: {segid} is not a current segment ID."
                    " Was the segment edited recently? Or did you"
                    " copy-paste the wrong thing?")
        if have_recently_uploaded(segid, table_name):
            return (f"ERROR: I recently uploaded segment {segid}"
                    f" to `{table_name}`. I'm not going to upload"
                    " it again.")

        # Find soma
        soma = fanc.lookup.somas_from_segids(segid, timestamp='now')
        if len(soma) > 1:
            return (f"ERROR: Segment {segid} has multiple entires"
                    " in the soma table, with the coordinates listed"
                    " below. Cannot add annotations.\n\n"
                    f"{np.vstack(soma.pt_position)}")
        elif len(soma) == 0 and len(tokens) == 1:
            return (f"ERROR: Segment {segid} has no entry in the soma"
                    " table.\n\nIf you clearly see a soma attached to"
                    " this object, probably the automated soma detection"
                    " algorithm missed this soma. If so, message Sumiya"
                    " Kuroda and he can add it to the soma table."
                    "\n\nIf you're sure this is a descending neuron or"
                    " a sensory neuron, you can specify a point to"
                    "anchor the annotation. Call 'help' for details.")
        elif len(soma) == 0 and len(tokens) != 4:
            return ("ERROR: You did not provide a segment ID followed"
                    " by an xyz point coordinate, at least not in the"
                    " expected format.")
        elif len(soma) == 0 and len(tokens) == 4:
            try:
                point = [float(i.strip(',')) for i in tokens[1:]]
            except ValueError:
                return (f"ERROR: Could not convert the last 3 words to"
                        " integers. Are they point coordinates?"
                        f"\n\n`{[i for i in tokens[1:]]}`")
            segid_from_point = fanc.lookup.segids_from_pts(point)
            if not segid_from_point == segid:
                return (f"ERROR: The provided point `{point}` is inside"
                        f" segment {segid_from_point} which doesn't"
                        f" match the segment ID you provided, {segid}.")

        elif len(soma) == 1 and len(tokens) > 1:
            return (f"ERROR: Segment {segid} has an entry in the"
                    f" soma table at {list(np.hstack(soma.pt_position))}"
                    " but you provided additional information."
                    " Additional information is unexpected when the"
                    " segment has a soma, so I didn't do anything.")
        else:
            point = list(np.hstack(soma.pt_position))

        stage = caveclient.annotation.stage_annotations(table_name)
        try:
            stage.add(
                tag=annotation,
                tag2=annotation_class,
                pt_position=point,
                user_id=cave_user_id,
                valid_id=segid
            )
        except Exception as e:
            return f"ERROR: Staging failed with error\n`{type(e)}`\n```{e}```"

        if fake:
            return (f"Upload FAKE for segment {segid} and point"
                    f" coordinate `{point}`.")
        try:
            response = caveclient.annotation.upload_staged_annotations(stage)
            record_upload(segid, cave_user_id, table_name)
            return (f"Upload to `{table_name}` succeeded:\n"
                    f"- Segment {segid}\n"
                    f"- Point coordinate `{point}`\n"
                    f"- Annotation: `{annotation}`\n"
                    f"- Annotation class: `{annotation_class}`\n"
                    f"- Annotation ID: {response}")
        except Exception as e:
            return f"ERROR: Upload failed with error\n`{type(e)}`\n```{e}```"

    else:
        return ("ERROR: The first word in your message isn't a segment"
                " ID terminated by a ! or a ?. Make a post containing"
                " the word 'help' if you need instructions.")
    

def fetch_messages_and_post_replies(channel, verbosity=1, fake=False):
    channel_data = slackclient.conversations_history(channel=channel['id'])
    for message in channel_data['messages']:
        if message.get('subtype', None):
            # Skip if this is a system message (not something posted by a user)
            continue
        if message.get('thread_ts', None):
            # Skip if this message has a reply already
            continue
        if not message['user'] == channel['user']:
            # Skip if this message wasn't sent by the user,
            # e.g. it was sent by the bot
            continue
        response = None
        if 'help' in message['text'].lower():
            response = show_help()

        if verbosity >= 2:
            print('Processing message:', message)
        elif verbosity >= 1:
            print('Processing message with timestamp', message['ts'])

        if response is None:
            response = process_message(message['text'],
                                       message['user'],
                                       fake=fake)

        if verbosity >= 1:
            print('Posting response:', response)

        slackclient.chat_postMessage(
            channel=channel['id'],
            thread_ts=message['ts'],
            text=response
        )


def record_upload(segid, user, table_name) -> None:
    uploads_fn = f'neuron_information_bot_uploads_{table_name}.txt'
    with open(uploads_fn, 'a') as f:
        f.write(f'{segid},{user}\n')


def have_recently_uploaded(segid, table_name) -> bool:
    uploads_fn = f'neuron_information_bot_uploads_{table_name}.txt'
    with open(uploads_fn, 'r') as f:
        recent_uploads = [int(line.strip().split(',')[0]) for line in f.readlines()]
    return segid in recent_uploads


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fake':
        fake = True
        print('Running in FAKE mode')
    else:
        fake = False

    while True:
        print(datetime.now().strftime('%A %Y-%h-%d %H:%M:%S'))
        direct_message_channels = slackclient.conversations_list(types='im')
        for channel in direct_message_channels['channels']:
            if channel['user'] == 'USLACKBOT':
                continue
            try:
                fetch_messages_and_post_replies(channel=channel, verbosity=2, fake=fake)
            except Exception as e:
                print('Encountered exception: {} {}'.format(type(e), e))
                logfn = os.path.join('exceptions_neuron_information_bot',
                                     datetime.now().strftime('%Y-%h-%d_%H-%M-%S.txt'))
                with open(logfn, 'w') as f:
                    f.write('{}\n{}'.format(type(e), e))
                time.sleep(50)
        time.sleep(10)
