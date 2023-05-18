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


def show_help():
    return (
"""
Send me a message that looks like one of the `example messages below` to get or upload information about neurons in FANC.

Get information:
- `648518346481082458?` -> get information about a segment
- `648518346481082458? all` -> get extended information about a segment

Upload annotations:
(Before contributing your own annotations, you are REQUIRED to read the description of the "neuron_information" table at https://cave.fanc-fly.com/annotation/views/aligned_volume/fanc_v4/table/neuron_information AND read the list of valid annotations at https://github.com/htem/FANC_auto_recon/wiki/Neuron-annotations )
- `648518346481082458! primary class > central neuron` -> annotate that the indicated segment's "primary class" is "central neuron" (as opposed to "sensory neuron" or "motor neuron").
- `648518346489818455! left-right projection pattern > bilateral` -> annotate that segment 648518346489818455 projects bilaterally, i.e. has synaptic connections on both sides of the VNC's midplane.

This bot is a work in progress - notably, you can't yet annotate most sensory neurons because the `peripheral_nerves` table is not complete yet. This will be addressed soon.
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
    # For some reason the '>' character typed into slack
    # is reaching this code as '&gt;', so revert it for readability.
    message = message.replace('&gt;', '>')
    tokens = message.strip(' ').split(' ')
    if len(tokens) == 0:
        return ("NO ACTION: Your message is empty or I couldn't understand"
                " it. Make a post containing the word 'help' if needed.")
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
            info = fanc.lookup.annotations(segid,
                                           table_name=table_name,
                                           return_as=return_as)
        except Exception as e:
            return f"`{type(e)}`\n```{e}```"
        if len(info) == 0:
            return "No annotations found."
        if return_as == 'dataframe':
            info.drop(columns=['id', 'valid', 'pt_supervoxel_id',
                               'pt_root_id', 'pt_position', 'deleted',
                               'superceded_id'], inplace=True)
            info.rename(columns={'tag': 'annotation',
                                 'tag2': 'annotation_class'}, inplace=True)
            info['created'] = info.created.apply(lambda x: x.date())
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

        # Parse and validate annotations
        if len(tokens) < 4 or '>' not in tokens:
            return ("ERROR: To upload neuron information, your message"
                    " must have the format `{segid}! {annotation} >"
                    " {annotation_class}`. Run 'help' for examples.")
        annotation_tokens = ' '.join(tokens[1:]).split('>')
        if len(annotation_tokens) != 2:
            return (f"ERROR: Could not parse `{' '.join(tokens[1:])}`"
                    " into an annotation and annotation_class.")
        annotation_class = annotation_tokens[0].strip()
        annotation = annotation_tokens[1].strip()

        if fake:
            try:
                fanc.annotations.is_allowed_to_post(segid, annotation_class, annotation)
                point = list(fanc.lookup.anchor_point(segid))
            except Exception as e:
                return f"`{type(e)}`\n```{e}```"
            return (f"FAKE: Would upload segment {segid}, point {point}"
                    f", annotations `{annotation_class} > {annotation}`.")

        try:
            response = fanc.upload.annotate_neuron(segid, annotation_class,
                                                   annotation, cave_user_id)
            record_upload(segid, annotation, annotation_class,
                          cave_user_id, table_name)
            uploaded_data = caveclient.annotation.get_annotation(table_name,
                                                                 response)[0]
            return (f"Upload to `{table_name}` succeeded:\n"
                    f"- Segment {segid}\n"
                    f"- Point coordinate `{uploaded_data['pt_position']}`\n"
                    f"- Annotation: `{annotation}`\n"
                    f"- Annotation class: `{annotation_class}`\n"
                    f"- Annotation ID: {response[0]}")
        except Exception as e:
            return f"ERROR: Annotation failed due to\n`{type(e)}`\n```{e}```"

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


def record_upload(segid, tag, tag2, user_id, table_name) -> None:
    uploads_fn = f'neuron_information_bot_uploads_to_{table_name}.txt'
    with open(uploads_fn, 'a') as f:
        f.write(f'{segid},{tag},{tag2},{user_id}\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fake':
        fake = True
        print('Running in FAKE mode')
    else:
        fake = False

    while True:
        print(datetime.now().strftime('%A %Y-%h-%d %H:%M:%S'))
        try:
            direct_message_channels = slackclient.conversations_list(types='im')
        except Exception as e:
            print('Encountered exception: {} {}'.format(type(e), e))
            logfn = os.path.join('exceptions_neuron_information_bot',
                                 datetime.now().strftime('%Y-%h-%d_%H-%M-%S.txt'))
            with open(logfn, 'w') as f:
                f.write('{}\n{}'.format(type(e), e))
            time.sleep(50)
            continue
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
