#!/usr/bin/env python3
"""
Install the slack python package with `pip install slack_sdk`
Some useful Slack API info pages:
  - https://api.slack.com/messaging/retrieving
  - https://api.slack.com/messaging/sending

View your slack apps: https://api.slack.com/apps
This app requires the following permissions:
  - channels:history
  - channels:read
  - chat:write
  - users:read
Save your app's auth token to your shell environment by adding a line
like this to your shell startup file (e.g. ~/.bashrc, ~/.zshrc):
    export SLACK_BOT_TOKEN=xoxb-123456789012-...
"""

import os
import sys
import json
import time
from datetime import datetime

import requests
import slack_sdk
import numpy as np
import pandas as pd
from caveclient import CAVEclient

import fanc


caveclient = CAVEclient('fanc_production_mar2021')
all_tables = caveclient.materialize.get_tables()
default_proofreading_table = 'proofreading_status_table_v0'

if len(sys.argv) > 1 and sys.argv[1] in os.environ:
    token = os.environ[sys.argv[1]]
else:
    token = os.environ['SLACK_TOKEN_FANC_PROOFREADINGSTATUSBOT']
slackclient = slack_sdk.WebClient(token=token)

with open('slack_user_permissions.json', 'r') as f:
    permissions = json.load(f)

# Build some dicts that make it easy to look up IDs for users and channels
all_users = slackclient.users_list()['members']
userid_to_username = {
    user['id']: user['profile']['display_name'].lower()
    if user['profile']['display_name'] else user['profile']['real_name'].lower()
    for user in all_users
}
username_to_userid = {v: k for k, v in userid_to_username.items()}
all_conversations = slackclient.conversations_list()['channels']
channelname_to_channelid = {x['name']: x['id'] for x in all_conversations}
channelid_to_channelname = {x['id']: x['name'] for x in all_conversations}

channel_id = channelname_to_channelid['proofreading-status-bot']  # returns 'C04Q90KGGRH'


def show_help():
    return (
"""
Valid messages must follow one of the following formats:

`@proofreading-status-bot 648518346481082458?`
A segment ID followed by a `?` indicates that you want to know whether this segment ID is already in the proofreading table.

`@proofreading-status-bot 648518346481082458!`
A segment ID followed by a `!` indicates that you want to mark this segment ID as being proofread. This message format only works if the segment ID has exactly one soma attached to it, in which case the soma's location will be used to anchor the annotation. If the segment ID is a descending neuron or sensory neuron and so it has no soma, use the format described in the section below.

`@proofreading-status-bot 648518346481082458! 48848 114737 2690` or
`@proofreading-status-bot 648518346481082458! 48848, 114737, 2690`
A segment ID followed by a `!` followed by an xyz point coordinate (typically copied from the top bar of neuroglancer) indicates that you want to mark this segment ID as being proofread, using the given xyz coordinate as a representative point inside the neuron's soma or large-diameter backbone.

• These examples use the segment ID `648518346481082458` but you should substitute this with the segment ID that you're interested in.
• If you want to confirm the bot is working properly, try sending the first example message to the channel and make sure you get a response.
• If you want to add a large number of entries to the proofreading status table, you can contact Jasper directly instead of using this bot.
""")

def is_proofread(segid: int, table_name: str, return_previous_ids=False):
    """
    Determine whether the a segment has been marked as proofread

    Returns
    -------
    int (if return_previous_ids is False)
     0: Not marked as proofread
     1: The given segment ID has been marked as proofread
     2: A previous version of this segment was marked as proofread,
        but this exact segment ID has not.
    int or list (if return_previous_ids is True)
     Same as above, but instead of returning the value 2 in the final case, a
     list of the previous segment IDs that were marked as proofread are
     returned.
    """
    now = datetime.utcnow()
    try:
        valid_id_matches = caveclient.materialize.live_live_query(
            table_name,
            now,
            filter_equal_dict={table_name: {'valid_id': segid}},
            #allow_missing_lookups=True
        )
        if len(valid_id_matches) > 0:
            return 1
    except requests.exceptions.HTTPError as e:
        if 'returned no results' not in e.args[0]:
            raise e

    try:
        root_id_matches = caveclient.materialize.live_live_query(
            table_name,
            now,
            filter_equal_dict={table_name: {'pt_root_id': segid}},
            #allow_missing_lookups=True
        )
        if len(root_id_matches) > 0:
            if return_previous_ids:
                return list(root_id_matches.valid_id.values)
            else:
                return 2
    except requests.exceptions.HTTPError as e:
        if 'returned no results' not in e.args[0]:
            raise e

    return 0


def process_message(message: str, user: str, fake=False) -> str:
    """
    Process a slack message posted by a user, and return a text response.

    See the `show_help()` function in this module for a description of
    valid message formats and how they will be processed.

    Arguments
    ---------
    message : str
        The user's slack post, with the leading '@proofreading-status-bot' removed

    Returns
    -------
    response : str
        A message to tell the user the information they requested, or to
        tell them the result of the upload operation their message triggered.
    """
    tokens = message.strip(' ').split(' ')
    if len(tokens) == 0:
        return ("NO ACTION: Your message is empty or I couldn't understand"
                " it. Make a post containing the word 'help' if needed.")
    if tokens[0] not in all_tables and tokens[0][-1] not in ['?', '!']:
        return (f"ERROR: Could not understand first word `{tokens[0]}`. Make "
                "a post containing the word 'help' if you need instructions.")

    if tokens[0] in all_tables:
        table_name = tokens.pop(0)
    else:
        table_name = default_proofreading_table
    try:
        segid = int(tokens[0][:-1])
    except ValueError:
        return (f"ERROR: Could not convert the first word"
                f" {tokens[0][:-1]} to int. Is it a segID?")

    if tokens[0].endswith('?'):  # Query
        try:
            proofreading_status = is_proofread(segid, table_name,
                                               return_previous_ids=True)
        except Exception as e:
            return f"`{type(e)}`\n```{e}```"

        if proofreading_status == 0: 
            return f"Segment {segid} is not marked as proofread in `{table_name}`"
        elif proofreading_status == 1:
            return (f"Segment {segid} has been marked as proofread in"
                    f" `{table_name}`.")
        elif isinstance(proofreading_status, list) or proofreading_status == 2:
            return (f"A previous version(s) of segment {segid} was found in"
                    f" `{table_name}`: {proofreading_status}.\nThis means it"
                    " was at some point marked as proofread but then edited"
                    " afterward, and the new version has not yet been marked"
                    " as proofread.")
        else:
            return ValueError('Expected proofreading_status to be 0, 1, or 2')

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
        if is_proofread(segid, table_name) == 1:
            return (f"ERROR: {segid} is already marked as proofread in"
                    f" table `{table_name}`. Taking no action.")

        # Find soma
        soma = fanc.lookup.somas_from_segids(segid, timestamp='now')
        if len(soma) > 1:
            return (f"ERROR: Segment {segid} has multiple entires"
                    " in the soma table, with the coordinates listed"
                    " below. Shame on you for marking a cell as"
                    " proofread when it still has two somas! (Or"
                    " there's a bug in my code.)\n\n"
                    f"{np.vstack(soma.pt_position)}")
        elif len(soma) == 0 and len(tokens) == 1:
            return ("ERROR: Segment {segid} has no entry in the soma"
                    " table.\n\nIf you clearly see a soma attached to"
                    " this object, probably the automated soma detection"
                    " algorithm missed this soma. If so, tag Sumiya"
                    " Kuroda here and he can add it to the soma table."
                    "\n\nIf you're sure this is a descending neuron or"
                    " a sensory neuron, you can specify a point to"
                    "anchor the proofreading annotation. Call 'help'"
                    " for details.")
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
            segid_from_point = fanc.lookup.segids_from_pts(pt)
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
                proofread=True,
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
            record_upload(segid, table_name)
            return (f"Upload to `{table_name}` succeeded:\n"
                    f"- Segment {segid}\n"
                    f"- Point coordinate `{point}`\n"
                    f"- Annotation ID: {response}")
        except Exception as e:
            return f"ERROR: Upload failed with error\n`{type(e)}`\n```{e}```"

    else:
        return ("ERROR: The first word in your message isn't a segment"
                " ID terminated by a ! or a ?. Make a post containing"
                " the word 'help' if you need instructions.")
    

def fetch_messages_and_post_replies(verbosity=1, fake=False):
    channel_data = slackclient.conversations_history(channel=channel_id)
    for message in channel_data['messages']:
        if message.get('subtype', None):
            # Skip if this is a system message (not something posted by a user)
            continue
        if message.get('thread_ts', None): #message['ts']) != message['ts']:
            # Skip if this message has a reply already
            continue
        response = None
        if 'help' in message['text'].lower():
            response = show_help()
        elif not message['text'].startswith('<@U04PUHVDSLX>'):
            # Skip if this message doesn't start with @proofreading-status-bot
            continue

        if verbosity >= 2:
            print('Processing message:', message)
        elif verbosity >= 1:
            print('Processing message with timestamp', message['ts'])

        if response is None:
            response = process_message(message['text'].strip('<@U04PUHVDSLX>'),
                                       message['user'],
                                       fake=fake)


        if verbosity >= 1:
            print('Posting response:', response)

        slackclient.chat_postMessage(
            channel=channel_id,
            thread_ts=message['ts'],
            text=response
        )


def record_upload(segid, table_name):
    uploads_fn = f'proofreading_status_bot_uploads_{table_name}.txt'
    with open(uploads_fn, 'a') as f:
        f.write(f'{segid}\n')


def have_recently_uploaded(segid, table_name):
    uploads_fn = f'proofreading_status_bot_uploads_{table_name}.txt'
    with open(uploads_fn, 'r') as f:
        recent_uploads = [int(line.strip()) for line in f.readlines()]
    return segid in recent_uploads


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fake':
        fake = True
        print('Running in FAKE mode')
    else:
        fake = False

    while True:
        print(datetime.now().strftime('%A %Y-%h-%d %H:%M:%S'))
        try:
            fetch_messages_and_post_replies(verbosity=2, fake=fake)
        except Exception as e:
            print('Encountered exception: {} {}'.format(type(e), e))
            logfn = os.path.join('exceptions_proofreading_status_bot', datetime.now().strftime('%Y-%h-%d_%H-%M-%S') + '.txt')
            with open(logfn, 'w') as f:
                f.write('{}\n{}'.format(type(e), e))
            time.sleep(50)
        time.sleep(10)
