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
all_tables = caveclient.materialize.get_tables()
# If there are multiple proofreading tables indicating different levels of
# completion, the line below must list them in order of least complete
# proofreading status to most complete proofreading status
proofreading_tables = ['proofread_first_pass', 'proofread_second_pass']
default_proofreading_table = proofreading_tables[0]

if len(sys.argv) > 1 and sys.argv[1] in os.environ:
    token = os.environ[sys.argv[1]]
else:
    token = os.environ['SLACK_TOKEN_FANC_PROOFREADINGSTATUSBOT']
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
Hello! I can help you get or upload information about which neurons in FANC have been proofread. FANC has has two proofreading tables, `proofread_first_pass` and `proofread_second_pass`. Before you proceed, please read these descriptions carefully:

`proofread_first_pass`:
```
This table lists cells that have received a "first pass" of proofreading, meaning someone has spent at least ~10 minutes proofreading it, the cell has no obvious large merge errors remaining, and all the thick branches have been extended to so that the neuron's backbone is relatively complete. **Users should not hesitate to add entries to this "first pass" table for any cell that is in reasonable shape!** (We have a separate table, "proofread_second_pass", that indicates neurons that are basically complete, where we want to be more selective about only posting really well-proofread neurons.) All users can add entries to this table.
```
`proofread_second_pass`:
```
This table lists cells that have received a "second pass" of focused proofreading work to ensure that the cell's anatomy and connectivity are quite accurately reconstructed. **Users should only add cells to this table if they're confident that any remaining proofreading on this neuron would only cause trivial, almost unnoticeable changes to its morphology and connectivity.** (For cells that are in decent shape but have not received extensive attention, consider adding the cell to the table "proofread_first_pass" instead.) Only authorized users (typically only project leaders or proofreaders with >1 year of experience) can add entries to this table.
```

You can send me a message that looks like one of the `example messages below` to get or upload proofreading status information.

`648518346481082458?`
A segment ID followed by a "?" indicates that you want to know whether this segment ID has already been marked as proofread.

`648518346481082458!`
A segment ID followed by a "!" indicates that you want to mark this segment ID as "first pass" proofread.

`648518346481082458!!`
A segment ID followed by a "!!" indicates that you want to mark this segment ID as "second pass" proofread.

The "!" and "!!" commands above only work if the segment ID has exactly one soma attached to it, in which case the soma's location will be used to anchor the annotation. If the segment ID is a descending neuron or sensory neuron and so it has no soma, use the following type of command:

`648518346481082458! 48848 114737 2690` or
`648518346481082458! 48848, 114737, 2690`
To mark a *neuron with no soma* as proofread, provide an xyz point coordinate (typically copied from the top bar of neuroglancer) to use as a representative point. This should be a point inside the neuron's large-diameter backbone that is unlikely to be affected by any future edits to this neuron. (Like before, you can use either "!" to mark the neuron as "first pass" proofread, or use "!!" to mark it as "second pass" proofread.)


• These examples use the segment ID 648518346481082458 but you should substitute this with the segment ID that you're interested in.
• If you want to confirm I'm working properly, try sending me the first example message, `648518346481082458?`, and make sure you get a response.
• If you want to mark a large number of segments as proofread, you can send a message to Jasper on Slack instead of using this bot.
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
    if tokens[0] not in all_tables and tokens[0][-1] not in ['?', '!']:
        return (f"ERROR: Could not understand first word `{tokens[0]}`. Make "
                "a post containing the word 'help' if you need instructions.")

    if tokens[0] in all_tables:
        table_name = tokens.pop(0)
    elif tokens[0].endswith('!!'):
        table_name = proofreading_tables[1]
        # Convert !! to ! so both upload commands now end in just one !
        tokens[0] = tokens[0][:-1]
    else:
        table_name = default_proofreading_table
    try:
        segid = int(tokens[0][:-1])
    except ValueError:
        return (f"ERROR: Could not convert the first word"
                f" `{tokens[0][:-1]}` to int. Is it a segID?")

    caveclient.materialize.version = caveclient.materialize.most_recent_version()

    if tokens[0].endswith('?'):  # Query
        # Loop through tables backwards, so if segment is found in the later
        # tables, a message will be returned and the earlier tables won't be
        # searched
        for table in proofreading_tables[::-1]:
            try:
                query_result = fanc.lookup.is_proofread(segid, table,
                                                         return_previous_ids=True)
            except Exception as e:
                return f"`{type(e)}`\n```{e}```"

            if query_result == 1:
                return (f"Yes, segment {segid} is `{table}`.")
            elif isinstance(query_result, list):
                return (f"A previous version(s) of segment {segid} was found in"
                        f" `{table_name}`: {proofreading_status}.\nThis means it"
                        " was marked as proofread at some point but then edited"
                        " afterward, and the new version has not yet been marked"
                        " as proofread.")

            if query_result != 0:
                return ValueError(f'Unexpected query results from'
                                  f' table `{table}`:\n{query_result}')

        return f"No, segment {segid} is not marked as proofread."

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
        if fanc.lookup.is_proofread(segid, table_name) == 1:
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
            return (f"ERROR: Segment {segid} has no entry in the soma"
                    " table.\n\nIf you clearly see a soma attached to"
                    " this object, probably the automated soma detection"
                    " algorithm missed this soma. If so, message Sumiya"
                    " Kuroda and he can add it to the soma table."
                    "\n\nIf you're sure this is a descending neuron or"
                    " a sensory neuron, you can specify a point to"
                    " anchor the proofreading annotation. Call 'help'"
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
                proofread=True,
                pt_position=point,
                user_id=cave_user_id,
                valid_id=segid
            )
        except Exception as e:
            return f"ERROR: Staging failed with error\n`{type(e)}`\n```{e}```"

        if fake:
            return (f"FAKE: Would upload segment {segid} and point"
                    f" coordinate `{point}` to `{table_name}`.")
        try:
            response = caveclient.annotation.upload_staged_annotations(stage)
            record_upload(segid, cave_user_id, table_name)
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
    uploads_fn = f'proofreading_status_bot_uploads_{table_name}.txt'
    with open(uploads_fn, 'a') as f:
        f.write(f'{segid},{user}\n')


def have_recently_uploaded(segid, table_name) -> bool:
    uploads_fn = f'proofreading_status_bot_uploads_{table_name}.txt'
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
        try:
            direct_message_channels = slackclient.conversations_list(types='im')
        except Exception as e:
            print('Encountered exception: {} {}'.format(type(e), e))
            logfn = os.path.join('exceptions_proofreading_status_bot',
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
                logfn = os.path.join('exceptions_proofreading_status_bot',
                                     datetime.now().strftime('%Y-%h-%d_%H-%M-%S.txt'))
                with open(logfn, 'w') as f:
                    f.write('{}\n{}'.format(type(e), e))
                time.sleep(50)
        time.sleep(10)
