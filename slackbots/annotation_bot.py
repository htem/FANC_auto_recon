#!/usr/bin/env python3
"""
This Slack app uses the "socket mode" feature of Slack's Bolt framework.
This allows the app to receive messages from Slack without needing to
have your own public server. Some useful links that describe this:
- https://api.slack.com/apis/connections/socket
- https://api.slack.com/apis/connections/events-api

--- Getting started ---
Install the slack bolt python package: `pip install slack-bolt`

View and configure your Slack app: https://api.slack.com/apps
- In Features > App Home > Show Tabs, select "Allow users to send
    Slash commands and messages from the messages tab" to enable DMs.
- In Settings > Socket Mode, enable Socket Mode. Create a token with
    connections:write permissions if prompted to. You can name the token
    anything, but 'websockets' is a reasonable choice.
- In Features > Event Subscriptions, toggle Enable Events on. Then
    open "Subscribe to bot events" and add the following events:
      message.im
    Press "Save Changes" when done.

Get your app's tokens:
- From Settings > Basic Information > App Credentials, copy the Signing Secret.
    Add it to your shell environment by adding a line like this to your shell
    startup file (e.g. ~/.bashrc, ~/.zshrc):
      export SLACK_BOT_SIGNING_SECRET=abcdef1234567890...
- From Settings > Basic Information > App-Level Tokens, click on the token you
    made earlier (e.g. 'websockets'). Copy the token. Add it to your shell
    startup file (e.g. ~/.bashrc, ~/.zshrc):
      export SLACK_BOT_WEBSOCKETS_TOKEN=xapp-abcdef1234567890...
- From Features > OAuth & Permissions > OAuth Tokens for Your Workspace,
    copy your Bot User OAuth Token and add it to your shell startup file:
      export SLACK_BOT_TOKEN=xoxb-abcdef1234567890...

Then run this script with `python proofreading_status_bot.py` to start
listening for events triggered by users interacting with your Slack app.

If you want to keep this running constantly so that the app is always
listening and responding, you can run this script in the background
using a utility like `screen`.
"""

import os
import sys
import json
import re
from datetime import datetime, timezone
from typing import Union

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import fanc

# Setup
verbosity = 2
convert_given_point_to_anchor_point = False
annotate_recursively = True

caveclient = fanc.get_caveclient()
tables = ['neuron_information', 'proofread_first_pass', 'proofread_second_pass']

with open('slack_user_permissions.json', 'r') as f:
    permissions = json.load(f)

app = App(token=os.environ['SLACK_TOKEN_FANC_NEURONINFORMATIONBOT'],
          signing_secret=os.environ['SLACK_SIGNING_SECRET_FANC_NEURONINFORMATIONBOT'])
handler = SocketModeHandler(app, os.environ['SLACK_TOKEN_FANC_NEURONINFORMATIONBOT_WEBSOCKETS'])


def show_help():
    return (
"""
Hello! Before using me for the first time, you may want to read through:
- <https://github.com/htem/FANC_auto_recon/wiki/Neuron-annotations|the list of available annotations>
- <https://cave.fanc-fly.com/annotation/views/aligned_volume/fanc_v4/table/neuron_information|the description of the "neuron_information" CAVE table>
- <https://cave.fanc-fly.com/annotation/views/aligned_volume/fanc_v4/table/proofread_first_pass|the description of the "proofread_first_pass" CAVE table>
- <https://cave.fanc-fly.com/annotation/views/aligned_volume/fanc_v4/table/proofread_second_pass|the description of the "proofread_second_pass" CAVE table>

You can send me a message that looks like one of the `example messages below` to find certain types of neurons, or get or upload information about specific neurons.

Find neurons with some annotations:
- `find DNx01` -> Get a neuroglancer state showing all neurons currently annotated with "DNx01" (which should be exactly two neurons)
- `find chordotonal neuron and ascending` -> Get a neuroglancer state showing all neurons currently annotated with "chordotonal neuron" and "ascending"
- `find left T1 ventral nerve and motor neuron` -> Get a neuroglancer link that shows all neurons currently annotated with "left T1 ventral nerve" and "motor neuron"
- You can use as many search terms if you want, e.g. `find W and X and Y and Z`

Get information about a specific neuron:
- `648518346486614449?` -> get annotations for segment 648518346486614449
- `648518346486614449??` or `648518346486614449? all` -> get extended annotation details for segment 648518346486614449

Upload annotations to a CAVE table that the whole community can benefit from:
- `648518346486614449! primary class > central neuron` -> annotate that the indicated segment's "primary class" is "central neuron" (as opposed to "sensory neuron" or "motor neuron").
- `648518346489818455! left-right projection pattern > bilateral` -> annotate that segment 648518346489818455 projects bilaterally, i.e. has synaptic connections on both sides of the VNC's midplane.
(To upload annotations, Jasper needs to first give you permissions, so send him a message to ask if you're interested.)
- `648518346486614449! proofread first pass` -> annotate that segment 648518346486614449 has received a "first pass" of proofreading, meaning it has no major merge errors and most major branches have been extended.
- `648518346486614449! proofread second pass` -> annotate that segment 648518346486614449 has received a "second pass" of proofreading, meaning it has been inspected carefully by an experienced proofreader and the only remaining errors are tiny missing bits that wouldn't make a noticeable difference if added to the neuron.

Feel free to send <@ULH2UM0H4> any questions, suggestions, or bug reports!
""")


@app.event("message")
def direct_message(message, say, client):
    """
    Slack servers trigger this function when a user sends a direct message to the bot.

    'message' is a dictionary containing information about the message.
    'say' is a function that can be used to send a message back to the user.
    """
    print(datetime.now().strftime('%A %Y-%h-%d %H:%M:%S'))
    if message.get('channel_type', None) != 'im':
        # Skip if this is not a direct message
        return
    if message.get('subtype', None):
        # Skip if this is a system message (not something posted by a user)
        return
    if message.get('thread_ts', None):
        # Skip if this message has a reply already
        return
    if 'bot_id' in message:
        # Skip if this message was posted by another bot
        return

    if verbosity >= 2:
        print('Processing message:', message)
    elif verbosity >= 1:
        print('Processing message with timestamp', message['ts'])

    try:
        response = process_message(message['text'],
                                   message['user'],
                                   client=client,
                                   fake=fake)
    except Exception as e:
        response = f"`{type(e)}`\n```{e}```"

    if verbosity >= 1:
        print('Posting response:', response)
    if len(response) > 1500:
        say(response, thread_ts=message['ts'])
    else:
        say(response)


def process_message(message: str,
                    user: str,
                    client=None,
                    fake=False) -> str:
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
    while '  ' in message:
        message = message.replace('  ', ' ')

    # This is a dict specifying how to do different types of searches.
    # Each entry is a search name followed by a 3-tuple of:
    # - CAVE table name to search for neurons in
    # - A bounding box [[xmin, ymin, zmin], [xmax, ymax, zmax]] to
    #   restrict the search to, or None to search the whole table
    # - A list of annotations. Cells with any of these annotations
    #   will be _excluded_ from the search results.
    todos = {
        'left T1': (
            'somas_dec2022',
            [[2800, 75200, 600], [36000, 117000, 4250]],
            ['glia', 'proofread first pass', 'proofread second pass']),
    }

    if message.lower().startswith('todo'):
        if message.lower() == 'todo':
            message = 'todo left T1'
        elif ' ' not in message:
            return ("I couldn't understand your request."
                    " Type 'help' or 'todo help' for instructions.")
        roi = message[message.find(' ')+1:]
        if roi not in todos:
            msg = ("Here are the currently defined todos:\n```todo name: (CAVE table"
                   " name, coordinates of search region, tags to exclude)\n\n"
                   + "\n".join([f"{k}: {v}" for k, v in todos.items()]) +
                   f"```\nSend me the message `todo {list(todos)[0]}` to use the"
                   " first one, for example.")
            if "help" not in message:
                msg = (f"There is currently no todo defined for `{roi}`. If you'd"
                       " like one to be created, send Jasper a DM with the"
                       f" coordinates of the region you'd like to use for `{roi}`.\n"
                       + msg)
            return msg
        table_name, bounding_box, exclude_tags = todos[roi]
        if not isinstance(bounding_box, dict):
            bounding_box = {table_name: {'pt_position': bounding_box}}

        root_ids = caveclient.materialize.live_live_query(
            table_name,
            datetime.now(timezone.utc),
            filter_spatial_dict=bounding_box,
        ).pt_root_id
        root_ids = root_ids.loc[root_ids != 0]
        annos = fanc.lookup.annotations(root_ids)
        todos = [i for i, anno_list in zip(root_ids, annos) if
                 all([anno not in anno_list for anno in exclude_tags])]
        msg = (f"There are {len(todos)} `{roi}` cells that need"
               " proofreading and/or annotations")
        if len(todos) <= 5:
            return msg + ":\n```" + str(todos)[1:-1] + "```"
        import random
        return msg + ". Here are 5:\n```" + str(random.sample(todos, 5))[1:-1] + "```"

    if message.startswith(('get', 'find')):
        search_terms = message[message.find(' ')+1:].strip('"\'')

        if message.startswith(('getids', 'findids')):
            results = fanc.lookup.cells_annotated_with(search_terms,
                                                       return_as='list')
            if len(results) > 300:
                return (f"{len(results)} cells matched that search! Try a more"
                        " specific search (like `findids X and Y and Z`) to see"
                        " a list of IDs.")
            return f"Search successful:```{', '.join(map(str, results))}```"
        if message.startswith(('getnum', 'findnum')):
            results = fanc.lookup.cells_annotated_with(search_terms,
                                                       return_as='list')
            return f"Your search matched {len(results)} cells."

        return ("Search successful. View your results: " +
                fanc.lookup.cells_annotated_with(search_terms, return_as='url'))

    try:
        caveclient.materialize.version = caveclient.materialize.most_recent_version()
    except Exception as e:
        return ("CAVE appears to be offline. Please wait a few minutes"
                f" and try again: `{type(e)}`\n```{e}```")

    # Because HTML or something, the '>' character typed into slack
    # is reaching this code as '&gt;', so revert it for readability.
    message = message.replace('&gt;', '>')

    command_chars = ['?', '!', '-']
    try:
        command_index = min([message.find(char)
                             for char in command_chars
                             if char in message])
    except ValueError:
        if 'help' in message.lower():
            return show_help()
        return ("ERROR: Your message does not contain a `?`, `!`, or `-`"
                " character, so I don't know what you want me to do."
                " Make a post containing the word 'help' for instructions.")
    if 'help' in message[:command_index].lower():
        return show_help()
    command_char = message[command_index]

    if command_char == '?':
        if message.startswith('??', command_index):
            return_details = True
            message = message.replace('??', '?', 1)
        else:
            return_details = False

        neuron = message[:command_index]
        try:
            segid = int(neuron)
        except ValueError:
            try:
                point = [int(coordinate.strip(','))
                         for coordinate in re.split(r'[ ,]+', neuron)]
            except ValueError:
                return f"ERROR: Could not parse `{neuron}` as a segment ID or a point."
            segid = fanc.lookup.segid_from_pt(point)
        if not caveclient.chunkedgraph.is_latest_roots(segid):
            return (f"ERROR: {segid} is not a current segment ID."
                    " It may have been edited recently, or perhaps"
                    " you copy-pasted the wrong thing.")

        info = fanc.lookup.annotations(segid, return_details=return_details)
        if len(info) == 0:
            return "No annotations found."
        if return_details:
            info.drop(columns=['id', 'valid', 'pt_supervoxel_id',
                               'pt_root_id', 'pt_position', 'deleted',
                               'superceded_id'],
                      errors='ignore',
                      inplace=True)
            info.rename(columns={'tag': 'annotation',
                                 'tag2': 'annotation_class'}, inplace=True)
            info['created'] = info.created.apply(lambda x: x.date())
            return ('```' + info.to_string(index=False) + '```')
        else:
            return ('```' + '\n'.join(info) + '```')

    if command_char == '!':  # Upload
        neuron = message[:command_index]
        try:
            segid = int(neuron)
            neuron = segid
            try:
                point = fanc.lookup.anchor_point(segid)
            except Exception as e:
                return f"`{type(e)}`\n```{e}```"
        except ValueError:
            point = [int(coordinate.strip(','))
                     for coordinate in re.split(r'[ ,]+', neuron)]
            segid = fanc.lookup.segid_from_pt(point)
            if convert_given_point_to_anchor_point:
                point = fanc.lookup.anchor_point(segid)
            neuron = point

        if not caveclient.chunkedgraph.is_latest_roots(segid):
            return (f"ERROR: {segid} is not a current segment ID."
                    " It may have been edited recently, or perhaps"
                    " you copy-pasted the wrong thing.")
        annotation = message[command_index+1:].strip()
        invalidity_errors = []
        for table in tables:
            if annotation.replace(' ', '_').replace('-', '_') == table:
                annotation = True
            try:
                if not fanc.annotations.is_valid_annotation(annotation,
                                                            table_name=table,
                                                            response_on_unrecognized_table=True,
                                                            raise_errors=True):
                    raise ValueError(f'Invalid annotation "{annotation}"'
                                     f' for table "{table}".')
            except Exception as e:
                invalidity_errors.append(e)
                continue

            # Permissions
            table_permissions = permissions.get(table, None)
            if table_permissions is None:
                return f"ERROR: `{table}` not listed in permissions file."
            cave_user_id = table_permissions.get(user, None)
            if cave_user_id is None:
                return ("You have not yet been given permissions to post to"
                        f" `{table}`. Please send Jasper a DM on slack"
                        " to request permissions.")

            if fake:
                fanc.annotations.is_allowed_to_post(segid, annotation,
                                                    response_on_unrecognized_table=True,
                                                    table_name=table)
                return (f"FAKE: Would upload segment {segid}, point"
                        f" `{list(point)}`, annotation `{annotation}`"
                        f" to table `{table}`.")
            try:
                annotation_ids = fanc.upload.annotate_neuron(
                    neuron, annotation, cave_user_id, table_name=table,
                    recursive=annotate_recursively,
                    convert_given_point_to_anchor_point=convert_given_point_to_anchor_point
                )
                uploaded_data = caveclient.annotation.get_annotation(table,
                                                                     annotation_ids)
                msg = (f"Upload to `{table}` succeeded:\n"
                       f"- Segment {segid}\n"
                       f"- Point coordinate `{uploaded_data[0]['pt_position']}`\n")
                for anno in uploaded_data:
                    if msg.count('\n') > 3:
                        msg += '\n\n'
                    msg += f"- Annotation ID: {anno['id']}"
                    if 'proofread' in anno:
                        msg += f"\n- Annotation: `{anno['proofread']}`"
                        record_upload(anno['id'], segid,
                                      anno['proofread'],
                                      cave_user_id, table)
                    elif 'tag' in anno and 'tag2' in anno:
                        msg += f"\n- Annotation: `{anno['tag']}`"
                        msg += f"\n- Annotation class: `{anno['tag2']}`"
                        record_upload(anno['id'], segid,
                                      anno['tag2'] + ': ' + anno['tag'],
                                      cave_user_id, table)
                    elif 'tag' in anno:
                        msg += f"\n- Annotation: `{anno['tag']}`"
                        record_upload(anno['id'], segid,
                                      anno['tag'],
                                      cave_user_id, table)
                    else:
                        msg = (msg + "\n\nWARNING: Something went wrong with recording"
                               " your upload on the slackbot server. Please send Jasper"
                               " a screenshot of your message and this response.")
                return msg
            except Exception as e:
                return f"ERROR: Annotation failed due to\n`{type(e)}`\n```{e}```"

        msg = (f"ERROR: Annotation `{annotation}` is not valid for any of the"
               " CAVE tables I know how to post to:")
        for table, e in zip(tables, invalidity_errors):
            msg += f"\n\nTable `{table}` gave `{type(e)}`:\n```{e}```"
        return msg
    if command_char == '-':  # Delete annotation
        neuron = message[:command_index]
        try:
            segid = int(neuron)
        except ValueError:
            try:
                point = [int(coordinate.strip(','))
                         for coordinate in re.split(r'[ ,]+', neuron)]
            except ValueError:
                return f"ERROR: Could not parse `{neuron}` as a segment ID or a point."
            segid = fanc.lookup.segid_from_pt(point)
        if not caveclient.chunkedgraph.is_latest_roots(segid):
            return (f"ERROR: {segid} is not a current segment ID."
                    " It may have been edited recently, or perhaps"
                    " you copy-pasted the wrong thing.")

        annotation = message[command_index+1:].strip()

        user_id = None
        for table in permissions:
            if user in permissions[table]:
                user_id = permissions[table][user]
                break
        if user_id is None:
            return ("You have not yet been given permissions to delete"
                    " annotations. Please send Jasper a DM on slack"
                    " to request permissions.")
        try:
            response = fanc.upload.delete_annotation(segid, annotation, user_id)
            return (f'Successfully deleted annotation with ID {response[1]}'
                    f' from table `{response[0]}`.')
        except Exception as e:
            return f"ERROR: Deletion failed due to\n`{type(e)}`\n```{e}```"


def record_upload(annotation_id, segid, annotation, user_id, table_name) -> None:
    uploads_fn = f'annotation_bot_uploads_to_{table_name}.csv'
    with open(uploads_fn, 'a') as f:
        f.write(f'{annotation_id},{segid},{annotation},{user_id}\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fake':
        fake = True
        print('Running in FAKE mode')
    else:
        fake = False
    handler.start()
