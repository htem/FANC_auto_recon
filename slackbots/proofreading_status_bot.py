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
    token = os.environ['SLACK_TOKEN_FANC_PROOFREADINGSTATUSBOT']
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
""")


def process_message(message: str) -> str:
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
        return ("Doing nothing: Your message is empty or I couldn't understand"
                " it. Make a post containing the word 'help' if needed.")
    segid = tokens[0][:-1]
    if tokens[0].endswith('?'):
        return f"Querying {segid} (to be implemented)"
    elif tokens[0].endswith('!'):
        return f"Posting {segid} (to be implemented)"
    else:
        return ("Doing nothing: The first word in your message isn't a segment"
                " ID terminated by a ! or a ?. Make a post containing the word"
                " 'help' if needed.")
    

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

        if verbosity >= 1:
            print('Processing message with timestamp', message['ts'])

        if response is None:
            response = process_message(message['text'].strip('<@U04PUHVDSLX>'))


        if verbosity >= 2:
            print('Slack user post:', message)
            print('Slack bot post:', response)
        if fake:
            print('fake=True, not posting message')
            return
        slackclient.chat_postMessage(
            channel=channel_id,
            thread_ts=message['ts'],
            text=response
        )

if __name__ == '__main__':
    while True:
        print(datetime.now().strftime('%A %Y-%h-%d %H:%M:%S'))
        try:
            fetch_messages_and_post_replies(verbosity=2, fake=False)
        except Exception as e:
            print('Encountered exception: {} {}'.format(type(e), e))
            logfn = os.path.join('exceptions_proofreading_status_bot', datetime.now().strftime('%Y-%h-%d_%H-%M-%S') + '.txt')
            with open(logfn, 'w') as f:
                f.write('{}\n{}'.format(type(e), e))
            time.sleep(50)
        time.sleep(10)
