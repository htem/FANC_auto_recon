import os
import pandas as pd
import sys
from slackclient import SlackClient
import json
from pathlib import Path
import time

HOME = Path.cwd()
fname = HOME / 'api_key.json'
with open(fname,mode='r') as f:
    key = json.load(f)


#Initiate
apikey = key['key']
sc = SlackClient(apikey)
starterbot_id = None
channel_name = 'test_script'

def find_id(chnnl):
    channels = sc.api_call("conversations.list")
    channels = channels['channels']
    for i in range(0, len(channels)):
        z = channels[i]
        if z['name'] == chnnl:
            break
    return channels[i]['id']

def get_messages(chnnl):
    channel_id = find_id(chnnl)
    convos = sc.api_call("conversations.history", channel=channel_id)
    return convos['messages']

def send_message(chnnl, message):
    channel_id = find_id(chnnl)
    sc.api_call("chat.postMessage", channel=channel_id, text=message)
    return chanel_id

def parse_bot_commands(slack_events):
    """
        Parses a list of events coming from the Slack RTM API to find bot commands.
        If a bot command is found, this function returns a tuple of command and channel.
        If its not found, then this function returns None, None.
    """
    for event in slack_events:
        if event["type"] == "message" and not "subtype" in event:
            user_id, message = parse_direct_mention(event["text"])
            if user_id == starterbot_id:
                return message, event["channel"]
    return None, None


def handle_command(command, channel):
    """
        Executes bot command if the command is known
    """
    # Default response is help text for the user
    default_response = "Not sure what you mean. Try *{}*.".format(EXAMPLE_COMMAND)

    # Finds and executes the given command, filling in response
    response = None
    # This is where you start to implement more commands!
    if command.startswith(EXAMPLE_COMMAND):
        response = "Sure...write some more code then I can do that!"

    # Sends the response back to the channel
    slack_client.api_call(
        "chat.postMessage",
        channel=channel,
        text=response or default_response
    )
    

def parse_messages(msgs):
    input_options = ['does this work?','help']
    print(len(msgs))
    for i in msgs:
        if i['text'] in input_options:
            print('Success!')
        else:
            print('Failure')
        
def script():
    msgs = get_messages(channel_name)
    parse_messages(msgs)
    
if __name__ == '__main__':

    
    while True:
        script()
        print ('I schleep')
        time.sleep(10)