#!/usr/bin/env python3
import os
import logging
from flask import Flask, json, request
from slack_sdk.web import WebClient
from slackeventsapi import SlackEventAdapter
from triggers import get_response_key
from response_map import response_dict
import pandas as pd
import threading
import ssl as ssl_lib
import ssl
import certifi
import random
import time
import re
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize a Flask app to host the events adapter
app = Flask(__name__)
slack_events_adapter = SlackEventAdapter(os.environ['SLACK_SIGNING_SECRET'], "/slack/events", app)

# Initialize a Web API client
slack_web_client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])



default_response = 'I do not understand. Here are some things you can ask for:\n 1. get upstream partners of 123456789\n 2. get downstream partners of 123456789 with a 10 synapse threshold\n 3. get top 10 upstream partners of 123456789\n 4. get all annotation tables\n 5. get annotation tables by bmark89@uw.edu\n 6. get annotation table: T1MN_somas\n 7. find neuron annotated with: MN\n 8. get neuroglancer link for skeleton ID 12345 in project 13'
    
    
def parse_args(command):
    ##TODO: Add find neuron in an individual table. 
    ##TODO: Add materialization across multiple segmentations.
    ##TODO: Find regex that actually works in find neuron / get annotation table. I do not know why '(?<=").*(?=")' does not work. 
    if 'annotation' in command or 'annotated' in command:
        
        args = {'table_name':None,
                'annotation':None,
                'user':None}
        if 'user' in command or 'by' in command:
            if 'mailto' in command:
                user = command[command.find(':')+1:command.find('|')]
            else:
                user = re.findall('(?<=by )[\w]+',command)
            args['user'] = user
        if 'annotated with' in command or 'find annotation' in command:
            args['annotation'] = command[command.find(':')+2:]
        
        if 'get annotation table:' in command:
            args['table_name'] = command[command.find(':')+2:]
    
    elif 'partners' in command:       
        match = [int(i)for i in re.findall('[0-9]+',command)]
        args = {'cutoff':None, 
                'root_id':None, 
                'threshold': None}
        
        if 'top' in command:
            args['cutoff'] = match[0]
            match = match[1:]
        if 'threshold' in command:
            args['threshold'] = match[-1]
            match = match[0:-1]
        args['root_id'] = match[0]
    
    if len(re.findall('skeleton [IiDd]+',command))>0:
        values = re.findall('[0-9]+',command)
        if len(values)>1:
            args = {'skid':int(values[0]),'project':int(values[1])}
        else:
            args = {'skid':int(values[0]),'project':13}
    
    return(args)



def parse_input(command):
    
    response = get_response_key(command)
    
    if response:
        
        args = parse_args(command)
        use_args = {}
  
        for k,v in args.items():
            if args[k]:
                use_args[k] = v
   
        if len(use_args) > 0:
            return response,use_args
        else:
            return response,None

    
    else:
        return(None,None)
        

def handle_command(command):
    
    response,use_args = parse_input(command)
    
    if response:
        if use_args:
            return(response_dict[response][0](**use_args))
        else:
            return response_dict[response][0]()
    else:
            return default_response



def payload_delivery(response,user_id,channel_id):
    """Create and send an onboarding welcome message to new users. Save the
    time stamp of this message so we can update this message in the future.
    """
    
    if isinstance(response,pd.core.frame.DataFrame):
        fname = str(random.randint(1111111,9999999)) + '.csv'
        response.to_csv(fname)
        r = slack_web_client.files_upload(
              channels=channel_id,
              filetype='csv',
              filename=fname,
              title='query response',
              file=fname)
        os.remove(fname)
    
    elif isinstance(response,pd.core.series.Series) or isinstance(response,list):
        payload = {
            "channel": channel_id,
            "username": "connectomics_bot",
            "text": 'Query Response:',
            "blocks": [
                {"type": "section",
                    "text": {
                    "type": "plain_text",
                        "text": str(response)}}]}
        r = slack_web_client.chat_postMessage(**payload)
    else:
        payload = {
                "channel": channel_id,
                "username": "connectomics_bot",
                "text": 'Query Response:',
                "blocks": [
                    {"type": "section",
                        "text": {
                        "type": "plain_text",
                            "text": response}}]}
        
        r = slack_web_client.chat_postMessage(**payload)
        
        return(r)


    

# ============== Message Events ============= #
# When a user sends a DM, the event type will be 'message'.
# Here we'll link the message callback to the 'message' event.
@app.route('/', methods=['POST'])

@slack_events_adapter.on("message")
def message(payload):
    """Collect DMs to the bot.
    """
    event = payload.get("event", {})

    channel_id = event.get("channel")
    user_id = event.get("user")
    text = event.get("text")
    
   
    sender = payload['authorizations'][0]['user_id']
    receiver = user_id

    print(text)
    if event.get("channel_type") == 'im' and sender != receiver and 'bot_id' not in event:
        payload = {
            "channel": channel_id,
            "username": "connectomics_bot",
            "text": 'Getting data...'}
        r = slack_web_client.chat_postMessage(**payload)
        
        x = threading.Thread(
            target=initiate_process,
            args=(text,user_id,channel_id))
        x.start()
             
        return 
    
    else:
        return 
    
def initiate_process(text,user_id,channel_id):
    
    response = handle_command(text)
    
    r = payload_delivery(response,user_id,channel_id)
    
    payload = {
        "channel": channel_id,
        "username": "connectomics_bot",
        "text": 'Complete!'}


    slack_web_client.chat_postMessage(**payload)
    
    return
    
        
    
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    app.run(port=3000)