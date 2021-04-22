import os 
import time 
from slackclient import SlackClient
from response_map import response_dict
from triggers import get_response_key


# Bot settings & connection handlers
BOT_NAME = "testbot2"
slack_client = SlackClient(os.environ.get("SLACK_BOT_TOKEN"))


def handle_command(command, channel):
    """Take cleaned command, look up response, and post to channel"""

    # Check for exact matches first then check for contained keywords
    response_key = get_response_key(command)

    # Default behavior if no response keys match the command
    if not (response_key or command):
        # If no command is given
        response_key = 'no_command'
    elif not response_key:
        response_key = 'search'
        command = 'search ' + command

    response = random.choice(response_dict[response_key])

    # If response is a function, call it with command as argument
    if callable(response):
        response = response(command)

    slack_client.api_call("chat.postMessage", channel=channel, text=response, as_user=True)


def parse_slack_output(slack_rtm_output):
    """Parses output data from Slack message stream"""

    # read data from slack channels
    output_list = slack_rtm_output

    if output_list and len(output_list) > 0:
        for output in output_list:

            if output and 'text' in output:
                text = output['text']

                # if bot name is mentioned, take text to the right of the mention as the command
                if BOT_NAME in text.lower():
                    return text.lower().split(BOT_NAME)[1].strip().lower(), output['channel']

    return None, None
                
if __name__ == "__main__":
    if slack_client.rtm_connect():
        print("{BOT_NAME} now online.")

        while True:
            text_input, channel = parse_slack_output(slack_client.rtm_read())
            if text_input and channel:
                handle_command(text_input, channel)
            time.sleep(1)  # websocket read delay

    else:
        print("Connection failed.")
        
        
