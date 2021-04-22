import os
from slack_sdk.rtm.v2 import RTMClient

token = 'xoxb-1592070178098-1899862535489-4tfcUhYzmqer9Ow9uZjOSosZ'
rtm = RTMClient(token=token)

@rtm.on("message")
def handle(client: RTMClient, event: dict):
    if 'Hello' in event['text']:
        channel_id = event['channel']
        thread_ts = event['ts']
        user = event['U01SFRCFRED'] # This is not username but user ID (the format is either U*** or W***)

        client.web_client.chat_postMessage(
            channel=channel_id,
            text=f"Hi <@{user}>!",
            thread_ts=thread_ts
        )

rtm.start()