import json
import requests
from urllib.parse import urljoin

import pprint

URL = 'http://127.0.0.1:5000'

# test task
task_q = '/api/chat'

tools=[
        {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature for a specific location",
            "parameters": {
            "type": "object",
            "properties": {
                "location": {
                "type": "string",
                "description": "The city and state, e.g., San Francisco, CA"
                },
                "unit": {
                "type": "string",
                "enum": ["Celsius", "Fahrenheit"],
                "description": "The temperature unit to use. Infer this from the user's location."
                }
            },
            "required": ["location", "unit"]
            }
        }
        },
        {
        "type": "function",
        "function": {
            "name": "get_rain_probability",
            "description": "Get the probability of rain for a specific location",
            "parameters": {
            "type": "object",
            "properties": {
                "location": {
                "type": "string",
                "description": "The city and state, e.g., San Francisco, CA"
                }
            },
            "required": ["location"]
            }
        }
        }
    ]
    
messages = [
    { "role": "system", "content": "You are a weather bot. Use the provided functions to answer questions." },
    { "role": "user", "content": """What's the weather in New York today and the likelihood it'll rain?""" }
]

# messages = [
#     { "role": "system", "content": "You are a weather bot. Use the provided functions to answer questions." },
#     { "role": "user", "content": """What's the likelihood it'll rain in New York today?""" }
# ]

# messages = [
#     { "role": "system", "content": "You are a weather bot. Use the provided functions to answer questions." },
#     { "role": "user", "content": """What's the weather in Seoul today and the likelihood it'll rain?""" }
# ]

messages = [
    { "role": "system", "content": "You are a weather bot. Use the provided functions to answer questions." },
    { "role": "user", "content": """What's the likelihood it'll rain in New York and Seoul today?""" }
]


kwargs = {
    "messages": messages, 
    "tools": tools,
    "tool_choice": "auto",
}

# default test case
data = json.dumps(
    kwargs
)
headers = {'Content-Type': 'application/json; charset=utf-8'}  # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.status_code)
print(response.request)
pprint.pprint(response.json())