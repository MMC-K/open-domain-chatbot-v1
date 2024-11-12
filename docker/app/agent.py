import logging
from llama_cpp import Llama
import custom_llama_cpp_chat_format
from custom_llama_cpp_chat_format import function_calling_template_v050
import json

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHAT_FORMAT = "llama-3.1-keti-v0.50-fc-fast"
MODEL_PATH = "model/model-bf16-Q8_0.gguf"
CONTEXT_LENGTH = 8192
MAX_GEN_LENGTH = CONTEXT_LENGTH//3
MAX_PREV_CONTEXT_LENGTH = CONTEXT_LENGTH -  MAX_GEN_LENGTH


class Service():
    def __init__(self) -> None:
        self.llm = Llama(
            model_path=MODEL_PATH,
            chat_format=CHAT_FORMAT,
            n_ctx=CONTEXT_LENGTH,
            n_gpu_layers=-1,
        )
        self.max_prev_context_length = MAX_PREV_CONTEXT_LENGTH
        
        self.template_renderer = ImmutableSandboxedEnvironment(
            undefined=jinja2.StrictUndefined,
        ).from_string(function_calling_template_v050)
        
    
    def chat_completions(self, 
                         messages,
                         **kwargs):
        messages = self.remove_old_context(messages, **kwargs)
        
        logger.error(messages)
        
        response = self.llm.create_chat_completion(
            messages=messages,
            **kwargs
        )
        return response

    def remove_old_context(self, messages, tools=None, **kwargs):
        
        if (tools is None or len(tools) == 0):
            tool_calls = False
            tools = []
        else:
            tool_calls = True
        
        origin_prompt = self.template_renderer.render(
            messages=messages,
            tools=tools,
            tool_calls=tool_calls,
            add_generation_prompt=True,
        )
        origin_tks = self.llm.tokenize(
                origin_prompt.encode("utf-8"),
                add_bos=False
            )
        
        if len(origin_tks) < self.max_prev_context_length:
            return messages
        
        res = self.get_context_str_len(messages, tools)
        cur_len = res[0]
        new_msg = []
        
        zip_list = [i for i in zip(res[1:], messages[1:])]
        for l, msg in reversed(zip_list):
            if cur_len + l < self.max_prev_context_length:
                new_msg.insert(0, msg)
                cur_len+=l
        new_msg.insert(0, messages[0])
        return new_msg
        

    def get_context_str(self, messages, tools=None):
        if (tools is None or len(tools) == 0):
            tool_calls = False
            tools = []
        else:
            tool_calls = True
        
        new_comp_str = [
            self.template_renderer.render(
                messages=messages[:1],
                tools=tools,
                tool_calls=tool_calls,
                add_generation_prompt=False,
            )
        ]
        
        for msg in messages[1:]:
            new_comp_str.append(self.template_renderer.render(
                messages=[msg],
                tools=[],
                tool_calls=False,
                add_generation_prompt=False,
            ))
        return new_comp_str
        
    
    def get_context_str_len(self, messages, tools=None):
        ctx_str_list = self.get_context_str(messages, tools)
        ctx_len = [
            len(self.llm.tokenize(
                line.encode("utf-8"),
                add_bos=False
            )) for line in ctx_str_list
        ]
        return ctx_len
        
    
    



if __name__=="__main__":
    api_service = Service()
    
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
        { "role": "user", "content": """What's the weather in San Francisco and New York today and the likelihood it'll rain?""" }
    ]
    messages = [
        { "role": "user", "content": """Please give me a information about Seoul""" }
    ]
    messages = [
        { "role": "system", "content": "You are a weather bot. Use the provided functions to answer questions." },
        { "role": "user", "content": """What's the weather in Seoul today and the likelihood it'll rain?""" }
    ]
    messages = [
        { "role": "system", "content": "You are a weather bot. Use the provided functions to answer questions." },
        { "role": "user", "content": """What's the weather in Seoul and New York today and the likelihood it'll rain?""" }
    ]
    
    kwargs = {
        "tools": tools,
        "tool_choice": "auto",
    }


    response = api_service.chat_completions(
        messages=messages,
        **kwargs,
    )

    print(response)



