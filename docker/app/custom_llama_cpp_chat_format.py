from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union, Protocol, cast
import json
import re
import uuid
import logging

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

import llama_cpp.llama as llama
import llama_cpp.llama_types as llama_types
import llama_cpp.llama_grammar as llama_grammar
from llama_cpp.llama_chat_format import (
    register_chat_format,
    register_chat_completion_handler,
    ChatFormatterResponse,
    _format_no_colon_single,
    _grammar_for_response_format,
    _convert_completion_to_chat,
    _convert_completion_to_chat_function,
)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def _map_roles_and_fmt_content(
    messages: List[llama_types.ChatCompletionRequestMessage],
    role_map: Dict[str, str],
    special_role: Dict[str, Dict[str, str]],
) -> List[Tuple[str, Optional[str]]]:
    """Map the message roles."""
    output: List[Tuple[str, Optional[str]]] = []
    for message in messages:
        role = message["role"]
        if role in role_map:
            content: str | None = (
                message["content"] if isinstance(message["content"], str) else None
            )
            output.append((role_map[role], content))
        elif role in special_role:
            content: str = special_role[role]["start_token"]+message["content"]+special_role[role]["end_token"]
            output.append((role_map[special_role[role]["role_map"]], content))
    return output

@register_chat_format("llama-3-keti-v0.42")
def format_llama3_keti_v042(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    _roles = dict(
        system="<|start_header_id|>system<|end_header_id|>\n\n",
        user="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    _special_role = {
        "tool_calls": {
            "start_token": "<|start_tool_calls|>",
            "end_token": "<|end_tool_calls|>",
            "role_map": "assistant",
        },
        "function": {
            "start_token": "<|start_tool_response|>",
            "end_token": "",
            "role_map": "user",
        },
    }
    _sep = "<|eot_id|>"
    _messages = _map_roles_and_fmt_content(messages, _roles, _special_role)
    _messages.append((_roles["assistant"], None))
    _prompt = _format_no_colon_single("", _messages, _sep)
    return ChatFormatterResponse(prompt=_prompt, stop=_sep)


function_template_v043 = (
        # tool response
        "name: {{ name }}\n"
        "arguments: {{ arguments }}\n"
        "results: {{ results }}\n"
    )
function_template_renderer_v043 = ImmutableSandboxedEnvironment(
    undefined=jinja2.StrictUndefined,
).from_string(function_template_v043)


def formatting_tool_message_v043(messages):
    tool_call_dict = {}
    new_messages = []
    for utter_idx, utter in enumerate(messages):
        if "tool_calls" in utter:
            for tc in utter["tool_calls"]:
                tool_call_dict[tc["id"]] = tc["function"]
        if utter["role"] == "tool":
            src_tool_call = tool_call_dict[utter["tool_call_id"]]
                
            function_prompt = function_template_renderer_v043.render(
                name=src_tool_call["name"],
                arguments=src_tool_call["arguments"],
                results=json.dumps(json.loads(utter["content"]), ensure_ascii=False),
            )
            new_messages.append({"role": "user", "content": function_prompt.strip()})
        else:
            new_messages.append(utter)
    return new_messages

function_calling_template_v043 = (
        # Set system prompt
        "{% if messages[0].role == 'system' %}"
        '{% set system_prompt = messages[0].content %}'
        '{% set messages = messages[1:] %}'
        "{% elif tool_calls %}"
        "{% set system_prompt = '' %}"
        "{% else %}"
        "{% set system_prompt = none %}"
        "{% endif %}"
        
        # System message
        "{% if system_prompt is not none %}"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "{% if system_prompt != '' %}"
        "{{ system_prompt }}"
        "{% endif %}"
        # System message - tool calls
        "{% if tool_calls %}"
        "{% if system_prompt != '' %}"
        "\n\n"
        "{% endif %}"
        "You have access to the following functions:\n"
        "{% for tool in tools %}"
        "\nfunctions.{{ tool.function.name }}:\n"
        "description: {{ tool.function.description }}\n"
        "arguments: {{ tool.function.parameters | tojson }}"
        "\n{% endfor %}"
        "\n\nYou can respond to users messages with either a single message or one or more function calls."
        "\n\nTo respond with a message begin the message with 'message:', use the following format:"
        "\n\nmessage:"
        "\n<message>"
        "\n\nTo respond with one or more function calls begin the message with 'functions.<function_name>:', use the following format:"
        "\n\nfunctions.<function_name>:"
        '\n{ "arg1": "value1", "arg2": "value2" }'
        "\nfunctions.<function_name>:"
        '\n{ "arg1": "value1", "arg2": "value2" }'
        "{% endif %}"
        "<|eot_id|>"
        "{% endif %}"
        # Other messages
        "{% for message in messages %}"
        "<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n"
        # User message
        "{% if message.role == 'user' %}"
        "{{ message.content }}"
        "<|eot_id|>"
        "{% endif %}"
        # Assistant message
        "{% if message.role == 'assistant' %}"
        ## Reglar message
        "{% if message.content and message.content | length > 0 %}"
        "{% if tool_calls %}"
        "message:\n"
        "{% endif %}"
        "{{ message.content }}"
        "<|eot_id|>"
        "{% endif %}"
        ## Function calls
        "{% if 'tool_calls' in message %}"
        "{% for tool_call in message.tool_calls %}"
        "functions.{{ tool_call.function.name }}:\n"
        "{{ tool_call.function.arguments }}\n"
        "{% endfor %}"
        "<|eot_id|>"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
    )


@register_chat_completion_handler("llama-3-keti-v0.43-fc")
def keti_function_calling_v043(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    **kwargs,  # type: ignore
) -> Union[
    llama_types.CreateChatCompletionResponse,
    Iterator[llama_types.CreateChatCompletionStreamResponse],
]:
    function_calling_template = (
        # Set system prompt
        "{% if messages[0].role == 'system' %}"
        '{% set system_prompt = messages[0].content %}'
        '{% set messages = messages[1:] %}'
        "{% elif tool_calls %}"
        "{% set system_prompt = '' %}"
        "{% else %}"
        "{% set system_prompt = none %}"
        "{% endif %}"
        
        # System message
        "{% if system_prompt is not none %}"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "{% if system_prompt != '' %}"
        "{{ system_prompt }}"
        "{% endif %}"
        # System message - tool calls
        "{% if tool_calls %}"
        "{% if system_prompt != '' %}"
        "\n\n"
        "{% endif %}"
        "You have access to the following functions:\n"
        "{% for tool in tools %}"
        "\nfunctions.{{ tool.function.name }}:\n"
        "description: {{ tool.function.description }}\n"
        "arguments: {{ tool.function.parameters | tojson }}"
        "\n{% endfor %}"
        "\n\nYou can respond to users messages with either a single message or one or more function calls."
        "\n\nTo respond with a message begin the message with 'message:', use the following format:"
        "\n\nmessage:"
        "\n<message>"
        "\n\nTo respond with one or more function calls begin the message with 'functions.<function_name>:', use the following format:"
        "\n\nfunctions.<function_name>:"
        '\n{ "arg1": "value1", "arg2": "value2" }'
        "\nfunctions.<function_name>:"
        '\n{ "arg1": "value1", "arg2": "value2" }'
        "{% endif %}"
        "<|eot_id|>"
        "{% endif %}"
        # Other messages
        "{% for message in messages %}"
        "<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n"
        # User message
        "{% if message.role == 'user' %}"
        "{{ message.content }}"
        "<|eot_id|>"
        "{% endif %}"
        # Assistant message
        "{% if message.role == 'assistant' %}"
        ## Reglar message
        "{% if message.content and message.content | length > 0 %}"
        "{% if tool_calls %}"
        "message:\n"
        "{% endif %}"
        "{{ message.content }}"
        "<|eot_id|>"
        "{% endif %}"
        ## Function calls
        "{% if 'tool_calls' in message %}"
        "{% for tool_call in message.tool_calls %}"
        "functions.{{ tool_call.function.name }}:\n"
        "{{ tool_call.function.arguments }}\n"
        "{% endfor %}"
        "<|eot_id|>"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
    )
    template_renderer = ImmutableSandboxedEnvironment(
        undefined=jinja2.StrictUndefined,
    ).from_string(function_calling_template)
    
    messages = formatting_tool_message_v043(messages)

    # Convert legacy functions to tools
    if functions is not None:
        tools = [
            {
                "type": "function",
                "function": function,
            }
            for function in functions
        ]

    # Convert legacy function_call to tool_choice
    if function_call is not None:
        if isinstance(function_call, str) and (
            function_call == "none" or function_call == "auto"
        ):
            tool_choice = function_call
        if isinstance(function_call, dict) and "name" in function_call:
            tool_choice = {
                "type": "function",
                "function": {
                    "name": function_call["name"],
                },
            }

    stop = [stop, "<|eot_id|>"] if isinstance(stop, str) else stop + ["<|eot_id|>"] if stop else ["<|eot_id|>"]

    # Case 1: No tool choice by user
    if (
        tool_choice is None
        or (isinstance(tool_choice, str) and tool_choice == "none")
        or tools is None
        or len(tools) == 0
    ):
        prompt = template_renderer.render(
            messages=messages,
            tools=[],
            tool_calls=None,
            add_generation_prompt=True,
        )

        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(response_format)

        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
                logprobs=top_logprobs if logprobs else None,
            ),
            stream=stream,
        )

    # Case 2: Tool choice by user
    if isinstance(tool_choice, dict):
        tool_name = tool_choice["function"]["name"]
        tool = next(
            (tool for tool in tools if tool["function"]["name"] == tool_name), None
        )
        if tool is None:
            raise ValueError(f"Tool with name '{tool_name}' not found in tools")
        prompt = template_renderer.render(
            messages=messages,
            tools=tools,
            tool_calls=True,
            add_generation_prompt=True,
        )
        prompt += f"functions.{tool_name}:\n"
        try:
            grammar = llama_grammar.LlamaGrammar.from_json_schema(
                json.dumps(tool["function"]["parameters"]), verbose=llama.verbose
            )
        except Exception as e:
            grammar = llama_grammar.LlamaGrammar.from_string(
                llama_grammar.JSON_GBNF, verbose=llama.verbose
            )
            if llama.verbose:
                print(
                    "Failed to parse function body as JSON schema, falling back to default grammar"
                )
                print(e)
        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
        )
        return _convert_completion_to_chat_function(
            tool_name, completion_or_chunks, stream
        )

    # Case 3: Automatic tool choice
    assert isinstance(tool_choice, str) and tool_choice == "auto"
    function_names = " | ".join(
        [f'''"functions.{tool['function']['name']}:"''' for tool in tools]
    )
    initial_gbnf_tool_grammar = (
        """root   ::= functions | "message:"\n"""
        f"""functions ::= {function_names}\n"""
    )
    
    function_names_follow_up = " | ".join(
        [f'''[\n]"functions.{tool['function']['name']}:"''' for tool in tools]
    )
    
    follow_up_gbnf_tool_grammar = (
        """root   ::= functions | "<|eot_id|>"\n"""
        f"""functions ::= {function_names_follow_up}\n"""
    )
    
    prompt = template_renderer.render(
        messages=messages,
        tools=tools,
        tool_calls=True,
        add_generation_prompt=True,
    )
    completion_or_chunks = llama.create_completion(
        prompt=prompt,
        temperature=0,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        stream=False,
        stop=stop,
        max_tokens=None,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
        grammar=llama_grammar.LlamaGrammar.from_string(
            initial_gbnf_tool_grammar, verbose=llama.verbose
        ),
    )
    completion: llama_types.CreateCompletionResponse = completion_or_chunks  # type: ignore
    text = completion["choices"][0]["text"]
    
    if "message" in text:
        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt + "message:\n",
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=["<|eot_id|>"],
                logprobs=top_logprobs if logprobs else None,
                max_tokens=None,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                # grammar=llama_grammar.LlamaGrammar.from_string(
                #     follow_up_gbnf_tool_grammar, verbose=llama.verbose
                # ),
            ),
            stream=stream,
        )

    tool_dict_unique = []
    # One or more function calls
    tool_name = text[len("functions.") :-1]
    
    tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    if not stream:
        completions: List[llama_types.CreateCompletionResponse] = []
        completions_tool_name: List[str] = []
        while tool is not None:
            prompt += f"functions.{tool_name}:\n"
            try:
                grammar = llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(tool["function"]["parameters"], ensure_ascii=False), verbose=llama.verbose
                )
            except Exception as e:
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF, verbose=llama.verbose
                )
                if llama.verbose:
                    print(
                        "Failed to parse function body as JSON schema, falling back to default grammar"
                    )
                    print(e)
            completion_or_chunks = llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=False,
                stop=stop,
                max_tokens=None,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
            )
            completion_or_chunks = cast(llama_types.CreateCompletionResponse, completion_or_chunks)
            
            tool_dict = {
                "name": tool_name,
                "arguments": completion_or_chunks["choices"][0]["text"]
            }
            if tool_dict in tool_dict_unique:
                break
            else:
                tool_dict_unique.append(tool_dict)
            
            
            completions.append(completion_or_chunks)
            completions_tool_name.append(tool_name)
            prompt += completion_or_chunks["choices"][0]["text"]
            # prompt += "\n"

            response = llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=False,
                stop=stop,
                max_tokens=None,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=llama_grammar.LlamaGrammar.from_string(
                    follow_up_gbnf_tool_grammar, verbose=llama.verbose
                ),
            )
            response = cast(llama_types.CreateCompletionResponse, response)

            tool_name = response["choices"][0]["text"][len("functions.") :-1]
            tool = next(
                (tool for tool in tools if tool["function"]["name"] == tool_name), None
            )
            

        # Merge completions
        function_call_dict: Union[Dict[str, str], Dict[Literal["function_call"], llama_types.ChatCompletionRequestAssistantMessageFunctionCall]] = { 
            "function_call": {
                "name": completions_tool_name[0],
                "arguments": completions[0]["choices"][0]["text"],
            }
        } if len(completions) == 1 else {}
        return {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "logprobs": completion["choices"][0]["logprobs"],
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_"
                                + f"_{i}_"
                                + tool_name
                                + "_"
                                + completion["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": completion["choices"][0]["text"],
                                },
                            }
                            for i, (tool_name, completion) in enumerate(
                                zip(completions_tool_name, completions)
                            )
                        ],
                        **function_call_dict
                    },
                }
            ],
            "usage": {
                "completion_tokens": sum(
                    completion["usage"]["completion_tokens"] if "usage" in completion else 0
                    for completion in completions
                ),
                "prompt_tokens": sum(
                    completion["usage"]["prompt_tokens"] if "usage" in completion else 0
                    for completion in completions
                ),
                "total_tokens": sum(
                    completion["usage"]["total_tokens"] if "usage" in completion else 0
                    for completion in completions
                ),
            },
        }

    raise ValueError("Automatic streaming tool choice is not supported")


function_template_v046 = (
        # tool response
        "{{ content | tojson }}"
    )
function_template_renderer_v046 = ImmutableSandboxedEnvironment(
    undefined=jinja2.StrictUndefined,
).from_string(function_template_v046)


def formatting_tool_message_v046(messages):
    tool_call_dict = {}
    new_messages = []
    for utter_idx, utter in enumerate(messages):
        if "tool_calls" in utter:
            for tc in utter["tool_calls"]:
                tool_call_dict[tc["id"]] = tc["function"]
        if utter["role"] == "tool":
            try:
                src_tool_call = tool_call_dict[utter["tool_call_id"]]
                
                function_prompt = function_template_renderer_v046.render(
                    content={
                        "name": src_tool_call["name"],
                        "arguments": src_tool_call["arguments"],
                        "results": json.loads(utter["content"]),
                    }
                )
                new_messages.append({"role": utter["role"], "content": function_prompt.strip()})
            except Exception as e:
                logger.warning(f"there is no 'tool_call_id'[{utter['tool_call_id']}] in the previous tool_calls")
        else:
            new_messages.append(utter)
    return new_messages


formatting_tool_message_v050 = formatting_tool_message_v046

function_calling_template_v050 = (
        # Set system prompt
        "{% if messages[0].role == 'system' %}"
            '{% set system_prompt = messages[0].content %}'
            '{% set messages = messages[1:] %}'
        "{% elif tool_calls %}"
            "{% set system_prompt = '' %}"
        "{% else %}"
            "{% set system_prompt = none %}"
        "{% endif %}"
        
        # System message
        "{% if system_prompt is not none %}"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "{% if system_prompt != '' %}"
                "{{ system_prompt }}"
            "{% endif %}"
            # System message - tool calls
            "{% if tool_calls %}"
                "{% if system_prompt != '' %}"
                    "\n\n"
                "{% endif %}"
                
                "\n# Tool Instructions"
                "\n\nYou have access to the following functions:"
                
                "{% for tool in tools %}"
                    "\n\nUse the function '{{ tool.function.name }}' to: '{{ tool.function.description }}'\n"
                    "{{ tool.function | tojson }}"
                "{% endfor %}"
                
                "\n\n\nTo respond to the message, start the message with '<message>' and end it with '</message>'. Use the following format"
                "\n{start_msg_tag}{response}{end_msg_tag}"
                "\nwhere\n\n"
                "start_msg_tag => `<message>`"
                "\nresponse => Response to the user's message."
                "\nend_msg_tag => `</message>`"
                "\n\nHere is an example,"
                "\n<message>example_message</message>"
                "\n\n\nIf a you choose to call a function ONLY reply in the following format:"
                "\n<{start_tag}={function_name}>{parameters}{end_tag}"
                "\nwhere\n\n"
                "start_tag => `<function`"
                "\nparameters => a JSON dict with the function argument name as key and function argument value as value."
                "\nend_tag => `</function>`"
                "\n\nHere is an example,"
                "\n<function=example_function_name>{\"example_name\": \"example_value\"}</function>"
                "\n\nReminder:"
                "\n- Function calls MUST follow the specified format"
                "\n- Required parameters MUST be specified"
                "\n- You can respond to users messages with either a single message or one or more function calls."
                "\n- Put the entire function call reply on one line"
                "\n\nYou are a helpful Assistant."
            "{% endif %}"
            "<|eot_id|>"
        "{% endif %}"
        # Other messages
        "{% for message in messages %}"
            "{% if message.role == 'tool' %}"
                "<|start_header_id|>ipython<|end_header_id|>\n\n"
            "{% else %}"
                "<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n"
            "{% endif %}"
            # User message
            "{% if message.role in 'user' %}"
                "{{ message.content }}"
                "<|eot_id|>"
            "{% endif %}"
            # Tool message
            "{% if message.role == 'tool' %}"
                "{{ message.content }}"
                "<|eot_id|>"
            "{% endif %}"
            # Assistant message
            "{% if message.role == 'assistant' %}"
                ## Assistant - Reglar message
                "{% if message.content and message.content | length > 0 %}"
                    "{% if tool_calls %}"
                        "<message>{{ message.content }}</message>"
                    "{% else %}"
                        "{{ message.content }}"
                    "{% endif %}"
                    "<|eot_id|>"
                "{% endif %}"
                ## Assistant - Function calls
                "{% if 'tool_calls' in message %}"
                    "{% for tool_call in message.tool_calls %}"
                        "<function={{ tool_call.function.name }}>"
                        "{{ tool_call.function.arguments | tojson }}"
                        
                        "{% if message|length == loop.index + 1 %}"
                            "</function>"
                        "{% else %}"
                            "</function>\n"
                        "{% endif %}"
                    "{% endfor %}"
                "<|eot_id|>"
                ## Assistant - End function calls
                "{% endif %}"
            ## end Assistant message
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
    )


@register_chat_completion_handler("llama-3.1-keti-v0.50-fc")
def keti_function_calling_v050(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    **kwargs,  # type: ignore
) -> Union[
    llama_types.CreateChatCompletionResponse,
    Iterator[llama_types.CreateChatCompletionStreamResponse],
]:
    template_renderer = ImmutableSandboxedEnvironment(
        undefined=jinja2.StrictUndefined,
    ).from_string(function_calling_template_v050)
    
    messages = formatting_tool_message_v050(messages)

    # Convert legacy functions to tools
    if functions is not None:
        tools = [
            {
                "type": "function",
                "function": function,
            }
            for function in functions
        ]

    # Convert legacy function_call to tool_choice
    if function_call is not None:
        if isinstance(function_call, str) and (
            function_call == "none" or function_call == "auto"
        ):
            tool_choice = function_call
        if isinstance(function_call, dict) and "name" in function_call:
            tool_choice = {
                "type": "function",
                "function": {
                    "name": function_call["name"],
                },
            }

    stop = [stop, "<|eot_id|>", "<|eom_id|>"] if isinstance(stop, str) else stop + ["<|eot_id|>", "<|eom_id|>"] if stop else ["<|eot_id|>", "<|eom_id|>"]

    # Case 1: No tool choice by user
    if (
        tool_choice is None
        or (isinstance(tool_choice, str) and tool_choice == "none")
        or tools is None
        or len(tools) == 0
    ):
        prompt = template_renderer.render(
            messages=messages,
            tools=[],
            tool_calls=None,
            add_generation_prompt=True,
        )

        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(response_format)

        return _convert_completion_to_chat_v050(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
                logprobs=top_logprobs if logprobs else None,
            ),
            stream=stream,
        )

    # Case 2: Tool choice by user
    if isinstance(tool_choice, dict):
        tool_name = tool_choice["function"]["name"]
        tool = next(
            (tool for tool in tools if tool["function"]["name"] == tool_name), None
        )
        if tool is None:
            raise ValueError(f"Tool with name '{tool_name}' not found in tools")
        prompt = template_renderer.render(
            messages=messages,
            tools=tools,
            tool_calls=True,
            add_generation_prompt=True,
        )
        prompt += f"<function={tool_name}>"
        try:
            grammar = llama_grammar.LlamaGrammar.from_json_schema(
                json.dumps(tool["function"]["parameters"]), verbose=llama.verbose
            )
        except Exception as e:
            grammar = llama_grammar.LlamaGrammar.from_string(
                llama_grammar.JSON_GBNF, verbose=llama.verbose
            )
            if llama.verbose:
                print(
                    "Failed to parse function body as JSON schema, falling back to default grammar"
                )
                print(e)
        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
        )
        return _convert_completion_to_chat_function(
            tool_name, completion_or_chunks, stream
        )

    # Case 3: Automatic tool choice
    assert isinstance(tool_choice, str) and tool_choice == "auto"
    function_names = " | ".join(
        [f'''"<function={tool['function']['name']}>"''' for tool in tools]
    )
    initial_gbnf_tool_grammar = (
        """root   ::= functions | "<message>"\n"""
        f"""functions ::= {function_names}\n"""
    )
    follow_up_gbnf_tool_grammar = (
        """root   ::= functions | "<|eot_id|>"\n"""
        f"""functions ::= {function_names}\n"""
    )
    prompt = template_renderer.render(
        messages=messages,
        tools=tools,
        tool_calls=True,
        add_generation_prompt=True,
    )
    completion_or_chunks = llama.create_completion(
        prompt=prompt,
        temperature=0,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        stream=False,
        stop=stop,
        max_tokens=None,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
        grammar=llama_grammar.LlamaGrammar.from_string(
            initial_gbnf_tool_grammar, verbose=llama.verbose
        ),
    )
    completion: llama_types.CreateCompletionResponse = completion_or_chunks  # type: ignore
    text = completion["choices"][0]["text"]
    
    if not text.startswith("<function"):
        return _convert_completion_to_chat_v050(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=["<|eot_id|>"],
                logprobs=top_logprobs if logprobs else None,
                max_tokens=None,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                # grammar=llama_grammar.LlamaGrammar.from_string(
                #     follow_up_gbnf_tool_grammar, verbose=llama.verbose
                # ),
            ),
            stream=stream,
        )

    # One or more function calls
    tool_dict_unique = []
    tool_name = text[len("<function=") : -1]
    tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    if not stream:
        completions: List[llama_types.CreateCompletionResponse] = []
        completions_tool_name: List[str] = []
        while tool is not None:
            prompt += f"<function={tool_name}>"
            try:
                grammar = llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(tool["function"]["parameters"], ensure_ascii=False), verbose=llama.verbose
                )
            except Exception as e:
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF, verbose=llama.verbose
                )
                if llama.verbose:
                    print(
                        "Failed to parse function body as JSON schema, falling back to default grammar"
                    )
                    print(e)
            completion_or_chunks = llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=False,
                stop=stop,
                max_tokens=None,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
            )
            completion_or_chunks = cast(llama_types.CreateCompletionResponse, completion_or_chunks)
            
            tool_dict = {
                "name": tool_name,
                "arguments": json.dumps(json.loads(completion_or_chunks["choices"][0]["text"]), ensure_ascii=False)
            }
            if tool_dict in tool_dict_unique:
                break
            else:
                tool_dict_unique.append(tool_dict)
            
            completions.append(completion_or_chunks)
            completions_tool_name.append(tool_name)
            prompt += completion_or_chunks["choices"][0]["text"]
            prompt += "</function>"

            response = llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=False,
                stop=stop,
                max_tokens=None,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=llama_grammar.LlamaGrammar.from_string(
                    follow_up_gbnf_tool_grammar, verbose=llama.verbose
                ),
            )
            response = cast(llama_types.CreateCompletionResponse, response)

            tool_name = response["choices"][0]["text"][len("<function=") :-1]
            tool = next(
                (tool for tool in tools if tool["function"]["name"] == tool_name), None
            )

        # Merge completions
        function_call_dict: Union[Dict[str, str], Dict[Literal["function_call"], llama_types.ChatCompletionRequestAssistantMessageFunctionCall]] = { 
            "function_call": {
                "name": tool_name,
                "arguments": completions[0]["choices"][0]["text"],
            }
        } if len(completions) == 1 else {}
        return {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "logprobs": completion["choices"][0]["logprobs"],
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_"
                                + f"_{i}_"
                                + tool_name
                                + "_"
                                + completion["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": completion["choices"][0]["text"],
                                },
                            }
                            for i, (tool_name, completion) in enumerate(
                                zip(completions_tool_name, completions)
                            )
                        ],
                        **function_call_dict
                    },
                }
            ],
            "usage": {
                "completion_tokens": sum(
                    completion["usage"]["completion_tokens"] if "usage" in completion else 0
                    for completion in completions
                ),
                "prompt_tokens": sum(
                    completion["usage"]["prompt_tokens"] if "usage" in completion else 0
                    for completion in completions
                ),
                "total_tokens": sum(
                    completion["usage"]["total_tokens"] if "usage" in completion else 0
                    for completion in completions
                ),
            },
        }

    raise ValueError("Automatic streaming tool choice is not supported")






def _convert_text_completion_chunks_to_chat_v050(
    chunks: Iterator[llama_types.CreateCompletionStreamResponse],
) -> Iterator[llama_types.ChatCompletionChunk]:
    
    inside_message = False  # 현재 <message> 안에 있는지 추적하는 변수
    message_end = False
    buffer = []  # <message> 태그 안의 내용을 저장하는 버퍼
    str_buffer = ""
    keep_num_token = 3
    buffer_cnt = 0
    has_message_wrap = False
    
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            if chunk["choices"][0]["text"].startswith('<message'):
                has_message_wrap = True
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                        },
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            
        if has_message_wrap:
            if not inside_message:
                str_buffer += chunk["choices"][0]["text"]
                if str_buffer.startswith("<message>"):
                    cur_chunk = str_buffer[len("<message>"):]
                    yield {
                        "id": "chat" + chunk["id"],
                        "model": chunk["model"],
                        "created": chunk["created"],
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": (
                                    {
                                        cur_chunk,
                                    }
                                    if chunk["choices"][0]["finish_reason"] is None
                                    else {}
                                ),
                                "logprobs": chunk["choices"][0]["logprobs"],
                                "finish_reason": chunk["choices"][0]["finish_reason"],
                            }
                        ],
                    }
                    inside_message = True
                    str_buffer = ""
            elif not message_end:
                str_buffer += chunk["choices"][0]["text"]
                if "</message>" in str_buffer:
                    buffer.append(chunk)
                    last_chunk = "".join([ch["choices"][0]["text"] for ch in buffer])
                    cur_chunk = last_chunk.replace("</message>", "")
                    yield {
                        "id": "chat" + chunk["id"],
                        "model": chunk["model"],
                        "created": chunk["created"],
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": (
                                    {
                                        cur_chunk,
                                    }
                                    if chunk["choices"][0]["finish_reason"] is None
                                    else {}
                                ),
                                "logprobs": chunk["choices"][0]["logprobs"],
                                "finish_reason": chunk["choices"][0]["finish_reason"],
                            }
                        ],
                    }
                    message_end = True
                        
                if buffer_cnt <= keep_num_token:
                    buffer_cnt+=1
                    buffer.append(chunk)
                else:
                    cchunk = buffer.pop(0)
                    yield {
                        "id": "chat" + cchunk["id"],
                        "model": cchunk["model"],
                        "created": cchunk["created"],
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": (
                                    {
                                        cchunk["choices"][0]["text"],
                                    }
                                    if cchunk["choices"][0]["finish_reason"] is None
                                    else {}
                                ),
                                "logprobs": cchunk["choices"][0]["logprobs"],
                                "finish_reason": cchunk["choices"][0]["finish_reason"],
                            }
                        ],
                    }
                    buffer.append(chunk)
            else:
                yield {
                    "id": "chat" + chunk["id"],
                    "model": chunk["model"],
                    "created": chunk["created"],
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": (
                                {
                                    "content": chunk["choices"][0]["text"],
                                }
                                if chunk["choices"][0]["finish_reason"] is None
                                else {}
                            ),
                            "logprobs": chunk["choices"][0]["logprobs"],
                            "finish_reason": chunk["choices"][0]["finish_reason"],
                        }
                    ],
                }
                
        else:
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": (
                            {
                                "content": chunk["choices"][0]["text"],
                            }
                            if chunk["choices"][0]["finish_reason"] is None
                            else {}
                        ),
                        "logprobs": chunk["choices"][0]["logprobs"],
                        "finish_reason": chunk["choices"][0]["finish_reason"],
                    }
                ],
            }


def _convert_text_completion_to_chat_v050(
    completion: llama_types.Completion,
) -> llama_types.ChatCompletion:
    assert "usage" in completion
    
    text = completion["choices"][0]["text"].strip()
    if text.startswith("<message>"):
        text = text[len("<message>"):]
    if text.endswith("</message>"):
        text = text[:-len("</message>")]
    
    return {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "logprobs": completion["choices"][0]["logprobs"],
                "finish_reason": completion["choices"][0]["finish_reason"],
            }
        ],
        "usage": completion["usage"],
    }


def _convert_completion_to_chat_v050(
    completion_or_chunks: Union[
        llama_types.CreateCompletionResponse,
        Iterator[llama_types.CreateCompletionStreamResponse],
    ],
    stream: bool = False,
) -> Union[
    llama_types.CreateChatCompletionResponse, Iterator[llama_types.ChatCompletionChunk]
]:
    if stream:
        chunks: Iterator[llama_types.CreateCompletionStreamResponse] = completion_or_chunks  # type: ignore
        return _convert_text_completion_chunks_to_chat_v050(chunks)
    else:
        completion: llama_types.Completion = completion_or_chunks  # type: ignore
        return _convert_text_completion_to_chat_v050(completion)






@register_chat_completion_handler("llama-3.1-keti-v0.50-fc-fast")
def keti_function_calling_v050_fast(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    **kwargs,  # type: ignore
) -> Union[
    llama_types.CreateChatCompletionResponse,
    Iterator[llama_types.CreateChatCompletionStreamResponse],
]:
    template_renderer = ImmutableSandboxedEnvironment(
        undefined=jinja2.StrictUndefined,
    ).from_string(function_calling_template_v050)
    
    messages = formatting_tool_message_v050(messages)

    # Convert legacy functions to tools
    if functions is not None:
        tools = [
            {
                "type": "function",
                "function": function,
            }
            for function in functions
        ]

    # Convert legacy function_call to tool_choice
    if function_call is not None:
        if isinstance(function_call, str) and (
            function_call == "none" or function_call == "auto"
        ):
            tool_choice = function_call
        if isinstance(function_call, dict) and "name" in function_call:
            tool_choice = {
                "type": "function",
                "function": {
                    "name": function_call["name"],
                },
            }

    stop = [stop, "<|eot_id|>", "<|eom_id|>"] if isinstance(stop, str) else stop + ["<|eot_id|>", "<|eom_id|>"] if stop else ["<|eot_id|>", "<|eom_id|>"]

    # Case 1: No tool choice by user
    if (
        tool_choice is None
        or (isinstance(tool_choice, str) and tool_choice == "none")
        or tools is None
        or len(tools) == 0
    ):
        prompt = template_renderer.render(
            messages=messages,
            tools=[],
            tool_calls=None,
            add_generation_prompt=True,
        )

        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(response_format)

        return _convert_completion_to_chat_v050(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
                logprobs=top_logprobs if logprobs else None,
            ),
            stream=stream,
        )

    # Case 2: Tool choice by user
    if isinstance(tool_choice, dict):
        tool_name = tool_choice["function"]["name"]
        tool = next(
            (tool for tool in tools if tool["function"]["name"] == tool_name), None
        )
        if tool is None:
            raise ValueError(f"Tool with name '{tool_name}' not found in tools")
        prompt = template_renderer.render(
            messages=messages,
            tools=tools,
            tool_calls=True,
            add_generation_prompt=True,
        )
        prompt += f"<function={tool_name}>"
        try:
            grammar = llama_grammar.LlamaGrammar.from_json_schema(
                json.dumps(tool["function"]["parameters"]), verbose=llama.verbose
            )
        except Exception as e:
            grammar = llama_grammar.LlamaGrammar.from_string(
                llama_grammar.JSON_GBNF, verbose=llama.verbose
            )
            if llama.verbose:
                print(
                    "Failed to parse function body as JSON schema, falling back to default grammar"
                )
                print(e)
        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
        )
        return _convert_completion_to_chat_function(
            tool_name, completion_or_chunks, stream
        )

    # Case 3: Automatic tool choice
    assert isinstance(tool_choice, str) and tool_choice == "auto"
    function_names = " | ".join(
        [f'''"<function={tool['function']['name']}>"''' for tool in tools]
    )
    initial_gbnf_tool_grammar = (
        """root   ::= functions | "<message>"\n"""
        f"""functions ::= {function_names}\n"""
    )
    follow_up_gbnf_tool_grammar = (
        """root   ::= functions | "<|eot_id|>"\n"""
        f"""functions ::= {function_names}\n"""
    )
    prompt = template_renderer.render(
        messages=messages,
        tools=tools,
        tool_calls=True,
        add_generation_prompt=True,
    )
    completion_or_chunks = llama.create_completion(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        stream=False,
        stop=stop,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
        grammar=grammar,
        logprobs=top_logprobs if logprobs else None,
    )
    
    completion: llama_types.CreateCompletionResponse = completion_or_chunks  # type: ignore
    return _convert_text_completion_to_chat_v050_fast(completion)


def extract_message_content(text):
    pattern = r"<message>(.*?)<\/message>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def extract_function_details(text):
    pattern = r"<function=(.*?)>(.*?)<\/function>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"function_name": match[0], "function_arguments": match[1]} for match in matches]


def extract_and_remove_message_content(text):
    # 정규 표현식으로 메시지 내용 추출
    pattern = r"<message>(.*?)<\/message>"
    messages = re.findall(pattern, text, re.DOTALL)
    
    # 원본 텍스트에서 메시지 태그 제거
    text_without_messages = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return messages, text_without_messages


def parsing_message_and_functions(text):
    messages, text_without_messages = extract_and_remove_message_content(text)
    functions = extract_function_details(text_without_messages)

    return messages, functions


def _convert_text_completion_to_chat_v050_fast(
    completion: llama_types.Completion,
) -> llama_types.ChatCompletion:
    assert "usage" in completion
    
    text = completion["choices"][0]["text"]
    messages, functions = parsing_message_and_functions(text)

    chat_completion = {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": " ".join(messages) if messages else None,
                },
                "logprobs": completion["choices"][0]["logprobs"],
                "finish_reason": completion["choices"][0]["finish_reason"],
            }
        ],
        "usage": completion["usage"],
    }
    
    if functions:
        choice = chat_completion["choices"][0]
        choice["finish_reason"] = "tool_calls"
        choice["message"]["tool_calls"] = [
            {
                "id": "call_"
                + f"_{i}_"
                + tool_info["function_name"]
                + "_"
                + str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": tool_info["function_name"],
                    "arguments": tool_info["function_arguments"],
                },
            }
            for i, tool_info in enumerate(functions)
        ]
    return chat_completion

