"""
Author: Jared Moore
Date: August, 2024

Contains utility functions for querying models.
"""

import string
from typing import Iterable

import tiktoken

OPTIONS = string.ascii_uppercase

SUMMARIZE = (
    "Summarize your answer to the question below by writing only the option letter."
)


def options_text(
    question: str, options: Iterable[str], single_letter_prompt: bool = False
):
    """
    Paramters
    - question, str: The question to ask
    - options, Iterable[str]: The possible answer to the question
    - single_letter_prompt, bool: Whether to add a phrase to ask that only the option
        letter be included in the response to the question.
    """
    # NB: Putting the response of 'A' or 'B' closer to the end to deal with the
    # 'cognitive load' point raised by Jennifer Hu from Harvard.
    prompt = ""
    if single_letter_prompt:
        prompt = SUMMARIZE + "\n\n"

    prompt += question.strip() + "\n"

    for letter, option in zip(OPTIONS[0 : len(options)], options):
        prompt += f"- ({letter.upper()}): {option}\n"

    return prompt.strip()


COT_DELIMITER = "---"


def split_thought_from_response(
    response_text: str, delimiter: str = COT_DELIMITER
) -> (str, str):
    """
    Splits the `response_text` assuming it starts with a CoT response and
    ends with the actual response.
    """
    thought = None
    response = response_text
    if delimiter in response_text:
        thought, response = response_text.split(delimiter, maxsplit=1)
        thought = thought.strip()
    return (thought, response.strip())


def convert_roles(list_of_dicts, role_mapping):
    """
    Convert the roles in a list of dictionaries based on a provided mapping.

    Parameters:
    list_of_dicts (list): A list of dictionaries, each containing a 'role' key.
    role_mapping (dict): A dictionary mapping original roles to new roles.

    Returns:
    list: The modified list of dictionaries with updated roles.
    """
    # Apply the role mapping
    for item in list_of_dicts:
        if item["role"] in role_mapping:
            item["role"] = role_mapping[item["role"]]

    return list_of_dicts


def messages_as_string(
    messages,
    open_ended_last_message=False,
    assistant_name="Assistant",
    user_name="User",
    system_name="System",
):
    """
    Converts a list of message dictionaries in OpenAI-style format into a single string.

    Parameters:
    - messages (list of dict): A list of dictionaries where each dictionary represents a message.
      Each dictionary should have the keys 'role' (indicating the sender, e.g., 'user',
      'assistant', or 'system') and 'content' (the message text).
    - open_ended_last_message (bool): If True and the last message is not from the assistant,
      appends the assistant's name to the end of the result string. Default is False.
    - assistant_name (str): The name to use for the assistant's messages. Default is 'Assistant'.
    - user_name (str): The name to use for the user's messages. Default is 'User'.

    Returns:
    - str: A single string representing the conversation, with each message prefixed by the
        sender's name.
    """
    name_prefix = "\n\n"
    name_suffix = ":"
    result = ""
    last_role = None
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role and content:
            name = name_prefix
            if role == "assistant":
                name += assistant_name
            elif role == "user":
                name += user_name
            elif role == "system":
                name += system_name
            else:
                raise ValueError(f"Unknown role, {role}")
            last_role = role
            name += name_suffix

            result += f"{name} {content}"
    if open_ended_last_message and last_role != "assistant":
        result = result.strip() + name_prefix + assistant_name + name_suffix
    return result


########### Token counting

# Will need to update these from here: https://openai.com/pricing
OPENAI_PRICES = {
    "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-0613": {"input": 0.03, "output": 0.06},
    "gpt-4-32k-0613": {"input": 0.06, "output": 0.12},
    "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
    "gpt-3.5-turbo-0613": {"input": 0.001, "output": 0.002},
    "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
    "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.002},
    "davinci-002": {"input": 0.002, "output": 0.002},
    "babbage-002": {"input": 0.0004, "output": 0.0004},
}


# The following two functions are from here:
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(input_string: str, model: str) -> int:
    """Returns the number of tokens in a text string.

    Args:
        input_string (str): The input text string.
        model (str): The model name to use for encoding.

    Returns:
        int: The number of tokens in the input string.
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(input_string))
    return num_tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages.

    Args:
        messages (list): A list of message dictionaries.
        model (str): The model name to use for encoding. Defaults to "gpt-3.5-turbo-0613".

    Returns:
        int: The total number of tokens used by the list of messages.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. "
            + "Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. \
            See https://github.com/openai/openai-python/blob/main/chatml.md for information \
            on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def tokens_to_cost(model, input_tokens, output_tokens):
    """Calculate the cost of tokens based on the model pricing.

    Args:
        model (str): The model name.
        input_tokens (int): The number of input tokens.
        output_tokens (int): The number of output tokens.

    Returns:
        float: The total cost of the tokens.
    """
    in_cost = OPENAI_PRICES[model]["input"]
    out_cost = OPENAI_PRICES[model]["output"]
    return input_tokens / 1000 * in_cost + output_tokens / 1000 * out_cost


def report_tokens(total_input, total_output, model):
    """Print a report of the total input tokens, output tokens, and cost.

    Args:
        total_input (int): The total number of input tokens.
        total_output (int): The total number of output tokens.
        model (str): The model name.
    """
    cost = tokens_to_cost(model, total_input, total_output)
    print(f"Total input tokens: {total_input}")
    print(f"Total output tokens: {total_output}")
    print(f"Total cost: ${cost}")


###########
