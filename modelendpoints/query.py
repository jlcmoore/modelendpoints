"""
Author: Jared Moore
Date: August, 2024

Contains various functions to spin up and otherwise query LLMs.
"""

import asyncio
import copy
import io
import functools
import json
import logging
import os
import pprint
import signal
import subprocess
import tempfile
import time
import types
from typing import Iterable, Type, Any, Callable
import warnings

import anthropic
import huggingface_hub
import openai
import requests
from requests.adapters import HTTPAdapter, Retry
import tenacity
import together
import tqdm

from .utils import OPTIONS

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

### VLLM

VLLM_ORIGIN = "http://localhost:8000/v1/"
VLLM_OPENAI_SERVER = ("vllm.entrypoints.openai.api_server",)

Messages = list[dict[str, str]]


def wait_for_load(origin: str):
    """
    Waits for the server to load by making repeated requests.

    Parameters:
    origin (str): The origin URL of the server.
    """
    s = requests.Session()

    retries = Retry(total=30, backoff_factor=10, status_forcelist=[500, 502, 503, 504])

    s.mount("http://", HTTPAdapter(max_retries=retries))

    response = s.get(f"{origin}models")
    logger.debug(pprint.pformat(response.json()))
    logger.info("Server loaded")


def start_vllm_process(
    entrypoint: tuple[str,] = VLLM_OPENAI_SERVER,
    additional_vllm_args: list[str] | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    **kwargs,
):
    """
    Starts the VLLM process.

    Parameters:
    entrypoint (tuple[str,]): The entrypoint for the VLLM process.
    additional_vllm_args (list): Additional arguments for the VLLM process.
    kwargs (dict): Additional keyword arguments.

    Returns:
    subprocess.Popen: The started process.
    """
    import torch  # Importing here to avoid memory costs of loading it in general

    gpus = torch.cuda.device_count()
    download_dir = os.path.join(os.environ.get("HF_HOME"), "hub")
    huggingface_hub.login(token=os.getenv("HF_TOKEN").strip())
    vllm_args = (
        [
            "python",
            "-m",
        ]
        + list(entrypoint)
        + [
            "--model",
            kwargs["model"],
            "--download-dir",
            download_dir,
            "--trust-remote-code",
            "--tensor-parallel-size",
            str(gpus),
        ]
    )
    if additional_vllm_args:
        vllm_args += additional_vllm_args
    if max_model_len:
        vllm_args += ["--max-model-len", str(max_model_len)]
    if gpu_memory_utilization:
        vllm_args += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    process = subprocess.Popen(vllm_args, shell=False)
    # If debugging, you can redirect output of vllm:
    #  stdout=subprocess.DEVNULL)
    logger.info(f"Vllm process {process}")
    if entrypoint == VLLM_OPENAI_SERVER:
        wait_for_load(VLLM_ORIGIN)
    return process


def close_vllm_process(process: subprocess.Popen):
    """
    Closes the VLLM process.

    Parameters:
    process (subprocess.Popen): The process to close.
    """
    if process:
        logger.info(f"Terminating process {process}")
        process.terminate()
        process.wait(timeout=60)
        process.kill()


### API Method Wrappers


def process_openai_resposne(response: dict[str, Any]) -> dict[str, Any]:
    """
    Puts the response of the Openai api into a standard format
    """
    # TODO: no logprob handling
    return {
        "text": response.choices[0].message.content,
        "got_stop_seq": response.choices[0].finish_reason == "stop",
    }


def openai_chat(client: openai.OpenAI, messages: Messages, **kwargs) -> dict[str, Any]:
    """
    Sends a chat message to OpenAI.

    Parameters:
    client (openai.OpenAI): The OpenAI client.
    messages (list): The list of messages to send.
    kwargs (dict): Additional keyword arguments.

    Returns:
    dict: The response from OpenAI.
    """
    # TODO: count tokens like this
    # if count_tokens:
    # in_tokens += num_tokens_from_messages(messages)
    messages, kwargs = preprocess_openai_call(messages, **kwargs)
    response = client.chat.completions.create(messages=messages, **kwargs)
    return process_openai_resposne(response)


def preprocess_openai_call(messages: Messages, **kwargs) -> (Messages, dict[str, Any]):
    """
    Formats the arguments that would be used for an OpenAI style call for the Openai api.
    Returns the relevant
        messages
        kwargs
    """
    if "o1-preview" in kwargs["model"]:
        # Can't accept system roles
        for msg in messages:
            if msg["role"] == "system":
                msg["role"] = "user"
    return (messages, kwargs)


def preprocess_anthropic_call(
    messages: Messages, **kwargs
) -> (Messages, dict[str, Any]):
    """
    Formats the arguments that would be used for an OpenAI style call for the Anthropic api.
    Returns the relevant
        messages
        kwargs
    """
    # Anthropic does not allow system messages in the messages
    system = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "system":
            if system:
                warnings.warn("Multiple system messages given to Anthropic model")
            system = messages[i]["content"]
            del messages[i]
    if not hasattr(kwargs, "system") and system is not None:
        kwargs["system"] = system
    if "stop" in kwargs:
        kwargs["stop_sequences"] = kwargs["stop"]
        del kwargs["stop"]
    if "logprobs" in kwargs:
        del kwargs["logprobs"]
    if "top_logprobs" in kwargs:
        del kwargs["top_logprobs"]
    return (messages, kwargs)


def process_anthropic_response(response: dict[str, Any]) -> dict[str, Any]:
    """
    Puts the response of the Anthropic api into a standard format
    """
    return {
        "text": response.content[0].text,
        "got_stop_seq": (response.stop_reason in set(["end_turn", "stop_sequence"])),
        "logprobs": None,
    }


def anthropic_chat(
    client: anthropic.Anthropic, messages: Messages, **kwargs
) -> dict[str, Any]:
    """
    Sends a chat message to Anthropic.

    Parameters:
    client (anthropic.Anthropic): The Anthropic client.
    messages (list): The list of messages to send.
    kwargs (dict): Additional keyword arguments.

    Returns:
    dict: The response from Anthropic.
    """
    messages, kwargs = preprocess_anthropic_call(messages, **kwargs)
    response = client.messages.create(messages=messages, **kwargs)
    return process_anthropic_response(response)


### Batching

OPENAI_CHAT_URL = "/v1/chat/completions"


def to_batch_json(keys_to_messages: dict[str, Messages], **kwargs) -> bytes:
    """
    Converts keys and messages to a batch JSON.

    Parameters:
    keys_to_messages (dict): A dictionary mapping keys to messages.
    kwargs (dict): Additional keyword arguments.

    Returns:
    bytes: The batch JSON as bytes.
    """
    queries = []
    for key, messages in keys_to_messages.items():
        request_body = copy.deepcopy(kwargs)
        request_body["messages"] = messages
        queries.append(
            {
                "custom_id": key,
                "method": "POST",
                "url": OPENAI_CHAT_URL,
                "body": request_body,
            }
        )

    jsonl_string = "\n".join(json.dumps(request) for request in queries)
    bytes_data = jsonl_string.encode("utf-8")
    return bytes_data


def process_batch_results(content: str) -> dict[str, dict[str, str]]:
    """
    Processes batch results.

    Parameters:
    content (str): The content to process, as json objects delimited by new lines

    Returns:
    dict: The processed results.
    """
    all_results = {}
    for response_str in content.splitlines():
        response = json.loads(response_str)
        error = response.get("error")
        if error:
            logging.error(error)
        else:
            body = response["response"]
            choices = body.get("choices")
            if not choices:  # NB: openai wants 'body' here
                choices = body.get("body").get("choices")
            choice = choices[0]
            result = {
                "text": choice["message"]["content"],
                "got_stop_seq": choice["finish_reason"] == "stop",
            }
            all_results[response["custom_id"]] = result
    return all_results


def vllm_batch(
    keys_to_messages: dict[str, Messages], **kwargs
) -> dict[str, dict[str, str]]:
    """
    Processes a batch of messages using VLLM.

    Parameters:
    keys_to_messages (dict): A dictionary mapping keys to messages.
    kwargs (dict): Additional keyword arguments.

    Returns:
    dict: The processed results.
    """
    all_results = None
    with tempfile.NamedTemporaryFile("wb") as in_file:
        with tempfile.NamedTemporaryFile("rb") as out_file:
            # Write the data to the temporary file
            jsonl_bytes = to_batch_json(keys_to_messages, **kwargs)
            in_file.write(jsonl_bytes)

            args = ["-i", in_file.name, "-o", out_file.name]
            process = None
            try:
                process = start_vllm_process(
                    entrypoint=("vllm.entrypoints.openai.run_batch",),
                    additional_vllm_args=args,
                    **kwargs,
                )
                process.wait()
            finally:
                close_vllm_process(process)

            content = out_file.read().decode("utf-8")
            all_results = process_batch_results(content)
    return all_results


def openai_batch(
    client: openai.OpenAI, keys_to_messages: dict[str, Messages], **kwargs
) -> dict[str, dict[str, str]]:
    """
    Processes a batch of messages using OpenAI.

    Parameters:
    client (openai.OpenAI): The OpenAI client.
    keys_to_messages (dict): A dictionary mapping keys to messages.
    kwargs (dict): Additional keyword arguments.

    Returns:
    dict: The processed results.
    """
    #### Making file
    bytes_data = to_batch_json(keys_to_messages, **kwargs)

    #### Uploading file
    file_response = client.files.create(file=io.BytesIO(bytes_data), purpose="batch")
    if not file_response or not hasattr(file_response, "id"):
        raise ValueError("Invalid file upload response")

    try:
        #### Making batch
        batch_response = client.batches.create(
            input_file_id=file_response.id,
            endpoint=OPENAI_CHAT_URL,
            completion_window="24h",
        )
        if not batch_response or not hasattr(batch_response, "id"):
            raise ValueError("Invalid batch response")

        #### Retrieving batch
        retrieve_response = None
        wait_time_seconds = 25 * 60 * 60  # 25 hours
        sleep_time_per_iteration = 10

        iterations = wait_time_seconds // sleep_time_per_iteration

        logging.info("Waiting for 25 hours...")

        # Use tqdm to show progress
        for _ in tqdm.tqdm(range(iterations), desc="Waiting for batch"):
            retrieve_response = client.batches.retrieve(batch_response.id)
            if not retrieve_response or not hasattr(
                retrieve_response, "output_file_id"
            ):
                raise ValueError("Invalid retrieve response")
            if retrieve_response.failed_at:
                logging.error(retrieve_response.errors)
                raise ValueError("Batch failed")
            if retrieve_response.status == "completed":
                break

            time.sleep(sleep_time_per_iteration)

        #### Formatting response
        binary_content = client.files.content(retrieve_response.output_file_id).content
        content = binary_content.decode("utf-8")
        all_results = process_batch_results(content)

        # if len(set(result.keys()) - set(keys_to_messages.keys())) > 0:
        # raise ValueError("Result size does not match input")

        return all_results
    finally:
        client.files.delete(file_response.id)


def batch_prompts_serial(
    function: types.FunctionType,
    preprocessor: types.FunctionType | None,
    postprocessor: types.FunctionType,
    keys_to_messages: dict[str, Messages],
    **kwargs,
) -> dict[str, dict[str, str]]:
    """
    Processes a batch of messagess, handling interrupts gracefully.
    Returns partial results if interrupted.

    Parameters:
        function: The function to call.
        preprocessor: a function to call to preprocess the arguments to `function`
        preprocessor: a function to call to postprocess the return from `function`
        keys_to_messages (dict): A dictionary mapping keys to messages.
        kwargs (dict): Additional keyword arguments to `function`.

    Returns:
    dict: The processed results.
    """
    results = {}
    remaining_keys = list(keys_to_messages.keys())
    pbar = tqdm.tqdm(total=len(keys_to_messages), desc="Processing batch")

    try:
        for key, messages in keys_to_messages.items():
            this_call_kwargs = copy.deepcopy(kwargs)
            if preprocessor:
                messages, this_call_kwargs = preprocessor(
                    messages=messages, **this_call_kwargs
                )
            if not messages:
                raise ValueError("Messages must not be empty.")
            assert "model" in this_call_kwargs

            try:
                response = function(messages=messages, **this_call_kwargs)
                processed_response = postprocessor(response)
                results[key] = processed_response
                remaining_keys.remove(key)
                pbar.update(1)

            except (
                openai.APIError,
                anthropic.APIError,
                asyncio.TimeoutError,
                ValueError,
                ConnectionError,
            ) as e:
                logger.error(f"Error for key {key}: {str(e)}")

    except KeyboardInterrupt:
        logger.warning(
            f"Batch processing interrupted. Completed {len(results)}/{len(keys_to_messages)} calls"
        )
        if remaining_keys:
            logger.warning(f"Remaining keys: {remaining_keys}")
    finally:
        pbar.close()
    return results


async def batch_prompts_async(
    async_function: types.FunctionType,
    preprocessor: types.FunctionType | None,
    postprocessor: types.FunctionType,
    keys_to_messages: dict[str, Messages],
    **kwargs,
) -> dict[str, dict[str, str]]:
    """
    Processes a batch of messages asynchronously, handling interrupts gracefully.
    Returns partial results if interrupted.

    Parameters:
        async_function: The async function to call.
        preprocessor: a function to call to preprocess the arguments to `async_function`
        preprocessor: a function to call to postprocess the return from `async_function`
        keys_to_messages (dict): A dictionary mapping keys to messages.
        kwargs (dict): Additional keyword arguments to `async_function`.

    Returns:
    dict: The processed results.
    """
    # Create an event for signaling shutdown
    shutdown_event = asyncio.Event()

    # Set up signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        shutdown_event.set()

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Track tasks with their corresponding keys
        tasks = {}
        results = {}

        # Create progress bar
        pbar = tqdm.tqdm(total=len(keys_to_messages), desc="Processing batch")

        # Create tasks for each message
        for key, messages in keys_to_messages.items():
            this_call_kwargs = copy.deepcopy(kwargs)
            if preprocessor:
                messages, this_call_kwargs = preprocessor(
                    messages=messages, **this_call_kwargs
                )
            if not messages:
                raise ValueError("Messages must not be empty.")
            assert "model" in this_call_kwargs

            task = asyncio.create_task(
                async_function(messages=messages, **this_call_kwargs)
            )
            tasks[key] = task

        # Process tasks as they complete
        while tasks and not shutdown_event.is_set():
            done, pending = await asyncio.wait(
                tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=1.0,  # Check shutdown event periodically
            )

            if not done and shutdown_event.is_set():
                # Shutdown requested, cancel pending tasks
                for task in pending:
                    task.cancel()
                # Wait for cancellation to complete
                if pending:
                    await asyncio.wait(pending)
                break

            # Process completed tasks
            for task in done:
                # Find key for completed task
                key = next(k for k, t in tasks.items() if t == task)
                try:
                    response = await task
                    results[key] = postprocessor(response)
                    pbar.update(1)
                except (
                    openai.APIError,
                    anthropic.APIError,
                    asyncio.TimeoutError,
                    ValueError,
                    ConnectionError,
                ) as e:
                    logger.error(f"Error for key {key}: {str(e)}")
                except asyncio.CancelledError:
                    logger.warning(f"Task for key {key} was cancelled")

                # Remove completed task
                del tasks[key]

        if tasks:
            remaining_keys = list(tasks.keys())
            logger.warning(
                f"Batch processing interrupted. Completed {len(results)}/{len(keys_to_messages)} calls"
            )
            if remaining_keys:
                logger.warning(f"Remaining keys: {remaining_keys}")

    finally:
        # Clean up signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        pbar.close()

    return results


### General


def get_model_source(model_name: str) -> str:
    """Returns the source of the given model name"""
    # Wildcard dictionary with prefixes and corresponding outputs
    wildcard_dict = {
        "gpt": "openai",
        "claude": "anthropic",
        "meta": "vllm",
        # Add more prefixes and their outputs as needed
    }

    # Check for matching prefix
    for prefix, output in wildcard_dict.items():
        model_name = model_name.lower()
        if model_name.startswith(prefix):
            # This is a hack to disambiguate between 'together' and 'vllm'
            if output == "vllm" and "lite" in model_name or "turbo" in model_name:
                return "together"
            return output
    raise ValueError(f"{model_name} unknown")


def get_option(text: str) -> str:
    """
    Strips the given `text` so that it just contains the option letter token.
    e.g. get_option('(A)') -> 'A'
    """
    token = text.strip()
    # This is the 'lower one eigth block'
    # a character used to indicate spaces.
    token = (
        token.replace("▁", "")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace(".", "")
    )
    token = token.split(" ")[0]
    # Returning more than the first character here
    # because tokenization should treat 'A' apart from 'Aardvark', e.g.
    return token


def find_answer(option: str, answers: Iterable[str]) -> str:
    """Given the chosen option, `option`, returns the corresponding
    answer in `answers` assuming they are indexed the same.
    """
    answer = None
    option = option.upper()
    if option in list(OPTIONS[: len(answers)].upper()):
        answer = answers[OPTIONS.index(option)]
    else:
        logger.warning("No option included")
    return answer


def retrying_async_func_wrapper(function: Callable) -> Callable:
    """Wraps the passed async function in a retry and returns the new function."""

    @tenacity.retry(
        wait=tenacity.wait_chain(
            *[tenacity.wait_fixed(3) for i in range(3)]
            + [tenacity.wait_fixed(5) for i in range(2)]
            + [tenacity.wait_fixed(10)]
        )
    )
    async def retrying_async_func(*args, **kwargs):
        return await function(*args, **kwargs)

    return retrying_async_func


def retrying_func_wrapper(function: Callable) -> Callable:
    """Wraps the passed function in a retry and returns the new function."""

    @tenacity.retry(
        wait=tenacity.wait_chain(
            *[tenacity.wait_fixed(3) for i in range(3)]
            + [tenacity.wait_fixed(5) for i in range(2)]
            + [tenacity.wait_fixed(10)]
        )
    )
    def retrying_func(*args, **kwargs):
        return function(*args, **kwargs)

    return retrying_func


class Endpoint:
    """
    A class to house either an OpenAI client or a VLLM process.
    """

    def __init__(
        self,
        source: str | None = None,
        async_function: bool = False,
        batch_prompts: bool = False,
        batch_function: bool = False,
        base_url: str | None = None,
        serial_batch_prompts: bool = False,
        retrying: bool = False,
        **kwargs,
    ):
        """
        Initializes the Endpoint.

        Parameters:
        source (str): The source of the endpoint.
            If None attempts to predict the source using `get_model_source(kwargs['model'])`
        async_function (bool): Whether to return an asynchronous callable function.
        batch_prompts (bool): Whether to batch prompts, returning on __enter__ a function which
            requires keys_to_messages not messages in its signature
        batch_function (bool): Just for vllm and openai, whether to use their specific batch
            functions (as opposed to simply wrapping a host of async calls in a loop)
        base_url (str): The (vllm) endpoint to direct the (OpenAI) client to.
        serial_batch_prompts (bool): Whether to internally run the `batch_prompt` function as
            serial. Default False. Also sets `async_function` to False.
        retrying (bool): Whether to wrap the returned function in a retry. Obfuscates errors.
        kwargs (dict): Additional keyword arguments.
        """
        self.source = (
            source if source is not None else get_model_source(kwargs["model"])
        )
        self.batch_prompts = batch_prompts
        self.batch_function = batch_function
        self.kwargs = kwargs
        self.process = None
        self.client = None
        self.base_url = base_url
        self.async_function = async_function and not serial_batch_prompts
        # TODO: need to add test cases for the below two arguments
        self.serial_batch_prompts = serial_batch_prompts
        self.retrying = retrying

        if self.batch_function and not self.batch_prompts:
            raise ValueError(
                "You must batch the prompts to use a special batch function."
            )

        if self.base_url and self.source != "openai":
            raise NotImplementedError(
                "`base_url` can be set only when source is openai"
            )

        if self.source not in {"openai", "together", "anthropic", "vllm"}:
            raise NotImplementedError(f"Unknown source, {self.source}")

        if self.async_function and not self.batch_prompts:
            raise NotImplementedError("Async functions only implemented when batching.")

        if self.batch_function and self.source not in {"openai", "vllm"}:
            raise ValueError(
                "Anthropic and Together do not provide specific batch functions."
            )

        if self.batch_prompts and not async_function:
            try:
                asyncio.get_running_loop()
                raise ValueError(
                    "You must request an async function in a running loop."
                )
            except RuntimeError:
                # No current event loop is running
                pass

    def __enter__(self):
        """
        Enters the context of the Endpoint.

        Returns:
        function: The appropriate function for the endpoint.
        """
        use_async = (
            self.async_function or self.batch_prompts and not self.batch_function
        ) and not self.serial_batch_prompts

        if self.source == "vllm":
            if not self.batch_function:
                self.process = start_vllm_process(**self.kwargs)
                api_key = "EMPTY"
                self.client = openai.OpenAI(api_key=api_key, base_url=VLLM_ORIGIN)
        else:

            endpoint_metadata = {
                "openai": {
                    "class": openai.OpenAI,
                    "async_class": openai.AsyncOpenAI,
                    "env_variable": "OPENAI_API_KEY",
                },
                "together": {
                    "class": together.Together,
                    "async_class": together.AsyncTogether,
                    "env_variable": "TOGETHER_API_KEY",
                },
                "anthropic": {
                    "class": anthropic.Anthropic,
                    "async_class": anthropic.AsyncAnthropic,
                    "env_variable": "ANTHROPIC_API_KEY",
                },
            }

            class_kind = "async_class" if use_async else "class"

            client_class = endpoint_metadata[self.source][class_kind]
            api_key = os.getenv(endpoint_metadata[self.source]["env_variable"]).strip()
            self.client = client_class(api_key=api_key, base_url=self.base_url)

        if self.source == "anthropic":
            llm_message_create = self.client.messages.create
            preprocessor = preprocess_anthropic_call
            postprocessor = process_anthropic_response
            chat_function = anthropic_chat
        else:
            assert self.source in {"openai", "vllm", "together"}
            if self.source != "vllm" or not self.batch_function:
                llm_message_create = self.client.chat.completions.create
            preprocessor = preprocess_openai_call
            postprocessor = process_openai_resposne
            chat_function = openai_chat

        if self.retrying and (self.source != "vllm" or not self.batch_function):
            chat_function = retrying_func_wrapper(chat_function)
            if use_async:
                llm_message_create = retrying_async_func_wrapper(llm_message_create)
            else:
                llm_message_create = retrying_func_wrapper(llm_message_create)

        if self.batch_prompts:
            # Here just serialize the calls if relevant
            if self.serial_batch_prompts:
                return functools.partial(
                    batch_prompts_serial,
                    function=llm_message_create,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    **self.kwargs,
                )

            # Here there may be genuine batching api functions
            if self.batch_function:
                if self.source == "openai":
                    # NB: we could do the same as the Together call below
                    # but this saves more money (although it takes longer)
                    batch_func = functools.partial(
                        openai_batch, client=self.client, **self.kwargs
                    )
                else:
                    assert self.source == "vllm"
                    batch_func = functools.partial(vllm_batch, **self.kwargs)

                ## For these two we need to make the sync functions async
                if self.async_function:

                    def wrapper(*args, **kwargs):
                        loop = asyncio.get_event_loop()
                        # None uses the default executor (ThreadPoolExecutor)
                        return loop.run_in_executor(
                            None, functools.partial(batch_func, *args, **kwargs)
                        )

                    return wrapper
                return batch_func

            # No real batch function here, but make all of the calls asynchronously.
            async_batch_func = functools.partial(
                batch_prompts_async,
                async_function=llm_message_create,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                **self.kwargs,
            )

            if self.async_function:
                return async_batch_func

            ## For these we need to make the async functions sync
            # A bit of a hack as this cannot be async nested.
            # Wrap the async call in its own run.
            def batched_func(*args, **kwargs):
                return asyncio.get_event_loop().run_until_complete(
                    async_batch_func(*args, **kwargs)
                )

            return batched_func

        # Not batching. All but anthropic use the openai protocol
        return functools.partial(chat_function, client=self.client, **self.kwargs)

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ):
        """
        Exits the context of the Endpoint and kills any process.
        """
        if (
            self.client
            and hasattr(self.client, "close")
            and callable(getattr(self.client, "close"))
            and not asyncio.iscoroutinefunction(self.client.close)
        ):

            self.client.close()

        close_vllm_process(self.process)
