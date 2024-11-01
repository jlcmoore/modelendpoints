
# ModelEndpoints

A simple, unified structure to query a variety of LLM API endpoints as well as to serve `vllm` models.

## Set-up

### For macOS/Linux - Install:

- `Make` (e.g. `brew install make`)

- `python>=3.10` (e.g. `brew install python@3.10`)

- `make init`

- `source env-modelendpoints/bin/activate`

### On a linux-based GPU machine

- `make init-conda`

(E.g. for testing `vllm` on a machine with 8, 25Gb GPUs:

```
vllm serve \
meta-llama/Llama-2-70b-chat-hf  \
--dtype auto \
--trust-remote-code \
--tensor-parallel-size 8 \
--gpu-memory-utilization .8
```
)

### Environment variables

Store the following as environment variables or pass them in as arguments.

- [HF_HOME](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables#hfhome)
- [HF_TOKEN](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables#hftoken)
- [OPENAI_API_KEY](https://platform.openai.com/api-keys)
- [TOGETHER_API_KEY](https://docs.together.ai/docs/quickstart)
- [ANTHROPIC_API_KEY](https://docs.anthropic.com/en/api/getting-started)

E.g.:
 - macOS/Linux: `echo export HF_HOME="<path>"' >> ~/.zshrc`

Or pass as arguments: `OPENAI_API_KEY=<key> make test`

## Tests

For basic tests, run:

`make test`


## Contributing

Aim to [`black`]( https://black.readthedocs.io) your code (e.g. `black src`).

Also use `pylint` (e.g. `pylint src` or just `darker --lint pylint src` which only applies pylint to the changed files, although it takes a while to run).

For large changes submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)

## Repository Structure

- `Makefile`
    - Defines various project level shell script utilities.
- `README.md`
- *`env-mindgames`*
    - Built by `make init`. Your local python virtual environment.
- `environment.yml`
    - Package installs and machine configuration for use with `conda`.
- `requirements.txt`
    - Package installs for use with `pip`.
- `setup.py`
- `modelendpoints`
    - `query.py`
        - The bulk of the code to set up Endpoints and query models.
    - `utils.py`
        - Utilities, such as turning a list of messages into a string, CoT prompting, etc.
- `tests`
    - `modelendpoints`
        - `test_query.py`   
            - Tests various endpoints.
- `pyproject.toml`
    - Defines variables for the project, namely the max line length for linters.

## License

This code is [licensed under an MIT license](LICENSE)
