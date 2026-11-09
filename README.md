# POPPER: Automated Hypothesis Validation with Agentic Sequential Falsifications

This is the [AI1379 research fork](https://github.com/AI1379/POPPER) of
[snap-stanford/POPPER](https://github.com/snap-stanford/POPPER), based on upstream
commit [`5961186`](https://github.com/snap-stanford/POPPER/commit/59611864e09dd15d7528e240ad87c7ddbb4a701c).
The original paper and authors are credited below.

## Research fork changes

| Area | Changes in this fork |
| --- | --- |
| Model endpoints | Explicit `base_url` and API-key routing across the proposal, checking, execution and summary clients. |
| Thinking and structured output | Compatible client retaining provider reasoning history, with opt-in automatic tool selection, local schema validation and bounded format-only repairs. |
| Execution provenance | Optional guarded text and native-tool executors; accepted p-values come from result IDs registered during a successful execution of the current test. |
| Python execution | AST-based execution handles final assignments and multiline statements, preserves tracebacks, and restores output/logging streams. |
| Output and context | Lossless tool-output archives, repeated-warning counts, diagnostic-preserving output views and bounded historical observations. |
| Runtime budgets | Optional terminable Python workers, request/compute limits, timing-based extensions and checkpointed resampling with a fixed plan. |
| Audit and recovery | Usage/protocol journals and opt-in HTTP request/response checkpoints with input/source snapshots, inspection, export and explicit replay. |
| Statistical instructions | Native execution carries the original ReAct template's statistical-integrity requirements, including valid tests, rigorous aggregation and no p-hacking. |

The upstream p-to-e calibrators and sequential aggregation/decision functions are
unchanged. Execution provenance verifies where a result came from; it does not
establish the statistical validity of generated code. Tool protocols, context
compression and budget feedback can affect model behavior, so record their
settings when comparing experiments.

The public API defaults to `execution_mode="legacy"` and native structured
output. Guarded execution, prompt-based structured submission, output compaction,
runtime budgets and request recording are optional. Examples below describe
their entry points. The companion `popper-bias` experiment runner is maintained
separately and is not included in this repository; its reviewer, persona and
other experimental prompt variants are not default engine policies.

This repository hosts the code base for the paper

**Automated Hypothesis Validation with Agentic Sequential Falsifications**

Kexin Huang*, Ying Jin*, Ryan Li*, Michael Y. Li, Emmanuel Candès, Jure Leskovec\
[Link to Paper](https://arxiv.org/abs/2502.09858)


If you find this work useful, please consider cite:

```
@misc{popper,
      title={Automated Hypothesis Validation with Agentic Sequential Falsifications}, 
      author={Kexin Huang and Ying Jin and Ryan Li and Michael Y. Li and Emmanuel Candès and Jure Leskovec},
      year={2025},
      eprint={2502.09858},
      archivePrefix={arXiv}
}
```


### Overview
Hypotheses are central to information acquisition, decision-making, and discovery. However, many real-world hypotheses are abstract, high-level statements that are difficult to validate directly. 
This challenge is further intensified by the rise of hypothesis generation from Large Language Models (LLMs), which are prone to hallucination and produce hypotheses in volumes that make manual validation impractical. Here we propose Popper, an agentic framework for rigorous automated validation of free-form hypotheses. 
Guided by Karl Popper's principle of falsification, Popper validates a hypothesis using LLM agents that design and execute falsification experiments targeting its measurable implications. A novel sequential testing framework ensures strict Type-I error control while actively gathering evidence from diverse observations, whether drawn from existing data or newly conducted procedures.
We demonstrate Popper on six domains including biology, economics, and sociology. Popper delivers robust error control, high power, and scalability. Furthermore, compared to human scientists, Popper achieved comparable performance in validating complex biological hypotheses while reducing time by 10 folds, providing a scalable, rigorous solution for hypothesis validation.


<p align="center"><img src="https://github.com/snap-stanford/POPPER/blob/main/figs/popper_agent_illustration.png" alt="logo" width="800px" /></p>


## Installation

We highly recommend using a virtual environment to manage the dependencies.

```bash
conda create -n popper_env python=3.10
conda activate popper_env
```

For direct usage of Popper, you can install the package via pip:
```bash
pip install popper_agent
```

The PyPI package above follows upstream. To use or develop this fork, install
from its source checkout:
```bash
git clone https://github.com/AI1379/POPPER.git
cd POPPER
pip install -e .
```

Add the OpenAI/Anthropic API key to the environment variables:
```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export ANTHROPIC_API_KEY="YOUR_API_KEY"
```

Datasets will be automatically downloaded to specified data folder when you run the code.

## Demo

A demo is provided in [here](demo.ipynb) to show how to use the Popper agent to validate a hypothesis and basic functionalities of the Popper agent.

## Core API Usage

```python
from popper import Popper

# Initialize the Popper agent
agent = Popper(llm="claude-3-5-sonnet-20240620")

# Register data for hypothesis testing; 
# for bio/discoverybench data in the paper, 
# it will be automatically downloaded to your specified data_path
agent.register_data(data_path='path/to/data', loader_type='bio')

# Configure the agent with custom parameters
agent.configure(
    alpha=0.1,
    max_num_of_tests=5,
    max_retry=3,
    time_limit=2,
    aggregate_test='E-value',
    relevance_checker=True,
    use_react_agent=True
)

# Validate a hypothesis
results = agent.validate(hypothesis="Your hypothesis here")

# Print the results
print(results)
```

## Running locally-served LLM with OpenAI-Compatible API
**Popper** supports inferencing with local LLM servers such as vLLM, SGLang, and llama.cpp, as long as they support OpenAI-compatible API. Here are some example usage with locally hosted LLMs:

Using [SGLang](https://github.com/sgl-project/sglang/tree/main):
```bash
# mistral large 2 with SGLang, using 4 GPUs with 8-bit quantization
python -m sglang.launch_server --model-path mistralai/Mistral-Large-Instruct-2411 --port 40000 --host 0.0.0.0 --tp 4 --quantization fp8 --mem-fraction-static 0.8 --trust-remote-code
```
```python
from popper import Popper
agent = Popper(llm="mistralai/Mistral-Large-Instruct-2411", is_locally_served=True, server_port=40000)
agent.configure(alpha=0.1)
agent.register_data(data_path='path/to/data', loader_type='bio')
agent.validate(hypothesis = 'YOUR HYPOTHESIS')
```

Using [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html):
```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```
```python
from popper import Popper
agent = Popper(llm="NousResearch/Meta-Llama-3-8B-Instruct", is_locally_served=True, server_port=8000, api_key="token-abc123")
```

Using [llama.cpp](https://github.com/ggml-org/llama.cpp):
```bash
llama-server -m model.gguf --port 8080
```
```python
from popper import Popper
agent = Popper(llm="qwen2 1.5B", is_locally_served=True, server_port=8080)
```

## Run on your own hypothesis and database

### Custom model names and OpenAI-compatible endpoints

Pass an optional `base_url` to use any model name on an OpenAI-compatible API:

```python
import os
from popper import Popper

agent = Popper(
    llm="your-custom-model",
    base_url="https://your-model-server.example/v1",
    api_key=os.environ["MY_LLM_API_KEY"],
)
```

An explicit `base_url` takes priority over model-name prefixes and `server_port`.
It is passed to all model clients, including the ReAct executor. No
`is_locally_served` or `server_port` argument is needed for this mode.
For authentication, an explicit non-placeholder `api_key` takes priority; otherwise
`OPENAI_API_KEY` is used when set, falling back to `"EMPTY"` for unauthenticated
servers. The service must support the structured outputs/tool calls used by POPPER.

Omitting `base_url` preserves the existing GPT/Claude/local model selection.

The compatible client also supports an opt-in `structured_output_mode="prompt"`.
This mode exposes only the requested schema tool with `tool_choice="auto"`, adds
an explicit submission instruction, and validates the tool name and arguments
locally. Invalid submissions receive format-only repair feedback (two retries by
default); unexpected calls are never executed. Successful outputs preserve the
existing Pydantic/dict and `include_raw` contracts. Exhaustion raises an error or
returns `parsed=None` with `parsing_error` when `include_raw=True`.

Thinking responses retain `reasoning_content` for subsequent tool-call requests.
Prompt mode uses complete, non-streaming responses because the installed
LangChain version discards reasoning deltas when streaming. Native mode remains
the default. After registering data and calling `agent.configure(...)`, select
the policy on every configured compatible client:

```python
from popper.request_scene import configured_clients

for client in configured_clients(agent).values():
    if client is not None:
        client.structured_output_mode = "prompt"
        client.structured_max_retries = 2
```

This example requires the explicit `base_url` mode above. Provider-specific
thinking options can be set through each client's `extra_body`; prompt mode
does not itself enable or disable thinking. The separate `popper-bias` runner
applies these options and records them in each experiment's configuration.

Offline regression tests (no external API requests):

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Use pytest to include both unittest-based cases and pytest functions, including
the request-checkpoint and native statistical-instruction regressions.

ReAct execution can be selected with the optional
`agent.configure(execution_mode="native_tools")` argument. The other choices are
`"legacy"` and `"guarded_text"`. The default preserves the
legacy text protocol. `guarded_text` rejects mixed action/observation transcripts;
`native_tools` uses actual API tool calls and replays assistant reasoning with
the tool history. Both guarded modes require Python to register a result through
`record_test_result(p_value, n=..., estimate=...)`, then submit its result ID.
Only successful executions from the current test can supply the accepted p-value.
This validates execution provenance, not the correctness of generated statistics.
Native execution requires a model client supporting `bind_tools` and
`tool_choice="auto"`; other structured-output nodes keep their existing settings.
The separate `popper-bias` workspace contains the paired `harness_audit.py`
experiment; it is not required to use this library's execution modes.

### Loading your data

You can simply dump in a set of datasets in your domain (e.g. business, economics, political science, etc.) and run Popper on your own hypothesis.
We only expect every file is in a csv or pkl format.

```python
from popper import Popper   

agent = Popper(llm="claude-3-5-sonnet-20240620")
agent.configure(alpha = 0.1)
agent.register_data(data_path='path/to/data', loader_type='custom')
agent.validate(hypothesis = 'YOUR HYPOTHESIS')
```

## Hypothesis in Popper

You can arbitrarily define any free-form hypothesis. In the paper, we provide two types of hypothesis: biological hypothesis and discovery-bench hypothesis.

You can load the biological hypothesis with:

```python
from popper.benchmark import gene_perturb_hypothesis
bm = gene_perturb_hypothesis(num_of_samples = samples, permuted=False, dataset = 'IL2', path = path)
example = bm.get_example(0)
```
It will return something like:

```
{'prompt': 'Gene VAV1 regulates the production of Interleukin-2 (IL-2).',
 'gene': 'VAV1',
 'answer': 2.916,
 'binary_answer': True}
```

`num_of_samples` is the number of samples you want to generate, `permuted` is whether you want to permute the dataset for type I error estimation, and `dataset` is the dataset you want to use and you can choose from `IL2` and `IFNG`.

For discovery-bench, you can load the hypothesis with:

```python
from popper.benchmark import discovery_bench_hypothesis
bm = discovery_bench_hypothesis(num_samples = samples, path = path)
example = bm.get_example(0)
```

It will return something like:

```
{'task': 'archaeology',
 'domain': 'humanities',
 'metadataid': 5,
 'query_id': 0,
 'prompt': 'From 1700 BCE onwards, the use of hatchets and swords increased while the use of daggers decreased.',
 'data_loader': <popper.utils.DiscoveryBenchDataLoader at 0x7c20793e9f70>,
 'answer': True}
```

As each hypothesis in discoverybench has its own associated dataset, the example will return `data_loader` its own dataset.


## Run benchmarks in the paper

Bash scripts for reproducing the paper is provided in the `benchmark_scripts/run_targetval.sh` for `TargetVal` benchmark and `benchmark_scripts/run_discoverybench.sh` for `DiscoveryBench` benchmark.

**Note:** the Popper agent can read or write files to your filesystem. We recommend running the benchmark scripts inside a containerized environments. We have provided a working `Dockerfile` and an example script to launch a Docker container and execute the scripts in `benchmark_scripts/run_discoverybench_docker.sh`.

**To run paper benchmarks with locally-served models,** you can simply passed in the extra parameters to the benchmark script, e.g.,
```bash
python benchmark_scripts/run_discovery_bench.py --exp_name discovery_bench --model llama-3.3-70b --num_tests 5 --samples 100 --permute --e_value --react --relevance_checker --is_locally_served --server_port 30000 --path PATH_TO_YOUR_DATASET
```

## UI interface
You can deploy a simple UI interface with one line of code using your datasets or our bio dataset - a gradio UI will be generated and you can interact with it to validate your hypothesis. 

```python
agent.launch_UI()
```

An interface like this will be popped up:

[![demo](https://img.youtube.com/vi/jYFEeP2mEY8/0.jpg)](https://www.youtube.com/watch?v=jYFEeP2mEY8)

## Acknowledgement
The DiscoveryBench benchmark and some of the baseline agents are built on top of [allenai/discoverybench](https://github.com/allenai/discoverybench). Thanks for their awsome work!

## Contact

For any questions, please raise an issue in the GitHub or contact Kexin Huang (kexinh@cs.stanford.edu).

## Model request checkpoints

The research runner now separates terminal statistical decisions from report
generation. `SequentialFalsificationTest.decision_snapshot` is set before the
summarizer; an optional `decision_path` writes it atomically. `test_audit` records
failed implementations, internal attempts and relevance rejections separately.
The companion `popper-bias` runner retains completed decisions after reporting
errors and reports failed-test rates and fixed-trajectory p=1 sensitivity results.

Guarded executors pass ordinary syntax errors to Python as observations and count
only consecutive protocol violations against the repair limit. An explicit
conservative fallback is `record_test_result(1.0, fallback_reason="reason")`,
followed by submission of its result ID. P-value parser exhaustion no longer
reruns the whole legacy analysis. These changes define a new research harness;
`legacy` is not an exact upstream baseline. Offline differential tests under
`tests/test_upstream_comparison.py` use source excerpts pinned to upstream commit
`59611864e09dd15d7528e240ad87c7ddbb4a701c`, including intentional executor differences.

The compatible OpenAI client can record each model call before HTTP transmission.
Enable it with `Popper(..., base_url=..., request_scene_dir="artifacts/request_scenes")`,
or call `popper.configure_request_recording(directory)` after `configure()`.
The statistical experiment runner also enables this when `run_popper` receives
an `artifact_root`.

Each scene contains the actual serialized request body, readable request JSON,
response bytes, endpoint, role, timestamps, hashes, transport timeout and library
versions. Tool schemas, delivered tool-result views and provider continuation
fields are preserved. A failed request retains its pre-send snapshot, SDK retries
receive separate scene IDs, and incomplete response streams are marked partial.
Authentication headers are excluded. Known credential values echoed in a body
are redacted, and the scene records that its body is no longer exact.

With the statistical runner, `request_scenes/context/` also contains configuration,
public CSV/problem inputs and the Python source snapshot. Files larger than 64 MiB
are referenced by original path and SHA-256 in the manifest. These checkpoints
restore the model-call context. Python kernel memory and arbitrary process state
require a separate process-checkpoint mechanism.

```bash
python -m popper.request_scene list artifacts/request_scenes
python -m popper.request_scene inspect artifacts/request_scenes/designer/SCENE_ID
python -m popper.request_scene export artifacts/request_scenes/designer/SCENE_ID --output recovered_call
python -m popper.request_scene replay recovered_call
```

`replay` defaults to hash checks and a metadata preview. To send one request using
the recorded body, supply `--send`, a fresh output directory, an explicit matching
endpoint and the environment-variable name holding its API key.

```bash
python -m popper.request_scene replay recovered_call --send --output replay_01 --endpoint https://api.deepseek.com/chat/completions --api-key-env DEEPSEEK_API_KEY
```

The replay command records the new response and leaves returned tool calls for
inspection. Its sending mode supports non-streaming requests. Streaming scenes
remain available for inspection/export. `request_public.json` and
`response_public.json` omit opaque continuation fields from readable exports.
Raw scene bodies retain those fields when the provider supplied them.
