# Set Up Clusters

This guide explains how to initialize NeMo RL clusters using various methods.

## Slurm

 The following code provides instructions on how to use Slurm to run batched job submissions and run jobs interactively.

### Batched Job Submission

```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

COMMAND="uv pip install -e .; uv run ./examples/run_grpo_math.py" \
RAY_DEDUP_LOGS=0 \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
```

Notes:
* Some clusters may or may not need `--gres=gpu:8` to be added to the `sbatch` command.

Which will print the `SLURM_JOB_ID`:
```text
Submitted batch job 1980204
```
Make note of the the job submission number. Once the job begins, you can track its process in the driver logs which you can `tail`:
```sh
tail -f 1980204-logs/ray-driver.log
```

:::{note}
`UV_CACHE_DIR` defaults to `$SLURM_SUBMIT_DIR/uv_cache` and is mounted to head and worker nodes
to ensure fast `venv` creation.

You can override the default location by setting it to a path accessible by all head and worker nodes via:

```sh
...
UV_CACHE_DIR=/path/that/all/workers/and/head/can/access \
sbatch ... \
    ray.sub
```
:::

### Interactive Launching

:::{tip}
A key advantage of running interactively on the head node is the ability to execute multiple multi-node jobs without needing to requeue in the Slurm job queue. This means that during debugging sessions, you can avoid submitting a new `sbatch` command each time. Instead, you can debug and re-submit your NeMo RL job directly from the interactive session.
:::

To run interactively, launch the same command as the [Batched Job Submission](#batched-job-submission), except omit the `COMMAND` line:
```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

RAY_DEDUP_LOGS=0 \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
```
Which will print the `SLURM_JOB_ID`:
```text
Submitted batch job 1980204
```
Once the Ray cluster is up, a script should be created to attach to the Ray head node,
which you can use to launch experiments.
```sh
bash 1980204-attach.sh
```
Now that you are on the head node, you can launch the command as follows:
```sh
uv venv .venv
uv pip install -e .
uv run ./examples/run_grpo_math.py
```

## Kubernetes

TBD

The following code provides instructions on how to use Kubernetes to run your jobs.