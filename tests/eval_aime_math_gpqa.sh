MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

# iterate over datasets AIME2024 AIME2025 GPQA-D and MATH500
for dataset in HuggingFaceH4/aime_2024 MathArena/aime_2025 zwhe99/gpqa_diamond_mc HuggingFaceH4/MATH-500; do
    echo "--------------------------------"
    echo DATASET: $dataset

    # number of repetitions as well as dataset keys depends on the dataset
    if [ $dataset = "HuggingFaceH4/MATH-500" ]; then
        dataset_key="test"
        problem_key="problem"
        solution_key="answer"
        num_repetitions=1
    elif [ $dataset = "HuggingFaceH4/aime_2024" ]; then
        dataset_key="train"
        problem_key="problem"
        solution_key="answer"
        num_repetitions=5
    elif [ $dataset = "MathArena/aime_2025" ]; then
        dataset_key="train"
        problem_key="problem"
        solution_key="answer"
        num_repetitions=5
    elif [ $dataset = "zwhe99/gpqa_diamond_mc" ]; then
        dataset_key="test"
        problem_key="problem"
        solution_key="solution"
        num_repetitions=5
    fi
    echo "dataset_key: $dataset_key"
    echo "problem_key: $problem_key"
    echo "solution_key: $solution_key"

    # repeat num_repetitions times
    for i in {1..num_repetitions}; do
        echo "Running test $i for dataset $dataset"
        #uv run python examples/run_eval_with_planted_thinking.py data.shuffle_seed=$RANDOM data.dataset_name=$dataset data.dataset_key=$dataset_key data.problem_key=$problem_key data.solution_key=$solution_key generation.vllm_cfg.max_model_len=24000 --config tests/configs/eval_nano.yaml  2>& 1 | grep -E "score=|Mean length"
    done
done

