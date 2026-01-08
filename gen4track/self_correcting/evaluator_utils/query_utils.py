import pandas as pd
import string
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

import tqdm
from pprint import pprint


# Main prompt template combining preamble, examples, and test input
_PROMPT_TEMPLATE = string.Template("""
$preamble

$examples

$test_input_output
""".strip())

# Template for in-context examples
_EXAMPLES_TEMPLATE = string.Template("""
$input_name: $input
$output_name: $output""".strip())

# Template for the test input (where model generates output)
_TEST_TEMPLATE = string.Template("""
$input_name: $test_input
$output_name: """.lstrip())

# Supported task names
_TASK_NAMES = [
    "tuple",
    "dependency",
    "question",
]

# Task-specific instruction preambles
_TUPLE_PREAMBLE = """Task: given input prompts, describe each scene with skill-specific tuples.
Do not generate same tuples again. Do not generate tuples that are not explicitly described in the prompts.
output format: id | tuple
""".strip()

_DEPENDENCY_PREAMBLE = """Task: given input prompts and tuples, describe the parent tuples of each tuple.
output format: id | dependencies (comma separated)
""".strip()

_QUESTION_PREAMBLE = """Task: given input prompts and skill-specific tuples, re-write tuple each in natural language question.
output format: id | question
""".strip()


def load_tifa160_data(path='tifa160-dev-anns.csv'):
    """Load TIFA160 annotation data from CSV file."""
    path = Path(__file__).parent / 'data' / path
    data_df = pd.read_csv(path)
    return data_df


def create_train_example(
    prompt: str,
    task: str = "tuple",
    tuples: Optional[List[str]] = None,
    dependencies: Optional[List[str]] = None,
    questions: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Create a formatted in-context example for tuple, dependency, or question generation.

    Args:
        prompt: Original text prompt.
        task: One of "tuple", "dependency", or "question".
        tuples: List of semantic tuples (required for dependency/question tasks).
        dependencies: List of dependency strings (for dependency task).
        questions: List of natural language questions (for question task).

    Returns:
        Dictionary with formatted "input" and "output" strings.
    """
    assert task in _TASK_NAMES, f"task == {task}"

    inputs = []
    outputs = []
    n_outputs = len(tuples) if tuples else 0

    if task == "tuple":
        # Input: just the prompt
        inputs += [prompt]
        # Output: numbered tuples
        for i in range(n_outputs):
            output = f"{i+1} | {tuples[i]}"
            output = " ".join(output.split())
            outputs += [output]

    elif task == "dependency":
        # Input: prompt + numbered tuples
        inputs += [prompt]
        for i in range(n_outputs):
            input_line = f"{i+1} | {tuples[i]}"
            input_line = " ".join(input_line.split())
            inputs += [input_line]
        # Output: numbered dependencies
        for i in range(n_outputs):
            output = f"{i+1} | {dependencies[i]}"
            output = " ".join(output.split())
            outputs += [output]

    elif task == "question":
        # Input: prompt + numbered tuples
        inputs += [prompt]
        for i in range(n_outputs):
            input_line = f"{i+1} | {tuples[i]}"
            input_line = " ".join(input_line.split())
            inputs += [input_line]
        # Output: numbered questions
        for i in range(n_outputs):
            output = f"{i+1} | {questions[i]}"
            output = " ".join(output.split())
            outputs += [output]

    return {
        "input": "\n".join(inputs),
        "output": "\n".join(outputs),
    }


def tifa_id2example(
    df: pd.DataFrame,
    id: str,
    task: str = "tuple",
) -> Dict[str, str]:
    """Extract and format a single example from TIFA annotation dataframe by item ID."""
    prompt = df[df.item_id == id].text.tolist()[0]
    all_tuples = df[df.item_id == id].tuple.tolist()
    all_dependencies = df[df.item_id == id].dependency.tolist()
    all_questions = df[df.item_id == id].question_natural_language.tolist()

    example = create_train_example(
        prompt=prompt,
        task=task,
        tuples=all_tuples,
        dependencies=all_dependencies,
        questions=all_questions,
    )

    return example


def get_tifa_examples(data_df, ids, task='tuple'):
    """Retrieve formatted examples for multiple IDs."""
    examples = []
    for id in ids:
        example = tifa_id2example(data_df, id, task=task)
        examples += [example]
    return examples


# Predefined training IDs for in-context learning
TIFA160_ICL_TRAIN_IDS = [
    'coco_361740', 'drawbench_155', 'partiprompt_86', 'paintskill_374',
    'coco_552592', 'partiprompt_1414', 'coco_627537', 'coco_744388',
    'partiprompt_1108', 'coco_397109', 'coco_666114', 'coco_62896',
    'paintskill_235', 'drawbench_159', 'partiprompt_893', 'coco_322041',
    'coco_292534', 'drawbench_57', 'partiprompt_555', 'coco_488166',
    'partiprompt_726', 'coco_323167', 'coco_625027',
]
assert len(TIFA160_ICL_TRAIN_IDS) == 23

# Load data and precompute in-context examples
_TIFA160_DF = load_tifa160_data()
_TUPLE_EXAMPLES = get_tifa_examples(_TIFA160_DF, TIFA160_ICL_TRAIN_IDS, task='tuple')
_DEPENDENCY_EXAMPLES = get_tifa_examples(_TIFA160_DF, TIFA160_ICL_TRAIN_IDS, task='dependency')
_QUESTION_EXAMPLES = get_tifa_examples(_TIFA160_DF, TIFA160_ICL_TRAIN_IDS, task='question')


def make_prompt(
    examples: List[Dict[str, str]],
    test_input: str,
    preamble: str = _TUPLE_PREAMBLE,
    input_name: str = "input",
    output_name: str = "output",
    verbose: bool = False,
) -> str:
    """
    Construct full prompt from preamble, in-context examples, and test input.

    Returns:
        Complete prompt string ready for language model inference.
    """
    # Format examples
    examples_str = []
    for example in examples:
        examples_str.append(
            _EXAMPLES_TEMPLATE.substitute(
                input_name=input_name,
                output_name=output_name,
                input=example["input"].strip(),
                output=example["output"].strip(),
            )
        )
    examples_str = "\n\n".join(examples_str)

    # Format test input
    test_input_str = _TEST_TEMPLATE.substitute(
        input_name=input_name,
        output_name=output_name,
        test_input=test_input
    )

    # Combine all parts
    prompt = _PROMPT_TEMPLATE.substitute(
        preamble=preamble,
        examples=examples_str,
        test_input_output=test_input_str,
    )

    if verbose:
        print(f"len(preamble): {len(preamble)} chars & {len(preamble.split())} words")
        print(f"len(examples): {len(examples_str)} chars & {len(examples_str.split())} words")
        print(f"len(total): {len(prompt)} chars & {len(prompt.split())} words")

    return prompt


def parse_with_input_name(text: str, input_name="input") -> str:
    """Extract model generation by cutting off at the next input verbalizer."""
    text = text.split(f"{input_name}:")[0]
    return text


def generate_with_in_context_examples(
    generate_fn: Callable[[str], str],
    id2inputs: Dict[str, Dict[str, str]],
    train_examples: List[Dict[str, Any]],
    preamble: str,
    input_name: str = "input",
    output_name: str = "output",
    parse_fn: Callable[[str], str] = parse_with_input_name,
    num_workers: int = 1,
    verbose=True,
) -> Dict[str, Dict[str, str]]:
    """
    Run inference with in-context examples for a batch of inputs.

    Returns:
        Dictionary mapping IDs to inputs and generated outputs.
    """
    ids = list(id2inputs.keys())

    # Prepare prompts
    total_kwargs = []
    for id_ in tqdm.tqdm(ids, desc="Preparing LM inputs", disable=not verbose):
        test_input = id2inputs[id_]["input"]
        prompt = make_prompt(
            examples=train_examples,
            test_input=test_input,
            preamble=preamble,
            input_name=input_name,
            output_name=output_name,
            verbose=False,
        )
        total_kwargs.append({"prompt": prompt})

    # Run model inference
    if verbose:
        print(f"Running LM calls with {num_workers} workers.")
    if num_workers == 1:
        total_output = [generate_fn(kwargs["prompt"]) for kwargs in tqdm.tqdm(total_kwargs, disable=not verbose)]
    else:
        from multiprocessing import Pool
        with Pool(num_workers) as p:
            total_inputs = [d['prompt'] for d in total_kwargs]
            total_output = list(tqdm.tqdm(p.imap(generate_fn, total_inputs), total=len(total_inputs), disable=not verbose))

    # Post-process outputs
    id2outputs = {}
    for i, id_ in enumerate(tqdm.tqdm(ids, desc="Postprocessing LM outputs", disable=not verbose)):
        test_input = id2inputs[id_]["input"]
        raw_prediction = total_output[i]
        prediction = parse_fn(raw_prediction).strip()

        id2outputs[id_] = {
            "id": id_,
            "input": test_input,
            "output": prediction,
        }

    return id2outputs


def generate_dsg(id2prompts: Dict[str, Dict[str, str]],
				 generate_fn: Callable[[str], str],
                 tuple_train_examples=_TUPLE_EXAMPLES,
                 dependency_train_examples=_DEPENDENCY_EXAMPLES,
                 question_train_examples=_QUESTION_EXAMPLES,
                 N_parallel_workers=1,
				 verbose=True
				 ):
	"""Generate DSG with a LM in three steps with in-context examples.
	
	Args:
		id2prompts: a input dictionary with following structure
			"id" (str) -> {
				"input": text prompt (str)
				"source": (str; optional)
			}
		generate_fn: a method that calls language model with a text input

		tuple_train_examples: list of examples for tuple generation task
		dependency_train_examples: list of examples for dependency generation task
		question_train_examples: list of examples for question generation task
		N_parallel_workers: number of workers for parallel call
		verbose: whether to print tqdm output / intermediate steps

	Returns:
		id2tuple_outputs: output dictionary with key with following structure
			"id" (str) -> {
				"input": text prompt (str),
				"output": generated tuples (str)
			}
		id2question_outputs: output dictionary with key with following structure
			"id" (str) -> {
				"input": text prompt (str),
				"output": generated questions (str)
			}
		id2dependency_outputs: output dictionary with key with following structure
			"id" (str) -> {
				"input": text prompt (str),
				"output": generated dependencies (str)
			}
	"""

	eval_data = []
	for id, input_dict in id2prompts.items():
		datum = {
			'id': id,
			'prompt': input_dict['input']
		}
		eval_data.append(datum)

	test_ids = [datum['id'] for datum in eval_data]

	# =====================================
	# Task 1: Tuple generation
	# =====================================
	task, preamble = ['tuple', _TUPLE_PREAMBLE]

	if verbose:
		print('Task 1: ', task)

	train_examples = tuple_train_examples

	id2inputs = {}
	for i, datum in enumerate(eval_data):
		input_dict = {}

		test_prompt = datum['prompt']
		id = datum['id']

		input_dict['input'] = test_prompt

		id2inputs[id] = input_dict

	if verbose:
		print('Run inference')
	# used as inputs to task 2 (question gen) & task 3 (dependency gen)
	id2tuple_outputs = generate_with_in_context_examples(
		generate_fn=generate_fn,
		id2inputs=id2inputs,
		train_examples=train_examples,
		preamble=preamble,
		num_workers=N_parallel_workers,
		verbose=verbose)

	if verbose:
		print('Sample results:')
		for id in test_ids[:1]:
			print('id:', id)
			pprint(id2tuple_outputs[id])

	# =====================================
	# Task 2: Question generation
	# =====================================
	task, preamble = ['question', _QUESTION_PREAMBLE]

	if verbose:
		print('Task 2: ', task)

	train_examples = question_train_examples

	id2inputs = {}
	for i, datum in enumerate(eval_data):
		input_dict = {}

		id = datum['id']

		test_prompt = datum['prompt']
		gen_tuple = id2tuple_outputs[id]['output'].strip()
		input_dict['input'] = "\n".join([test_prompt, gen_tuple])

		id2inputs[id] = input_dict

	if verbose:
		print('Run inference')
	id2question_outputs = generate_with_in_context_examples(
		generate_fn=generate_fn,
		id2inputs=id2inputs,
		train_examples=train_examples,
		preamble=preamble,
		num_workers=N_parallel_workers,
		verbose=verbose)

	if verbose:
		print('Sample results:')
		for id in test_ids[:1]:
			print('id:', id)
			print(id2question_outputs[id])

	# =====================================
	# Task 3: Dependency generation
	# =====================================
	task, preamble = ['dependency', _DEPENDENCY_PREAMBLE]

	if verbose:
		print('Task 3: ', task)

	train_examples = dependency_train_examples

	id2inputs = {}
	for i, datum in enumerate(eval_data):
		input_dict = {}

		id = datum['id']

		test_prompt = datum['prompt']
		gen_tuple = id2tuple_outputs[id]['output'].strip()
		input_dict['input'] = "\n".join([test_prompt, gen_tuple])

		id2inputs[id] = input_dict

	if verbose:
		print('Run inference')
	id2dependency_outputs = generate_with_in_context_examples(
		generate_fn=generate_fn,
		id2inputs=id2inputs,
		train_examples=train_examples,
		preamble=preamble,
		num_workers=N_parallel_workers,
		verbose=verbose)

	if verbose:
		print('Sample results:')
		for id in test_ids[:1]:
			print('id:', id)
			print(id2dependency_outputs[id])

	return id2tuple_outputs, id2question_outputs, id2dependency_outputs
