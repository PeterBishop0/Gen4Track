
def clean_tuple_str(tuple_str):
    """Clean a tuple string by removing everything after the opening parenthesis."""
    tuple_str = tuple_str

    # Remove content inside and after parentheses (e.g., "(x, y, z)" -> "")
    tuple_str = tuple_str.strip().split('(')[0]

    # Remove leading/trailing whitespace
    tuple_str = tuple_str.strip()

    return tuple_str


def parse_tuple_output(output_str) -> dict:
    """Parse the generated tuple output string into a dictionary mapping IDs to cleaned tuple names."""
    if 'output:' in output_str:
        # Extract content after 'output:' marker
        start_index = output_str.index('output:')
        output_str = output_str[start_index + len('output:'):]
        output_str = output_str.strip()

    id2tup = {}
    # Split output into lines, each line format: "ID | tuple_string"
    for id_tup in output_str.strip().split('\n'):
        tup_id, tup = id_tup.split('|')

        tup_id = tup_id.strip()
        tup = tup.strip()

        # Clean the tuple part (remove parameters)
        tup = clean_tuple_str(tup)

        tup_id = int(tup_id)

        id2tup[tup_id] = tup

    return id2tup


def clean_dependency_id(dependency_id_str):
    """Clean a dependency ID string by removing invalid or redundant entries."""
    dependency_ids = dependency_id_str

    # Split by comma
    dependency_ids = dependency_ids.strip().split(',')

    # Remove surrounding whitespace
    dependency_ids = [dep_id.strip() for dep_id in dependency_ids]

    # Keep only numeric IDs or '-' (special marker); filter out non-numeric strings like 'background'
    dependency_ids = [
        dep_id for dep_id in dependency_ids if dep_id.isnumeric() or dep_id == '-']

    # If multiple IDs present, remove '0' as it is often a default/no-dependency marker
    if len(dependency_ids) > 1:
        dependency_ids = [dep_id for dep_id in dependency_ids if dep_id != '0']

    # Join back into comma-separated string
    dependency_ids = ','.join(dependency_ids)

    return dependency_ids


def parse_dependency_output(output_str) -> dict:
    """Parse the generated dependency output string into a dictionary mapping question IDs to dependency ID lists."""
    if 'output:' in output_str:
        # Extract content after 'output:' marker
        start_index = output_str.index('output:')
        output_str = output_str[start_index + len('output:'):]
        output_str = output_str.strip()

    id2dep = {}
    # Split output into lines, each line format: "question_id | dependency_ids"
    for id_dep in output_str.strip().split('\n'):
        id_dep = id_dep.strip()
        if not id_dep:
            continue
        if '|' not in id_dep:
            continue

        question_id, dep = map(str.strip, id_dep.split('|', 1))

        question_id = question_id.strip()
        dep = dep.strip()

        # Clean the dependency string
        dep = clean_dependency_id(dep)

        question_id = int(question_id)

        # Convert cleaned string to list of integers
        deps = [int(d) for d in dep.split(',')]

        id2dep[question_id] = deps

    return id2dep


def parse_question_output(output_str) -> dict:
    """Parse the generated question output string into a dictionary mapping IDs to questions."""
    if 'output:' in output_str:
        # Extract content after 'output:' marker
        start_index = output_str.index('output:')
        output_str = output_str[start_index + len('output:'):]
        output_str = output_str.strip()

    id2question = {}
    # Split output into lines, each line format: "ID | question_text"
    for id_question in output_str.strip().split('\n'):
        question_id, question = id_question.split('|')

        question_id = question_id.strip()
        question = question.strip()

        question_id = int(question_id)

        id2question[question_id] = question

    return id2question
