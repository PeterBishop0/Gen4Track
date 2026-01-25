from openai import OpenAI
import re
def prompt_optimizer(llm_generate_fn,prompt_history, num_solutions):
        '''
        evaluation_result for example:
                Decomposed scene graph overall score: 0.4
                evaluation questions:
                Is there a bird? yes
                Is the bird green? no
                Is the bird's head white? yes
                Is there a riverbank? no
                Is the bird standing on the riverbank? yes
                Decomposed Clip score : 25.67
                visual elements:
                bird : tensor(26.6724)
                riverbank : tensor(22.8194)
                white head : tensor(24.5207)

        '''
        # combine multiple prompt history together
        context = ""

        for item in prompt_history:
                context += (
                        f"\n=== Round {item['round']} ===\n"
                        f"Prompt:\n{item['prompt']}\n\n"
                        f"Overall Score: {item.get('score', 'N/A')}\n\n"
                        f"Evaluation Result:\n{item['evaluation']}\n"
                )


        prompt = f"""\
        You are an expert prompt engineer for Stable Diffusion–style text-to-image models.

        Stable Diffusion expects prompts written as:
        - comma-separated keywords or short phrases
        - no complete sentences
        - no narrative or instructional language
        - no verbs such as "create", "show", or "depict"
        - explicit visual attributes (object appearance, pose, environment, lighting, camera)
        - parentheses () may be used for emphasis

        You will be given a context consisting of multiple optimization rounds.
        Each round includes:
        - the prompt used in that round
        - an overall consistency score
        - detailed evaluation results based on a decomposed scene graph

        The evaluation results contain:
        1. Binary judgments for global visual attributes (e.g., realism, sharp focus, ultra-detailed).
        2. Noun-level CLIP scores indicating how well each object is visually grounded.

        Interpretation rules:
        - A binary result of "no" indicates a missing or weak visual cue and MUST be explicitly addressed.
        - A noun with a low CLIP score (<40) lacks visual specificity and MUST be reinforced with concrete attributes.
        - Improving noun-level CLIP scores has higher priority than adding new objects.
        - Do NOT introduce new objects that are not present in the scene graph.
        - Do NOT remove existing objects or relations.

        Optimization objectives:
        - Preserve the original semantic meaning of the prompt.
        - Increase visual specificity and physical realism.
        - Explicitly reinforce low-scoring objects (e.g., appearance, pose, material, spatial relation).
        - Add realistic environment, lighting, and camera cues when needed.
        - Ensure the new prompt would achieve a higher overall consistency score than all previous rounds.

        Context:
        {context}

        Task:
        Generate {num_solutions} optimized prompts that satisfy all the above constraints.

        Output requirements:
        - Output ONLY Stable Diffusion–style prompts.
        - Each prompt must be written in tag-based, comma-separated format.
        - Do NOT output explanations, reasoning, or analysis.
        - Do NOT repeat or lightly paraphrase previous prompts.

        Output format:
        1. <PROMPT>optimized prompt 1</PROMPT>
        2. <PROMPT>optimized prompt 2</PROMPT>
        ...
        {num_solutions}. <PROMPT>optimized prompt {num_solutions}</PROMPT>
        """
        # Call the LLM API
        response_text = llm_generate_fn(prompt)
        print("LLM Response:", response_text)

        pattern = r"<PROMPT>(.*?)</PROMPT>"
        matches = re.findall(pattern, response_text, flags=re.DOTALL)

        prompts = []
        for p in matches:
                clean_p = p.strip()
                if clean_p:
                        prompts.append(clean_p)

        return prompts
