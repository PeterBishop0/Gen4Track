from openai import OpenAI

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
        prompt: You are an expert prompt optimizer for text-to-image models. Text-to-image models take a text prompt as input and generate images depicting the prompt as output. You translate prompts written by humans into better prompts for the text-to-image models. Your answers should be concise and effective.\
        Your task is to optimize the user prompt. Below are some previous prompts with the consistency of each prompt’s visual elements in the generated image via a set of binary questions andthe clip score of each visual element (range from 0 to 100). The prompts are arranged in ascending order based on their overall consistency score, which ranges from 0 to 100 (higher is better).\
        {context}
        Generate {num_solutions} paraphrases of the initial prompt which keep the semantic meaning and that have higher scores than all the prompts above. \
        Focus on optimizing for the visual elements that are not consistent. Prioritize optimizing for object with lowest clip scores. Add appropriate details and background descriptions.\
        Respond with each new prompt in between <PROMPT> and </PROMPT>, eg:
        1. <PROMPT>paraphrase 1</PROMPT>
        2. <PROMPT>paraphase 2</PROMPT>
        ...
        {num_solutions}. <PROMPT>paraphrase {num_solutions}</PROMPT>
        """
        # Call the LLM API
        response_text = self.llm_generate_fn(prompt)
        print("LLM Response:", response_text)

        pattern = r"<PROMPT>(.*?)</PROMPT>"
        matches = re.findall(pattern, response_text, flags=re.DOTALL)

        prompts = []
        for p in matches:
                clean_p = p.strip()
                if clean_p:
                        prompts.append(clean_p)

        return prompts
