
import os
import numpy as np
from functools import partial
from openai import OpenAI
from PIL import Image
import torch
from .evaluator_utils.query_utils import generate_dsg
from .evaluator_utils.parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output

import base64
import sys
import json
import imageio
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchmetrics.functional.multimodal import clip_score
import openai
from .evaluator_utils.openai_utils import openai_setup, openai_completion
from .evaluator_utils.vqa_utils import MPLUG


class DCSEvaluator:
    """
    Evaluator for Decomposition-to-CLIP-Score (DCS): measures how well generated images cover noun phrases from the prompt.
    """

    def __init__(self, llm_generate_fn, clip_model_path="openai/clip-vit-base-patch16"):
        """
        Initialize the DCS evaluator.

        Args:
            llm_generate_fn: Function to call LLM for noun decomposition.
            clip_model_path: Identifier for the CLIP model used in scoring.
        """
        self.llm_generate_fn = llm_generate_fn

        # Bind CLIP model to the scoring function
        self.clip_score_fn = partial(
            clip_score,
            model_name_or_path=clip_model_path
        )

    def generate_nouns(self, user_prompt):
        """
        Decompose the input prompt into individual noun phrases using an LLM.

        Args:
            user_prompt: Natural language description of the scene.

        Returns:
            List of extracted noun phrases.
        """
        prompt = f"""
        Decompose the following prompt sentence into individual noun phrases.
        Ignore prefixes such as 'a photo of', 'a picture of', etc.
        Your response should ONLY be a list of comma separated values, e.g.: foo, bar, baz.
        The <prompt> is: {user_prompt}.
        """

        response_text = self.llm_generate_fn(prompt)
        print("LLM Response:", response_text)

        nouns = [noun.strip() for noun in response_text.split(",")]
        return nouns

    @staticmethod
    def load_images_from_directory(directory):
        """
        Load all supported image files (PNG/JPG/JPEG) from a directory.

        Args:
            directory: Path to directory containing images.

        Returns:
            List of PIL Image objects in RGB mode.
        """
        images = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).convert("RGB")
                images.append(img)
        return images

    def calculate_clip_scores(self, images, prompt):
        """
        Compute CLIP similarity scores between images and a text prompt (noun phrase).

        Args:
            images: List of PIL Image objects.
            prompt: Single noun phrase.

        Returns:
            Tensor containing CLIP score(s).
        """
        img_arr = np.stack([np.array(img) for img in images])
        prompts = [prompt] * len(images)

        clip_score_value = self.clip_score_fn(
            torch.from_numpy(img_arr).permute(0, 3, 1, 2),  # Convert to NCHW
            prompts
        ).detach()

        return clip_score_value

    def evaluate_directory(self, directory, input_prompt, top_k=1):
        """
        Full DCS pipeline: extract nouns, load images, and compute CLIP scores.

        Args:
            directory: Directory containing generated images.
            input_prompt: Original text prompt.
            top_k: Number of top images to score per noun (default: 1).

        Returns:
            Tuple of (dictionary mapping noun to score tensor, average score across nouns).
        """
        nouns = self.generate_nouns(input_prompt)
        images = self.load_images_from_directory(directory)

        if len(images) == 0:
            raise ValueError("No images found in directory.")

        results = {}
        total_score = 0.0

        for noun in nouns:
            score = self.calculate_clip_scores(images[:top_k], noun)
            print(f"{noun}: {score}")
            results[noun] = score
            total_score += score.mean().item()  # Average over images if multiple

        average_score = total_score / len(nouns)
        return results, average_score


class DSGEvaluator:
    """Evaluator based on Dependency Skill Graph (DSG): decomposes prompt into tuples, questions, and dependencies, then validates via VQA."""

    def __init__(
        self,
        generate_dsg_fn,
        llm_generate_fn,
        parse_tuple_fn,
        parse_dependency_fn,
        parse_question_fn,
        vqa_model_class
    ):
        """
        Initialize the DSG evaluator.

        Args:
            generate_dsg_fn: Function to generate DSG components using LLM.
            llm_generate_fn: Function to call LLM for generation.
            parse_*_fn: Parsing functions for tuple/dependency/question outputs.
            vqa_model_class: Class for the VQA model (e.g., MPLUG).
        """
        self.generate_dsg_fn = generate_dsg_fn
        self.llm_generate_fn = llm_generate_fn
        self.parse_tuple_fn = parse_tuple_fn
        self.parse_dependency_fn = parse_dependency_fn
        self.parse_question_fn = parse_question_fn
        self.vqa_model = vqa_model_class()

    def run(self, image_path, text_prompt):
        """Execute the full DSG evaluation pipeline on a single image and prompt."""
        # Load generated image
        generated_image = Image.open(image_path).convert("RGB")

        # Prepare input format for DSG generation
        id2prompts = {'custom_0': {'input': text_prompt}}

        # Generate DSG components (tuples, questions, dependencies)
        id2tuple_outputs, id2question_outputs, id2dependency_outputs = self.generate_dsg_fn(
            id2prompts,
            generate_fn=self.llm_generate_fn
        )

        # Parse generated outputs
        qid2tuple = self.parse_tuple_fn(id2tuple_outputs['custom_0']['output'])
        qid2dependency = self.parse_dependency_fn(id2dependency_outputs['custom_0']['output'])
        qid2question = self.parse_question_fn(id2question_outputs['custom_0']['output'])

        # Run VQA on each question
        qid2answer = {}
        qid2scores = {}
        qid2validity = {}

        for qid, question in qid2question.items():
            answer = self.vqa_model.vqa(generated_image, question)
            qid2answer[qid] = answer.lower().strip()
            qid2scores[qid] = float(qid2answer[qid] == 'yes')

        # Apply dependency constraints: invalidate questions with failed parents
        for qid, parents in qid2dependency.items():
            any_parent_no = any(
                parent != 0 and qid2scores.get(parent, 1) == 0
                for parent in parents
            )
            if any_parent_no:
                qid2scores[qid] = 0
                qid2validity[qid] = False
            else:
                qid2validity[qid] = True

        # Compute final averaged score
        average_score = sum(qid2scores.values()) / len(qid2scores) if qid2scores else 0.0

        return {
            "qid2tuple": qid2tuple,
            "qid2question": qid2question,
            "qid2dependency": qid2dependency,
            "qid2answer": qid2answer,
            "qid2scores": qid2scores,
            "qid2validity": qid2validity,
            "average_score": average_score,
        }

    def format_as_string(self, result_dict):
        """Format DSG results as readable string for output."""
        avg_score = result_dict["average_score"]
        qid2question = result_dict["qid2question"]
        qid2answer = result_dict["qid2answer"]

        sorted_ids = sorted(qid2question.keys(), key=lambda x: int(x))

        lines = []
        lines.append(f"Decomposed scene graph overall score: {avg_score:.4f}")
        lines.append("evaluation questions:")

        for qid in sorted_ids:
            q = qid2question[qid]
            a = qid2answer[qid]
            lines.append(f"{q} → {a}")

        return "\n".join(lines)

    def print_results(self, result_dict):
        """Print detailed per-question DSG evaluation results."""
        print("Per-question eval results")
        for qid in result_dict["qid2question"]:
            print("ID:", qid)
            print("question:", result_dict["qid2question"][qid])
            print("answer:", result_dict["qid2answer"][qid])
            print("validity:", result_dict["qid2validity"].get(qid, True))
            print("score:", result_dict["qid2scores"][qid])
            print()

        print("average score:", result_dict["average_score"])


class VideoEvaluator:
    """Combined evaluator integrating DSG (semantic fidelity) and DCS (noun coverage via CLIP)."""

    def __init__(self, dsg_evaluator, noun_evaluator):
        """
        Initialize combined evaluator.

        Args:
            dsg_evaluator: Instance of DSGEvaluator.
            noun_evaluator: Instance of DCSEvaluator.
        """
        self.dsg = dsg_evaluator
        self.noun = noun_evaluator

    def run(self, image_path, prompt, image_dir=None):
        """
        Run both DSG and DCS evaluations and combine results.

        Args:
            image_path: Path to the main generated image (for VQA).
            prompt: Original text prompt.
            image_dir: Directory of generated frames/images (for CLIP scoring).

        Returns:
            Formatted output string and final combined score.
        """
        # Run DSG evaluation
        dsg_result = self.dsg.run(image_path, prompt)
        dsg_text = self.dsg.format_as_string(dsg_result)

        # Run DCS (CLIP-based noun coverage)
        clip_scores, clip_avg_score = self.noun.evaluate_directory(image_dir, prompt)

        # Format output
        output = "Scene Graph Evaluation Results:\n"
        output += dsg_text + "\n\nNoun-level CLIP Scores:\n"

        for noun, score_tensor in clip_scores.items():
            score_value = score_tensor.mean().item()
            output += f"{noun}: {score_value:.4f}\n"

        # Combined final score (normalized DSG + CLIP average)
        total_score = (dsg_result["average_score"] * 100 + clip_avg_score) / 2

        return output, total_score


if __name__ == "__main__":
    # Setup OpenAI API
    openai_setup("gen4track/self_correcting/evaluator_utils/openai_key.txt")

    # Initialize DSG evaluator with LLM and VQA model
    dsg_evaluator = DSGEvaluator(
        generate_dsg_fn=generate_dsg,
        llm_generate_fn=openai_completion,
        parse_tuple_fn=parse_tuple_output,
        parse_dependency_fn=parse_dependency_output,
        parse_question_fn=parse_question_output,
        vqa_model_class=MPLUG
    )

    # Initialize DCS (noun-level CLIP) evaluator
    noun_clip_evaluator = DCSEvaluator(llm_generate_fn=openai_completion)

    # Combined pipeline
    pipeline = VideoEvaluator(dsg_evaluator, noun_clip_evaluator)

    # Example usage
    output_text, final_score = pipeline.run(
        image_path="/home/zhangxinyu/Gen4Track/outputs/tiger-17/sam-test/tiger/frames/0000.png",
        prompt="(masterpiece, best quality, ultra-detailed, 4k, highres), (realistic, photorealistic, sharp focus), a white tiger walking by the riverbank",
        image_dir="/home/zhangxinyu/Gen4Track/outputs/tiger-17/sam-test/tiger/frames"
    )

    print(output_text)
    print(f"Final combined score: {final_score:.4f}")