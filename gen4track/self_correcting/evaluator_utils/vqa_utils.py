import os
from pathlib import Path
from PIL import Image
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional


def load_image(item_id, model_type, image_dir='../mspice/images/image_v1/'):
    """Load an image file corresponding to the given item_id and model_type."""
    image_path = Path(image_dir) / f'{item_id}_{model_type}.jpg'
    if os.path.exists(image_path):
        return Image.open(image_path).convert('RGB')
    return False


def parse_data_type(src_line):
    """Extract the data source type by removing the last part after the final underscore."""
    return '_'.join(src_line.split('_')[:-1])


def format_question(question, choices):
    """Format a multiple-choice question with choices for model input."""
    return f'Question: {question} Choices: {", ".join(choices)}. Answer:'


def calc_vqa_score(qid2answer, qid2dependency=None, qid2gtanswer=None) -> Dict[str, Any]:
    """
    Calculate VQA accuracy scores at question level and aggregate to item-level scores.

    Args:
        qid2answer: Mapping from question ID to model answer (typically 'yes'/'no').
        qid2dependency: Optional mapping from question ID to list of parent question IDs.
        qid2gtanswer: Optional mapping from question ID to ground-truth answer (default: all 'yes').

    Returns:
        Dictionary containing per-question scores, validity, and averaged scores.
    """
    if qid2gtanswer is None:
        qid2gtanswer = {qid: 'yes' for qid in qid2answer.keys()}

    qid2scores = {}
    for qid, answer in qid2answer.items():
        gt_answer = qid2gtanswer[qid]
        qid2scores[qid] = float(answer == gt_answer)

    # Average accuracy without considering dependencies
    try:
        average_score_without_dep = sum(qid2scores.values()) / len(qid2scores)
    except ZeroDivisionError:
        average_score_without_dep = 0.0

    # Handle dependencies: invalidate questions whose parent questions were answered incorrectly
    qid2validity = {}
    qid2scores_after_filtering = deepcopy(qid2scores)

    if qid2dependency is None:
        qid2dependency = {qid: [0] for qid in qid2answer.keys()}

    for qid, parent_ids in qid2dependency.items():
        any_parent_answered_no = False
        for parent_id in parent_ids:
            if parent_id == 0:  # 0 typically means no dependency
                continue
            if qid2scores.get(parent_id, 1.0) == 0:  # Parent answered incorrectly
                any_parent_answered_no = True
                break
        if any_parent_answered_no:
            qid2scores_after_filtering[qid] = 0.0
            qid2validity[qid] = False
        else:
            qid2validity[qid] = True

    # Average accuracy with dependency filtering
    try:
        average_score_with_dep = sum(qid2scores_after_filtering.values()) / len(qid2scores)
    except ZeroDivisionError:
        average_score_with_dep = 0.0

    return {
        'qid2dependency': qid2dependency,
        'qid2answer': qid2answer,
        'qid2scores': qid2scores,
        'qid2validity': qid2validity,
        'average_score_with_dependency': average_score_with_dep,
        'average_score_without_dependency': average_score_without_dep
    }


##### mPLUG-large #####

class MPLUG:
    """Wrapper for mPLUG-large visual question answering model using ModelScope pipeline."""
    def __init__(self, ckpt='damo/mplug_visual-question-answering_coco_large_en'):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt)

    def vqa(self, image, question):
        """Perform VQA inference and return the generated text answer."""
        input_vqa = {'image': image, 'question': question}
        result = self.pipeline_vqa(input_vqa)
        return result['text']


##### InstructBLIP #####

class InstructBLIP:
    """Wrapper for InstructBLIP model (Vicuna-7B variant) for visual question answering."""
    def __init__(self, ckpt='Salesforce/instructblip-vicuna-7b'):
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        self.processor = InstructBlipProcessor.from_pretrained(ckpt)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(ckpt)

    def vqa(self, image, question):
        """Generate answer using beam search decoding."""
        device = next(self.model.parameters()).device
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(device)
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


##### GPT-4o #####

import openai
import base64
import io


def encode_image(image_input):
    """Encode image (path or PIL Image) to base64 string for OpenAI API."""
    if isinstance(image_input, str):
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image_input, Image.Image):
        img_byte_arr = io.BytesIO()
        image_input.save(img_byte_arr, format=image_input.format or 'JPEG')
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    else:
        raise ValueError("Invalid input: must be a file path or a PIL Image object.")


class GPT4o:
    """Wrapper for GPT-4o vision model to perform yes/no visual question answering."""
    def __init__(self, ckpt='gpt-4o'):
        assert openai.api_key is not None, "OpenAI API key is not set"

    def vqa(self, image, question):
        """Query GPT-4o with image and question; enforce strict 'yes'/'no' answer."""
        base64_image = encode_image(image)

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Answer only with 'yes' or 'no'. Do not give other outputs or punctuation marks. Question: {question}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=20,
        )

        answer = response.choices[0].message.content
        answer = answer.lower().strip()

        # Remove common punctuation
        answer = answer.replace(".", "").replace(",", "").replace("?", "").replace("!", "")

        return answer
