from pipeline.invert import Inverter
from pipeline.generator import Generator
from utils import load_config, init_model, seed_everything, get_frame_ids
# from diffusers.models.unet_2d_condition import UNet2DConditionModel
from model.unet_2d_condition import UNet2DConditionModel
from gen4track.self_correcting import VideoEvaluator, prompt_optimizer,DSGEvaluator,DCSEvaluator
from gen4track.self_correcting.evaluator_utils.openai_utils import openai_setup, openai_completion
from gen4track.self_correcting.evaluator_utils.vqa_utils import MPLUG
from gen4track.self_correcting.evaluator_utils.query_utils import generate_dsg
from gen4track.self_correcting.evaluator_utils.parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output

import os
import torch
import time

if __name__ == "__main__":
    config = load_config()
    pipe, scheduler, model_key = init_model(
        config.device, config.sd_version, config.model_key, config.generation.control, config.float_precision)
    config.model_key = model_key
    seed_everything(config.seed)
    unet_ori = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16).to(config.device)
    
    print("Start inversion!")
    inversion = Inverter(pipe, scheduler, config)
    inversion(config.input_path,config.bbox_path, config.inversion.save_path)

    print("Start generation!")
    s_time = time.time()
    generator = Generator(pipe, scheduler, config, unet_ori.to(config.device))
    frame_ids = get_frame_ids(
        config.generation.frame_range, config.generation.frame_ids)
    gen_frame_path = generator(config.input_path, config.generation.latents_path,
              config.generation.output_path, frame_ids=frame_ids)
    e_time = time.time()
    print("cost time：", e_time-s_time)
    
    # Start evaluating and self correcting
    if config.self_correcting: 
        max_iteration = config.max_iteration
        prompt_history = [] 
        openai_setup("gen4track/self_correcting/evaluator_utils/openai_key.txt")
        dsg_evaluator = DSGEvaluator(
                generate_dsg_fn=generate_dsg,
                llm_generate_fn=openai_completion,
                parse_tuple_fn=parse_tuple_output,
                parse_dependency_fn=parse_dependency_output,
                parse_question_fn=parse_question_output,
                vqa_model_class=MPLUG
            )
        noun_clip_evaluator = DCSEvaluator(llm_generate_fn=openai_completion)
        evaluator = VideoEvaluator(dsg_evaluator, noun_clip_evaluator)

        # ===== initial prompt =====
        current_prompt = config.generation.prompt
        current_prompt = current_prompt.get("edit")

        for i in range(0,max_iteration):
            print(f"\n===== Iteration {i} =====")
            print(f"Prompt:\n{current_prompt}")
            # 1. Run evaluation
            try:
                images = os.listdir(gen_frame_path)
                image_path = os.path.join(gen_frame_path, images[0]) 

                evaluation_result, overall_score = evaluator.run(
                    image_path=image_path,
                    prompt=current_prompt,
                    image_dir=gen_frame_path
                )

                # 2. Store history (list[dict])
                prompt_history.append({
                    "round": i,
                    "prompt": current_prompt,
                    "evaluation": evaluation_result,
                    "score": overall_score
                })

                # 3. Optimize prompt based on full history
                num_solutions = 3
                new_prompt = prompt_optimizer(
                    openai_completion,
                    prompt_history,
                    evaluation_result,
                    num_solutions
                )

                # 4. Select next prompt
                current_prompt = new_prompt[0]
                new_output_path = os.path.join(config.generation.output_path, str(i + 1))

                # 5. Restart generation if not qualified
                if overall_score < 40:
                    print(f"Restart {i}-th generation!")
                    gen_frame_path = generator(
                        config.input_path,
                        config.generation.latents_path,
                        new_output_path,
                        frame_ids=frame_ids,
                        restart_generation=True,
                        new_prompt=current_prompt,
                        iteration=i
                    )
                else:
                    break
            except (FileNotFoundError, IndexError) as e:
                print(f"[WARN] No valid frame image in {gen_frame_path}: {e}")
    print("Finishing!")