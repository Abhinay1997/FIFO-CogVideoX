import math
import os
import torch
import argparse
import torchvision

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.utils.torch_utils import randn_tensor
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

import os, sys
from tqdm import trange, tqdm

sys.path.append(os.path.split(os.path.split(sys.path[0])[0])[0])
from opensora_fifo.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.text_encoder import get_text_enc
from opensora.utils.utils import save_video_grid

sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_videogen import VideoGenPipeline

import imageio

def shift_latents(latents, scheduler):
    # shift latents
    latents[:,:,:-1] = latents[:,:,1:].clone()

    # add new noise to the last frame
    latents[:,:,-1] = torch.randn_like(latents[:,:,-1]) * scheduler.init_noise_sigma

    return latents



def main(args):
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir=args.cache_dir).to(device, dtype=torch.float16)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
    vae.vae_scale_factor = ae_stride_config[args.ae]
    # Load model:
    transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder=args.version, cache_dir=args.cache_dir, torch_dtype=torch.float16).to(device)
    transformer_model.force_images = args.force_images
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, torch_dtype=torch.float16).to(device)

    # video_length, image_size = transformer_model.config.video_length, int(args.version.split('x')[1])
    # latent_size = (image_size // ae_stride_config[args.ae][1], image_size // ae_stride_config[args.ae][2])
    # vae.latent_size = latent_size
    if args.force_images:
        video_length = 1
        ext = 'jpg'
    else:
        ext = 'mp4'

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    schedulers = None
    if args.sample_method == 'DDIM':  #########
        scheduler = DDIMScheduler()
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':  #############
        scheduler = DDPMScheduler()
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler()
        schedulers = [PNDMScheduler() for _ in range(args.video_length)]
        for s in schedulers:
            s.set_timesteps(args.num_sampling_steps, device=device)
    elif args.sample_method == 'HeunDiscrete':  ########
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == 'KDPM2AncestralDiscrete':  #########
        scheduler = KDPM2AncestralDiscreteScheduler()
    print('videogen_pipeline', device)
    videogen_pipeline = VideoGenPipeline(vae=vae,
                                         text_encoder=text_encoder,
                                         tokenizer=tokenizer,
                                         scheduler=scheduler,
                                         transformer=transformer_model).to(device=device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()

    # video_grids = []
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        args.text_prompt = [i.strip() for i in text_prompt]
    for prompt in args.text_prompt:
        print('Processing the ({}) prompt'.format(prompt))
        prompt_save = prompt.replace(' ', '_')[:100]

        if args.output_dir is None:
            output_dir = f"results/opensora_fifo/{prompt_save}"

            if args.new_video_length != 100:
                output_dir = output_dir.replace(f"{prompt_save}", f"{prompt_save}/{args.new_video_length}frames")
            if not args.lookahead_denoising:
                output_dir = output_dir.replace(f"{prompt_save}", f"{prompt_save}/no_lookahead_denoising")
            if not args.num_partitions != 8:
                output_dir = output_dir.replace(f"{prompt_save}", f"{prompt_save}/{args.num_partitions}partitions")

            if args.version == "221x512x512":
                output_dir = output_dir.replace("opensora_fifo", "opensora_fifo_221")
        else:
            output_dir = args.output_dir

        print("The results will be saved in", output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        videogen_pipeline.scheduler.set_timesteps(args.num_sampling_steps, device=videogen_pipeline.text_encoder.device)
    

        if args.save_frames:
            fifo_dir = os.path.join(output_dir, "fifo")
            os.makedirs(fifo_dir, exist_ok=True)
        
        fifo_video_frames = []
        fifo_first_latents = []

        timesteps = videogen_pipeline.scheduler.timesteps
        timesteps = torch.flip(timesteps, [0])
        if args.lookahead_denoising:
            timesteps = torch.cat([torch.full((args.video_length//2,), timesteps[0]).to(timesteps.device), timesteps])
        
        max_queue_length = len(timesteps)
        bsz = 1
        ch = videogen_pipeline.transformer.config.in_channels
        h = args.height // videogen_pipeline.vae.vae_scale_factor[1]
        w = args.width // videogen_pipeline.vae.vae_scale_factor[2]
        init_noise_shape = [bsz, ch, max_queue_length, h, w]
        
        latents = randn_tensor(init_noise_shape, device=videogen_pipeline.text_encoder.device, dtype=torch.float16)
        
        num_vae = (args.new_video_length - 1) // (args.video_length-1)
        if (args.new_video_length - 1) % (args.video_length-1) != 0:
            num_vae += 1
        
        num_iterations = num_vae * (args.video_length-1) + 1 + args.queue_length
        for i in trange(num_iterations):
            num_inference_steps_per_gpu = args.video_length
            curr_timesteps = timesteps.clone()

            if i < max_queue_length-1:
                curr_timesteps[:-i-1] = curr_timesteps[-i-1]

            for rank in reversed(range(2 * args.num_partitions if args.lookahead_denoising else args.num_partitions)):
                if args.lookahead_denoising:
                    start_idx = (rank // 2) * num_inference_steps_per_gpu + (rank % 2) * (num_inference_steps_per_gpu // 2)
                else:
                    start_idx = rank * num_inference_steps_per_gpu
                midpoint_idx = start_idx + num_inference_steps_per_gpu // 2 + (rank % 2)
                end_idx = start_idx + num_inference_steps_per_gpu

                t = curr_timesteps[start_idx:end_idx]
                input_latents = latents[:,:,start_idx:end_idx].clone()

                output_latents, first_latent, first_frame = videogen_pipeline.fifo_onestep(prompt,
                                        video_length=args.video_length,
                                        height=args.height,
                                        width=args.width,
                                        num_inference_steps=args.num_sampling_steps,
                                        guidance_scale=args.guidance_scale,
                                        enable_temporal_attentions=not args.force_images,
                                        num_images_per_prompt=1,
                                        mask_feature=True,
                                        latents=input_latents,
                                        timesteps=t,
                                        save_frames=args.save_frames,
                                        )

                if args.lookahead_denoising:
                    latents[:,:,midpoint_idx:end_idx] = output_latents[:,:,-(end_idx-midpoint_idx):]
                else:
                    latents[:,:,start_idx:end_idx] = output_latents
                del output_latents

            latents = shift_latents(latents, videogen_pipeline.scheduler)
            
            

            if i >= max_queue_length:
                if args.save_frames:
                    output_path = os.path.join(fifo_dir, f"frame_{i:04d}.png")
                    imageio.mimwrite(output_path, first_frame, quality=9)  # highest quality is 10, lowest is 0
                fifo_first_latents.append(first_latent)
        

        fifo_vae_video_frames = []
        for i in range(num_vae):
            target_latents = torch.cat(fifo_first_latents[i*(args.video_length-1):(i+1)*(args.video_length-1)+1], dim=2)
            video = videogen_pipeline.decode_latents(target_latents)[0]

            if i == 0:
                fifo_vae_video_frames.append(video)
            else:
                fifo_vae_video_frames.append(video[1:])
        
        if num_vae > 0:
            fifo_vae_video_frames = torch.cat(fifo_vae_video_frames, dim=0)
            if args.output_dir is None:
                output_vae_path = os.path.join(output_dir, "fifo_vae.mp4")
            else:
                output_vae_path = os.path.join(args.output_dir, f"{prompt_save}.mp4")
            imageio.mimwrite(output_vae_path, fifo_vae_video_frames, fps=args.fps, quality=9)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.1.0')
    parser.add_argument("--version", type=str, default=None, choices=[None, '65x512x512', '221x512x512', '513x512x512'])
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="DDPM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--queue_length", type=int, default=17)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--force_images', action='store_true')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--video_length", "-f", type=int, default=17)
    parser.add_argument("--new_video_length", "-N", type=int, default=None)
    parser.add_argument("--num_partitions", "-n", type=int, default=4)
    parser.add_argument("--lookahead_denoising", "-ld", action='store_false', default=True)
    parser.add_argument("--output_dir", type=str, default=None, help="custom output directory")
    parser.add_argument("--save_frames", action='store_true', default=False)

    args = parser.parse_args()

    assert args.num_frames == 4*args.video_length - 3

    args.queue_length = args.video_length * args.num_partitions
    args.num_sampling_steps = args.video_length * args.num_partitions

    main(args)