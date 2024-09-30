### IMPORTS

import torch
import argparse
from pipeline_cogvideox import CogVideoXPipeline, retrieve_timesteps
from transformer_cogvideox_v2 import CogVideoXTransformer3DModel
from sampler_cogvideox import CogVideoXDDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

from tqdm import trange, tqdm
from torchvision.utils import save_image


checkpoint = "THUDM/CogVideoX-2b"
dtype = torch.bfloat16

### INIT
transformer = CogVideoXTransformer3DModel.from_pretrained(
    checkpoint,
    torch_dtype=dtype,
    subfolder="transformer"
)
scheduler = CogVideoXDDIMScheduler.from_pretrained(
    checkpoint,
    torch_dtype=dtype,
    subfolder="scheduler"
)
pipe = CogVideoXPipeline.from_pretrained(
    checkpoint,
    torch_dtype=dtype,
    transformer=transformer,
    scheduler=scheduler
)
pipe.to("cuda")
pipe.vae.enable_tiling()
print("model loading")
### FIFO Pipeline

def shift_latents(latents, scheduler):
    # shift latents
    latents[:,:,:-1] = latents[:,:,1:].clone()

    # add new noise to the last frame
    latents[:,:,-1] = torch.randn_like(latents[:,:,-1]) * scheduler.init_noise_sigma

    return latents


@torch.no_grad()
def main(args):
    # args.video_length a.k.a f = 16 or 48 need to check if to use video frames or video latents
    # args.height
    # args.width
    # args.new_video_length #total num of frames in final video
    # args.video_length #num of frames per iteration to be processed/denoised a.k.a partition size
    # args.num_partitions
    # args.queue_length = args.video_length * args.num_partitions
    # args.num_sampling_steps = args.video_length * args.num_partitions
    # args.lookahead_denoising
    prompt = "An astronaut floating in space with the earth in the background."
    batch_size = 1
    num_videos_per_prompt = 1
    generator = torch.Generator('cuda').manual_seed(0)
    fifo_video_frames = []
    fifo_first_latents = []

    args.queue_length = (args.video_length - 1) // pipe.vae_scale_factor_temporal + 1 
    args.queue_length = args.queue_length * args.num_partitions

    args.num_sampling_steps = (args.video_length - 1) // pipe.vae_scale_factor_temporal + 1 
    args.num_sampling_steps = args.num_sampling_steps * args.num_partitions
    
    new_video_length = (args.new_video_length - 1) // pipe.vae_scale_factor_temporal + 1
    video_length = (args.video_length - 1) // pipe.vae_scale_factor_temporal + 1
    
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, args.num_sampling_steps, pipe._execution_device, None)
    
    # timesteps = pipe.scheduler.timesteps
    timesteps = torch.flip(timesteps, [0])
    if args.lookahead_denoising:
        timesteps = torch.cat([torch.full((video_length//2,), timesteps[0]).to(timesteps.device), timesteps])

    ## check this again
    max_queue_length = len(timesteps)
    print("max queue length", max_queue_length)

    # latents = pipe.prepare_latents(
    #         batch_size = batch_size * num_videos_per_prompt,
    #         num_channels_latents = pipe.transformer.config.in_channels,
    #         num_frames = max_queue_length * pipe.vae_scale_factor_temporal,
    #         height=args.height,
    #         width=args.width,
    #         dtype=dtype,
    #         device=pipe._execution_device,
    #         generator=generator
    #     )
    latents = randn_tensor(
        (batch_size * num_videos_per_prompt,
         max_queue_length,
         pipe.transformer.config.in_channels,
         args.height // pipe.vae_scale_factor_spatial,
         args.width // pipe.vae_scale_factor_spatial),
         generator=generator, device=pipe._execution_device, dtype=dtype
    )
    
    num_vae = (new_video_length - 1) // (video_length-1)
    if (new_video_length - 1) % (video_length-1) != 0:
        num_vae += 1
    
    num_iterations = num_vae * (video_length-1) + 1 + args.queue_length
    for i in trange(num_iterations):
        num_inference_steps_per_gpu = video_length
        curr_timesteps = timesteps.clone()

        if i < max_queue_length-1:
            print(curr_timesteps)
            curr_timesteps[:-i-1] = curr_timesteps[-i-1]
            print(curr_timesteps)
            
        for rank in reversed(range(2 * args.num_partitions if args.lookahead_denoising else args.num_partitions)):
            if args.lookahead_denoising:
                start_idx = (rank // 2) * num_inference_steps_per_gpu + (rank % 2) * (num_inference_steps_per_gpu // 2)
            else:
                start_idx = rank * num_inference_steps_per_gpu

            midpoint_idx = start_idx + num_inference_steps_per_gpu // 2 + (rank % 2)
            end_idx = start_idx + num_inference_steps_per_gpu
            
            t = curr_timesteps[start_idx:end_idx]
            print("timesteps shape", t.shape)
            input_latents = latents[:,start_idx:end_idx,:].clone()
            print("iter", i, start_idx, end_idx, input_latents.shape, latents.shape)
            print(t)
            output_latents, first_latent, first_frame = pipe(prompt,
                                    num_frames=args.video_length,
                                    height=args.height,
                                    width=args.width,
                                    num_inference_steps=args.num_sampling_steps,
                                    guidance_scale=args.guidance_scale,
                                    use_dynamic_cfg=False,
                                    latents=input_latents,
                                    timesteps=t,
                                    max_sequence_length=226,
                                    rank=rank
                                    )
            if first_latent is not None:
                print('first_latent', first_latent.shape)

            if args.lookahead_denoising:
                latents[:,midpoint_idx:end_idx,:] = output_latents[:,-(end_idx-midpoint_idx):,:]
            else:
                latents[:,start_idx:end_idx,:] = output_latents
            del output_latents

        latents = shift_latents(latents, pipe.scheduler)

        if i >= max_queue_length:
            if args.save_frames:
                output_path = os.path.join(fifo_dir, f"frame_{i:04d}.png")
                imageio.mimwrite(output_path, first_frame, quality=9)  # highest quality is 10, lowest is 0
            fifo_first_latents.append(first_latent)
            # video = pipe.decode_latents(torch.cat([first_latent] * 2, dim=1))
            # video = pipe.video_processor.postprocess_video(video=video, output_type='pil')
            # for j,frame in enumerate(video[0]):
            #     print(frame)
            #     frame.save(f'image_{i}{j}.png')
        fifo_vae_video_frames = []
        
    for i in range(num_vae):
        target_latents = torch.cat(fifo_first_latents[i*(video_length-1):(i+1)*(video_length-1)+1], dim=1)
        video = pipe.decode_latents(target_latents)
        video = pipe.video_processor.postprocess_video(video=video, output_type='pil')
        for j,frame in enumerate(video[0]):
            print(frame)
            frame.save(f'image_{i}{j}.png')
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
    # parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    # parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    # parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    # parser.add_argument("--sample_method", type=str, default="DDPM")
    parser.add_argument("--num_sampling_steps", type=int, default=None)
    parser.add_argument("--queue_length", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    # parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    # parser.add_argument('--force_images', action='store_true')
    # parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    # parser.add_argument('--enable_tiling', action='store_true')
    # parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--video_length", "-f", type=int, default=48)
    parser.add_argument("--new_video_length", "-N", type=int, default=100)
    parser.add_argument("--num_partitions", "-n", type=int, default=4)
    parser.add_argument("--lookahead_denoising", "-ld", action='store_false', default=False)
    parser.add_argument("--output_dir", type=str, default=None, help="custom output directory")
    parser.add_argument("--save_frames", action='store_true', default=False)

    args = parser.parse_args()

    # assert args.num_frames == 4*args.video_length - 3

    args.queue_length = args.video_length * args.num_partitions
    args.num_sampling_steps = args.video_length * args.num_partitions

    args.queue_length = None
    args.num_sampling_steps = None

    main(args)