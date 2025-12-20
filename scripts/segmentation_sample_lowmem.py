import argparse
import os
import sys
import gc
sys.path.append(".")

import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset3D
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from collections import OrderedDict

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)
    
    # Memory optimization settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    
    # Clear initial cache
    if th.cuda.is_available():
        th.cuda.empty_cache()
        gc.collect()
    
    logger.log("Creating data loader...")
    tran_list = [transforms.Resize((args.image_size, args.image_size)),]
    transform_test = transforms.Compose(tran_list)
    ds = BRATSDataset3D(args.data_dir, transform_test, test_flag=True)
    args.in_ch = 5
    
    # Force batch size = 1 for low memory
    datal = th.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Load model weights
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict
            break
    
    model.load_state_dict(new_state_dict)
    model.to(dist_util.dev())
    
    if args.use_fp16:
        logger.log("Using FP16 precision...")
        model.convert_to_fp16()
    
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    logger.log(f"Processing {len(datal)} batches...")
    
    processed_count = 0
    for batch_idx, (batch, m, path) in enumerate(datal):
        try:
            # Extract data
            if isinstance(batch, (list, tuple)):
                b, b_2_5d = batch
            else:
                b = batch
                b_2_5d = None
            
            # Create noisy input
            c = th.randn_like(b[:, :1, ...])
            img = th.cat((b, c), dim=1)
            
            # Extract slice ID
            slice_ID = path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
            
            logger.log(f"Sampling batch {batch_idx+1}/{len(datal)} - {slice_ID}...")
            
            # Process with memory optimization
            with th.no_grad():
                enslist = []
                
                for i in range(args.num_ensemble):
                    model_kwargs = {}
                    if b_2_5d is not None:
                        model_kwargs["x_2_5d"] = b_2_5d.to(dist_util.dev())
                    
                    # Run sampling
                    sample_fn = diffusion.p_sample_loop_known
                    sample, x_noisy, org, cal, cal_out = sample_fn(
                        model,
                        (1, 3, args.image_size, args.image_size),
                        img,
                        step=args.diffusion_steps,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                    )
                    
                    co = th.tensor(cal_out)
                    enslist.append(co)
                    
                    # Clear cache after each ensemble iteration
                    del sample, x_noisy, org, cal, cal_out, co
                    if th.cuda.is_available():
                        th.cuda.empty_cache()
                
                # Compute ensemble result
                ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
                
                # Save result
                output_path = os.path.join(args.out_dir, f'{slice_ID}_output_ens.jpg')
                vutils.save_image(ensres, fp=output_path, nrow=1, padding=10)
                
                # Clean up
                del enslist, ensres, img, b, c
                if b_2_5d is not None:
                    del b_2_5d
            
            # Clear cache after each batch
            if th.cuda.is_available():
                th.cuda.empty_cache()
                gc.collect()
            
            processed_count += 1
            
            # Log memory usage every 10 batches
            if batch_idx % 10 == 0 and th.cuda.is_available():
                memory_allocated = th.cuda.memory_allocated() / 1024**3
                memory_reserved = th.cuda.memory_reserved() / 1024**3
                logger.log(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        except Exception as e:
            logger.log(f"Error processing batch {batch_idx}: {str(e)}")
            # Try to recover by clearing memory
            if th.cuda.is_available():
                th.cuda.empty_cache()
                gc.collect()
            continue
    
    logger.log(f"Sampling complete! Processed {processed_count}/{len(datal)} batches.")

def create_argparser():
    defaults = dict(
        data_name='BRATS3D',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,  # Fixed at 1
        use_ddim=False,
        model_path="",
        num_ensemble=1,  # Reduced for memory
        gpu_dev="0",
        out_dir='./results/',
        multi_gpu=None,
        debug=False,
        version='new'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
