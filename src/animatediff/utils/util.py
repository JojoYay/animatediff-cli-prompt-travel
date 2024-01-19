import os
from rembg import remove
from PIL import Image

import logging
from os import PathLike
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from einops import rearrange
from PIL import Image
from torch import Tensor
from torchvision.utils import save_image
#from tqdm.rich import tqdm
from tqdm import tqdm


logger = logging.getLogger(__name__)

def zero_rank_print(s):
    if not isinstance(s, str): s = repr(s)
    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def remove_background(input_path, output_path):
    with open(input_path, 'rb') as input_file:
        input_data = input_file.read()

    output_data = remove(input_data)

    with open(output_path, 'wb') as output_file:
        output_file.write(output_data)

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            remove_background(input_path, output_path)
            print(f'Processed: {filename}')

def save_frames(video: Tensor, frames_dir: PathLike, show_progress:bool=True):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = rearrange(video, "b c t h w -> t b c h w")
    if show_progress:
        for idx, frame in enumerate(tqdm(frames, desc=f"Saving frames to {frames_dir.stem}", leave=True)):
            save_image(frame, frames_dir.joinpath(f"{idx:08d}.png"))
    else:
        for idx, frame in enumerate(frames):
            save_image(frame, frames_dir.joinpath(f"{idx:08d}.png"))

#     input_dir = frames_dir.with_name(frames_dir.name + "_bg")    
#     input_dir.mkdir(parents=True, exist_ok=True)

#     frames_dir.mkdir(parents=True, exist_ok=True)
#     frames = rearrange(video, "b c t h w -> t b c h w")

#     if show_progress:
#         for idx, frame in enumerate(tqdm(frames, desc=f"Saving frames to {input_dir.stem}")):
#             save_image(frame, input_dir.joinpath(f"{idx:08d}.png"))
#     else:
#         for idx, frame in enumerate(frames):
#             save_image(frame, input_dir.joinpath(f"{idx:08d}.png"))

#     process_images(str(input_dir), str(frames_dir))

def save_imgs(imgs:List[Image.Image], frames_dir: PathLike):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(tqdm(imgs, desc=f"Saving frames to {frames_dir.stem}", leave=True)):
        img.save( frames_dir.joinpath(f"{idx:08d}.png") )

def save_video(video: Tensor, save_path: PathLike, fps: int = 8):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if video.ndim == 5:
        # batch, channels, frame, width, height -> frame, channels, width, height
        frames = video.permute(0, 2, 1, 3, 4).squeeze(0)
    elif video.ndim == 4:
        # channels, frame, width, height -> frame, channels, width, height
        frames = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"video must be 4 or 5 dimensional, got {video.ndim}")

    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    frames = frames.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        fp=save_path, format="GIF", append_images=images[1:], save_all=True, duration=(1 / fps * 1000), loop=0
    )


def path_from_cwd(path: PathLike) -> str:
    path = Path(path)
    return str(path.absolute().relative_to(Path.cwd()))


def resize_for_condition_image(input_image: Image, us_width: int, us_height: int):
    input_image = input_image.convert("RGB")
    H = int(round(us_height / 8.0)) * 8
    W = int(round(us_width / 8.0)) * 8
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def get_resized_images(org_images_path: List[str], us_width: int, us_height: int):

    images = [Image.open( p ) for p in org_images_path]

    W, H = images[0].size

    if us_width == -1:
        us_width = W/H * us_height
    elif us_height == -1:
        us_height = H/W * us_width

    return [resize_for_condition_image(img, us_width, us_height) for img in images]

def get_resized_image(org_image_path: str, us_width: int, us_height: int):

    image = Image.open( org_image_path )

    W, H = image.size

    if us_width == -1:
        us_width = W/H * us_height
    elif us_height == -1:
        us_height = H/W * us_width

    return resize_for_condition_image(image, us_width, us_height)

def get_resized_image2(org_image_path: str, size: int):

    image = Image.open( org_image_path )

    W, H = image.size

    if size < 0:
        return resize_for_condition_image(image, W, H)

    if W < H:
        us_width = size
        us_height = int(size * H/W)
    else:
        us_width = int(size * W/H)
        us_height = size

    return resize_for_condition_image(image, us_width, us_height)


def show_bytes(comment, obj):

    import sys
#    memory_size = sys.getsizeof(tensor) + torch.numel(tensor)*tensor.element_size()

    if torch.is_tensor(obj):
        logger.info(f"{comment} : {obj.dtype=}")

        cpu_mem = sys.getsizeof(obj)/1024/1024
        cpu_mem = 0 if cpu_mem < 1 else cpu_mem
        logger.info(f"{comment} : CPU {cpu_mem} MB")

        gpu_mem = torch.numel(obj)*obj.element_size()/1024/1024
        gpu_mem = 0 if gpu_mem < 1 else gpu_mem
        logger.info(f"{comment} : GPU {gpu_mem} MB")
    elif type(obj) is tuple:
        logger.info(f"{comment} : {type(obj)}")
        cpu_mem = 0
        gpu_mem = 0

        for o in obj:
            cpu_mem += sys.getsizeof(o)/1024/1024
            gpu_mem += torch.numel(o)*o.element_size()/1024/1024

        cpu_mem = 0 if cpu_mem < 1 else cpu_mem
        logger.info(f"{comment} : CPU {cpu_mem} MB")

        gpu_mem = 0 if gpu_mem < 1 else gpu_mem
        logger.info(f"{comment} : GPU {gpu_mem} MB")

    else:
        logger.info(f"{comment} : unknown type")



def show_gpu(comment=""):
    return
    import inspect
    callerframerecord = inspect.stack()[1]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)

    import time

    import GPUtil
    torch.cuda.synchronize()

#    time.sleep(1.5)

    #logger.info(comment)
    logger.info(f"{info.filename}/{info.lineno}/{comment}")
    GPUtil.showUtilization()


PROFILE_ON = False

def start_profile():
    if PROFILE_ON:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()
        return pr
    else:
        return None

def end_profile(pr, file_name):
    if PROFILE_ON:
        import io
        import pstats

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open(file_name, 'w+') as f:
            f.write(s.getvalue())

STOPWATCH_ON = False

time_record = []
start_time = 0

def stopwatch_start():
    global start_time,time_record
    import time

    if STOPWATCH_ON:
        time_record = []
        torch.cuda.synchronize()
        start_time = time.time()

def stopwatch_record(comment):
    import time

    if STOPWATCH_ON:
        torch.cuda.synchronize()
        time_record.append(((time.time() - start_time) , comment))

def stopwatch_stop(comment):

    if STOPWATCH_ON:
        stopwatch_record(comment)

        for rec in time_record:
            logger.info(rec)


def prepare_ip_adapter():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/ip_adapter/models/image_encoder", exist_ok=True)
    for hub_file in [
        "models/image_encoder/config.json",
        "models/image_encoder/pytorch_model.bin",
        "models/ip-adapter-plus_sd15.bin",
        "models/ip-adapter_sd15.bin",
        "models/ip-adapter_sd15_light.bin",
        "models/ip-adapter-plus-face_sd15.bin",
        "models/ip-adapter-full-face_sd15.bin",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/ip_adapter" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="h94/IP-Adapter", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/ip_adapter"
        )

def prepare_ip_adapter_sdxl():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/ip_adapter/sdxl_models/image_encoder", exist_ok=True)
    for hub_file in [
        "models/image_encoder/config.json",
        "models/image_encoder/pytorch_model.bin",
        "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
        "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
        "sdxl_models/ip-adapter_sdxl_vit-h.bin",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/ip_adapter" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="h94/IP-Adapter", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/ip_adapter"
        )


def prepare_lcm_lora():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/lcm_lora/sdxl", exist_ok=True)
    for hub_file in [
        "pytorch_lora_weights.safetensors",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/lcm_lora/sdxl" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="latent-consistency/lcm-lora-sdxl", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/lcm_lora/sdxl"
        )

    os.makedirs("data/models/lcm_lora/sd15", exist_ok=True)
    for hub_file in [
        "pytorch_lora_weights.safetensors",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/lcm_lora/sd15" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="latent-consistency/lcm-lora-sdv1-5", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/lcm_lora/sd15"
        )

def prepare_lllite():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/lllite", exist_ok=True)
    for hub_file in [
        "bdsqlsz_controlllite_xl_canny.safetensors",
        "bdsqlsz_controlllite_xl_depth.safetensors",
        "bdsqlsz_controlllite_xl_dw_openpose.safetensors",
        "bdsqlsz_controlllite_xl_lineart_anime_denoise.safetensors",
        "bdsqlsz_controlllite_xl_mlsd_V2.safetensors",
        "bdsqlsz_controlllite_xl_normal.safetensors",
        "bdsqlsz_controlllite_xl_recolor_luminance.safetensors",
        "bdsqlsz_controlllite_xl_segment_animeface_V2.safetensors",
        "bdsqlsz_controlllite_xl_sketch.safetensors",
        "bdsqlsz_controlllite_xl_softedge.safetensors",
        "bdsqlsz_controlllite_xl_t2i-adapter_color_shuffle.safetensors",
        "bdsqlsz_controlllite_xl_tile_anime_α.safetensors",        # alpha
        "bdsqlsz_controlllite_xl_tile_anime_β.safetensors",        # beta
    ]:
        path = Path(hub_file)

        saved_path = "data/models/lllite" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="bdsqlsz/qinglong_controlnet-lllite", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/lllite"
        )


def prepare_extra_controlnet():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/controlnet/animatediff_controlnet", exist_ok=True)
    for hub_file in [
        "controlnet_checkpoint.ckpt"
    ]:
        path = Path(hub_file)

        saved_path = "data/models/controlnet/animatediff_controlnet" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="crishhh/animatediff_controlnet", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/controlnet/animatediff_controlnet"
        )


def prepare_motion_module():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/motion-module", exist_ok=True)
    for hub_file in [
        "mm_sd_v15_v2.ckpt",
        "mm_sdxl_v10_beta.ckpt",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/motion-module" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="guoyww/animatediff", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/motion-module"
        )

def prepare_wd14tagger():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/WD14tagger", exist_ok=True)
    for hub_file in [
        "model.onnx",
        "selected_tags.csv",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/WD14tagger" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/WD14tagger"
        )

def prepare_dwpose():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/DWPose", exist_ok=True)
    for hub_file in [
        "dw-ll_ucoco_384.onnx",
        "yolox_l.onnx",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/DWPose" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="yzd-v/DWPose", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/DWPose"
        )



def prepare_softsplat():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/softsplat", exist_ok=True)
    for hub_file in [
        "softsplat-lf",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/softsplat" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="s9roll74/softsplat_mirror", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/softsplat"
        )


def extract_frames(movie_file_path, fps, out_dir, aspect_ratio, duration, offset, size_of_short_edge=-1, low_vram_mode=False):
    import ffmpeg
    print(f"movie_file_path: {movie_file_path}")
    probe = ffmpeg.probe(movie_file_path)
    video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video['width'])
    height = int(video['height'])

    node = ffmpeg.input( str(movie_file_path.resolve()) )

    node = node.filter( "fps", fps=fps )


    if duration > 0:
        node = node.trim(start=offset,end=offset+duration).setpts('PTS-STARTPTS')
    elif offset > 0:
        node = node.trim(start=offset).setpts('PTS-STARTPTS')

    if size_of_short_edge != -1:
        if width < height:
            r = height / width
            width = size_of_short_edge
            height = int( (size_of_short_edge * r)//8 * 8)
            node = node.filter('scale', size_of_short_edge, height)
        else:
            r = width / height
            height = size_of_short_edge
            width = int( (size_of_short_edge * r)//8 * 8)
            node = node.filter('scale', width, size_of_short_edge)

    if low_vram_mode:
        if aspect_ratio == -1:
            aspect_ratio = width/height
            logger.info(f"low {aspect_ratio=}")
            aspect_ratio = max(min( aspect_ratio, 1.5 ), 0.6666)
            logger.info(f"low {aspect_ratio=}")

    if aspect_ratio > 0:
        # aspect ratio (width / height)
        ww = round(height * aspect_ratio)
        if ww < width:
            x= (width - ww)//2
            y= 0
            w = ww
            h = height
        else:
            hh = round(width/aspect_ratio)
            x = 0
            y = (height - hh)//2
            w = width
            h = hh
        w = int(w // 8 * 8)
        h = int(h // 8 * 8)
        logger.info(f"crop to {w=},{h=}")
        node = node.crop(x, y, w, h)

    node = node.output( str(out_dir.resolve().joinpath("%08d.png")), start_number=0 )

    node.run(quiet=False, overwrite_output=True)






def is_v2_motion_module(motion_module_path:Path):
    if motion_module_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        loaded = load_file(motion_module_path, "cpu")
    else:
        from torch import load
        loaded = load(motion_module_path, "cpu")

    is_v2 = "mid_block.motion_modules.0.temporal_transformer.norm.bias" in loaded

    loaded = None
    torch.cuda.empty_cache()

    logger.info(f"{is_v2=}")

    return is_v2

def is_sdxl_checkpoint(checkpoint_path:Path):
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        loaded = load_file(checkpoint_path, "cpu")
    else:
        from torch import load
        loaded = load(checkpoint_path, "cpu")

    is_sdxl = False

    if "conditioner.embedders.1.model.ln_final.weight" in loaded:
        is_sdxl = True
    if "conditioner.embedders.0.model.ln_final.weight" in loaded:
        is_sdxl = True

    loaded = None
    torch.cuda.empty_cache()

    logger.info(f"{is_sdxl=}")
    return is_sdxl


tensor_interpolation = None

def get_tensor_interpolation_method():
    return tensor_interpolation

def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear

def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2

def slerp(
    v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        #logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()



def prepare_sam_hq(low_vram):
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/SAM", exist_ok=True)
    for hub_file in [
        "sam_hq_vit_h.pth" if not low_vram else "sam_hq_vit_b.pth"
    ]:
        path = Path(hub_file)

        saved_path = "data/models/SAM" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="lkeab/hq-sam", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/SAM"
        )

def prepare_groundingDINO():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/GroundingDINO", exist_ok=True)
    for hub_file in [
        "groundingdino_swinb_cogcoor.pth",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/GroundingDINO" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/GroundingDINO"
        )


def prepare_propainter():
    import os

    import git

    if os.path.isdir("src/animatediff/repo/ProPainter"):
        if os.listdir("src/animatediff/repo/ProPainter"):
            return

    repo = git.Repo.clone_from(url="https://github.com/sczhou/ProPainter", to_path="src/animatediff/repo/ProPainter", no_checkout=True )
    repo.git.checkout("a8a5827ca5e7e8c1b4c360ea77cbb2adb3c18370")


def prepare_anime_seg():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/anime_seg", exist_ok=True)
    for hub_file in [
        "isnetis.onnx",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/anime_seg" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="skytnt/anime-seg", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/anime_seg"
        )
