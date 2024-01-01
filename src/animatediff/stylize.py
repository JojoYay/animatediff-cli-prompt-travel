import glob
import json
import logging
import os.path
import shutil
import pytz
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from PIL import Image
#from tqdm.rich import tqdm
from tqdm import tqdm

from animatediff import __version__, get_dir
from animatediff.settings import ModelConfig, get_model_config
from animatediff.utils.tagger import get_labels
from animatediff.utils.util import (extract_frames, get_resized_image,
                                    path_from_cwd, prepare_anime_seg,
                                    prepare_groundingDINO, prepare_propainter,
                                    prepare_sam_hq, prepare_softsplat)
logger = logging.getLogger(__name__)



stylize: typer.Typer = typer.Typer(
    name="stylize",
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    help="stylize video",
)

data_dir = get_dir("data")

controlnet_dirs = [
    "controlnet_canny",
    "controlnet_depth",
    "controlnet_inpaint",
    "controlnet_ip2p",
    "controlnet_lineart",
    "controlnet_lineart_anime",
    "controlnet_mlsd",
    "controlnet_normalbae",
    "controlnet_openpose",
    "controlnet_scribble",
    "controlnet_seg",
    "controlnet_shuffle",
    "controlnet_softedge",
    "controlnet_tile",
    "qr_code_monster_v1",
    "qr_code_monster_v2",
    "controlnet_mediapipe_face",
    "animatediff_controlnet",
    ]

def create_controlnet_dir(controlnet_root):
    for c in controlnet_dirs:
        c_dir = controlnet_root.joinpath(c)
        c_dir.mkdir(parents=True, exist_ok=True)

@stylize.command(no_args_is_help=True)
def create_config(
    org_movie: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=True, dir_okay=False, exists=True, help="Path to movie file"),
    ] = ...,
    config_org: Annotated[
        Path,
        typer.Option(
            "--config-org",
            "-c",
            path_type=Path,
            dir_okay=False,
            exists=True,
            help="Path to original config file",
        ),
    ] = Path("config/fix/real_base2.json"),
    ignore_list: Annotated[
        Path,
        typer.Option(
            "--ignore-list",
            "-g",
            path_type=Path,
            dir_okay=False,
            exists=True,
            help="path to ignore token list file",
        ),
    ] = Path("config/prompts/ignore_tokens.txt"),
    out_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--out-dir",
            "-o",
            path_type=Path,
            file_okay=False,
            help="output directory",
        ),
    ] = Path("stylize/"),
    fps: Annotated[
        int,
        typer.Option(
            "--fps",
            "-f",
            min=1,
            max=120,
            help="fps",
        ),
    ] = 8,
    duration: Annotated[
        int,
        typer.Option(
            "--duration",
            "-d",
            min=-1,
            max=3600,
            help="Video duration in seconds. -1 means that the duration of the input video is used as is",
        ),
    ] = -1,
    offset: Annotated[
        int,
        typer.Option(
            "--offset",
            "-of",
            min=0,
            max=3600,
            help="offset in seconds. '-d 30 -of 1200' means to use 1200-1230 seconds of the input video",
        ),
    ] = 0,
    aspect_ratio: Annotated[
        float,
        typer.Option(
            "--aspect-ratio",
            "-a",
            min=-1,
            max=5.0,
            help="aspect ratio (width / height). (ex. 512 / 512 = 1.0 , 512 / 768 = 0.6666 , 768 / 512 = 1.5) -1 means that the aspect ratio of the input video is used as is.",
        ),
    ] = -1,
    size_of_short_edge: Annotated[
        int,
        typer.Option(
            "--short-edge",
            "-sh",
            min=100,
            max=1024,
            help="size of short edge",
        ),
    ] = 512,
    predicte_interval: Annotated[
        int,
        typer.Option(
            "--predicte-interval",
            "-p",
            min=1,
            max=120,
            help="Interval of frames to be predicted",
        ),
    ] = 1,
    general_threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-th",
            min=0.0,
            max=1.0,
            help="threshold for general token confidence",
        ),
    ] = 0.35,
    character_threshold: Annotated[
        float,
        typer.Option(
            "--threshold2",
            "-th2",
            min=0.0,
            max=1.0,
            help="threshold for character token confidence",
        ),
    ] = 0.85,
    without_confidence: Annotated[
        bool,
        typer.Option(
            "--no-confidence-format",
            "-ncf",
            is_flag=True,
            help="confidence token format or not. ex. '(close-up:0.57), (monochrome:1.1)' -> 'close-up, monochrome'",
        ),
    ] = False,
    is_no_danbooru_format: Annotated[
        bool,
        typer.Option(
            "--no-danbooru-format",
            "-ndf",
            is_flag=True,
            help="danbooru token format or not. ex. 'bandaid_on_leg, short_hair' -> 'bandaid on leg, short hair'",
        ),
#    ] = False,
    ] = True,
    is_img2img: Annotated[
        bool,
        typer.Option(
            "--img2img",
            "-i2i",
            is_flag=True,
            help="img2img or not(txt2img).",
        ),
    ] = False,
    low_vram: Annotated[
        bool,
        typer.Option(
            "--low_vram",
            "-lo",
            is_flag=True,
            help="low vram mode",
        ),
    ] = False,
    gradual_latent_hires_fix: Annotated[
        bool,
        typer.Option(
            "--gradual_latent_hires_fix",
            "-gh",
            is_flag=True,
            help="gradual latent hires fix",
        ),
    ] = False,
):
    """Create a config file for video stylization"""
    is_danbooru_format = not is_no_danbooru_format
    with_confidence = not without_confidence
    logger.info(f"{org_movie=}")
    logger.info(f"{config_org=}")
    logger.info(f"{ignore_list=}")
    logger.info(f"{out_dir=}")
    logger.info(f"{fps=}")
    logger.info(f"{duration=}")
    logger.info(f"{offset=}")
    logger.info(f"{aspect_ratio=}")
    logger.info(f"{size_of_short_edge=}")
    logger.info(f"{predicte_interval=}")
    logger.info(f"{general_threshold=}")
    logger.info(f"{character_threshold=}")
    logger.info(f"{with_confidence=}")
    logger.info(f"{is_danbooru_format=}")
    logger.info(f"{is_img2img=}")
    logger.info(f"{low_vram=}")
    logger.info(f"{gradual_latent_hires_fix=}")

    model_config: ModelConfig = get_model_config(config_org)

    # get a timestamp for the output directory
#    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
#    time_str = "jjj"
    time_str = model_config.name
    # make the output directory
#    save_dir = out_dir.joinpath(f"{time_str}-{model_config.save_name}")
    org_movie = Path(org_movie)
#    save_dir = out_dir.joinpath(f"{time_str}-{org_movie.stem}")
    save_dir = out_dir.joinpath(f"{org_movie.stem}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Will save outputs to ./{path_from_cwd(save_dir)}")

    img2img_dir = save_dir.joinpath("00_img2img")
    img2img_dir.mkdir(parents=True, exist_ok=True)
    extract_frames(org_movie, fps, img2img_dir, aspect_ratio, duration, offset, size_of_short_edge, low_vram)

    controlnet_img_dir = save_dir.joinpath("00_controlnet_image")

    create_controlnet_dir(controlnet_img_dir)

#    if not is_img2img:
    shutil.copytree(img2img_dir, controlnet_img_dir.joinpath("controlnet_tile"), dirs_exist_ok=True)
#    else:
    shutil.copytree(img2img_dir, controlnet_img_dir.joinpath("controlnet_openpose"), dirs_exist_ok=True)
    shutil.copytree(img2img_dir, controlnet_img_dir.joinpath("controlnet_lineart_anime"), dirs_exist_ok=True)
    shutil.copytree(img2img_dir, controlnet_img_dir.joinpath("controlnet_lineart"), dirs_exist_ok=True)
    shutil.copytree(img2img_dir, controlnet_img_dir.joinpath("controlnet_depth"), dirs_exist_ok=True)
    shutil.copytree(img2img_dir, controlnet_img_dir.joinpath("controlnet_scribble"), dirs_exist_ok=True)
    shutil.copytree(img2img_dir, controlnet_img_dir.joinpath("controlnet_normalbae"), dirs_exist_ok=True)
    shutil.copytree(img2img_dir, controlnet_img_dir.joinpath("animatediff_controlnet"), dirs_exist_ok=True)
    shutil.copytree(img2img_dir, controlnet_img_dir.joinpath("controlnet_mediapipe_face"), dirs_exist_ok=True)
    
#    shutil.copytree(img2img_dir, controlnet_img_dir.joinpath("controlnet_ip2p"), dirs_exist_ok=True)

    black_list = []
    if ignore_list.is_file():
        with open(ignore_list) as f:
            black_list = [s.strip() for s in f.readlines()]

#    model_config.prompt_map = get_labels(
#        frame_dir=img2img_dir,
#        interval=predicte_interval,
#        general_threshold=general_threshold,
#        character_threshold=character_threshold,
#        ignore_tokens=black_list,
#        with_confidence=with_confidence,
#        is_danbooru_format=is_danbooru_format,
#        is_cpu = False,
#    )

    model_config.prompt_map = {
                "0": "best quality"
    }

#    model_config.head_prompt = ""
#    model_config.tail_prompt = ""
    model_config.controlnet_map["input_image_dir"] = os.path.relpath(controlnet_img_dir.absolute(), data_dir)
    model_config.controlnet_map["is_loop"] = False

#    model_config.lora_map={}
#    model_config.motion_lora_map={}

    model_config.controlnet_map["max_samples_on_vram"] = 0
    model_config.controlnet_map["max_models_on_vram"] = 0


#    if not is_img2img:
#        model_config.controlnet_map["controlnet_tile"] = {
#        "enable": True,
#        "use_preprocessor":True,
#        "guess_mode":False,
#        "controlnet_conditioning_scale": 1.0,
#        "control_guidance_start": 0.0,
#        "control_guidance_end": 1.0,
#        "control_scale_list":[]
#        }
#    else:
    model_config.controlnet_map["controlnet_openpose"] = {
        "enable": True,
        "use_preprocessor":True,
        "guess_mode":False,
        "controlnet_conditioning_scale": 1.0,
        "control_guidance_start": 0.0,
        "control_guidance_end": 1.0,
        "control_scale_list":[],
        "control_region_list":[]
    }


    model_config.controlnet_map["controlnet_ip2p"] = {
#      "enable": True,
      "enable": False,
      "use_preprocessor":True,
      "guess_mode":False,
      "controlnet_conditioning_scale": 0.5,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list":[],
      "control_region_list":[]
    }

    for m in model_config.controlnet_map:
        if isinstance(model_config.controlnet_map[m] ,dict):
            if "control_scale_list" in model_config.controlnet_map[m]:
                model_config.controlnet_map[m]["control_scale_list"]=[]

    ip_adapter_dir = save_dir.joinpath("00_ipadapter")
    ip_adapter_dir.mkdir(parents=True, exist_ok=True)

    model_config.ip_adapter_map = {
#        "enable": True,
        "enable": False,
        "input_image_dir": os.path.relpath(ip_adapter_dir.absolute(), data_dir),
        "prompt_fixed_ratio": 0.5,
        "save_input_image": True,
        "resized_to_square": False,
        "scale": 0.5,
        "is_full_face": False,
        "is_plus_face": False,
        "is_plus": True,
        "is_light": False
    }

    model_config.img2img_map = {
#        "enable": is_img2img,
        "enable": False,
        "init_img_dir" : os.path.relpath(img2img_dir.absolute(), data_dir),
        "save_init_image": False,
        "denoising_strength" : 0.7
    }

    model_config.region_map = {

    }

    model_config.gradual_latent_hires_fix_map = {
        "enable" : True,
        "scale" : {
            "0": 0.5,
            "0.7": 1.0
        },
        "reverse_steps": 5,
        "noise_add_count": 3
    }

    model_config.output = {
        "format" : "mp4",
        "fps" : fps,
        "encode_param":{
            "crf": 10
        }
    }

    img = Image.open( img2img_dir.joinpath("00000000.png") )
    W, H = img.size

    base_size = 768 if gradual_latent_hires_fix else 512

    if W < H:
        width = base_size
        height = int(base_size * H/W)
    else:
        width = int(base_size * W/H)
        height = base_size

    width = int(width//8*8)
    height = int(height//8*8)

    length = len(glob.glob( os.path.join(img2img_dir, "[0-9]*.png"), recursive=False))

    model_config.stylize_config={
        "original_video":{
            "path":org_movie,
            "aspect_ratio":aspect_ratio,
            "offset":offset,
        },
        "create_mask": [
            "person"
        ],
        "composite": {
            "fg_list": [
                {
                    "path" : " absolute path to frame dir ",
                    "mask_path" : " absolute path to mask dir (this is optional) ",
                    "mask_prompt" : "person"
                },
                {
                    "path" : " absolute path to frame dir ",
                    "mask_path" : " absolute path to mask dir (this is optional) ",
                    "mask_prompt" : "cat"
                },
            ],
            "bg_frame_dir": "Absolute path to the BG frame directory",
            "hint": ""
        },
        "0":{
            "width": width,
            "height": height,
            "length": length,
            "context": 16,
            "overlap": 16//4,
            "stride": 0,
#        },
#        "1":{
#            "steps": model_config.steps,
#            "guidance_scale": model_config.guidance_scale,
#            "width": int(width * 1.5 //8*8),
#            "height": int(height * 1.5 //8*8),
#            "length": length,
#            "context": 8,
#            "overlap": 8//4,
#            "stride": 0,
#            "controlnet_tile":{
#                "enable": True,
#                "use_preprocessor":True,
#                "guess_mode":False,
#                "controlnet_conditioning_scale": 1.0,
#                "control_guidance_start": 0.0,
#                "control_guidance_end": 1.0,
#                "control_scale_list":[]
#            },
#            "controlnet_ip2p": {
#                "enable": False,
#                "use_preprocessor":True,
#                "guess_mode":False,
#                "controlnet_conditioning_scale": 0.5,
#                "control_guidance_start": 0.0,
#                "control_guidance_end": 1.0,
#                "control_scale_list":[]
#            },
#            "ip_adapter": False,
#            "reference": False,
#            "img2img": False,
#            "interpolation_multiplier": 1
        }
    }

    if gradual_latent_hires_fix:
        model_config.stylize_config.pop("1")


    save_config_path = save_dir.joinpath("prompt.json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")

    logger.info(f"config = { save_config_path }")
    logger.info(f"stylize_dir = { save_dir }")
    logger.info("create_config completed")

    # logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # logger.info(f"Hint. Edit the config file before starting the generation")
    # logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # logger.info(f"1. Change 'path' and 'motion_module' as needed")
    # logger.info(f"2. Enter the 'head_prompt' or 'tail_prompt' with your preferred prompt, quality prompt, lora trigger word, or any other prompt you wish to add.")
    # logger.info(f"3. Change 'n_prompt' as needed")
    # logger.info(f"4. Add the lora you need to 'lora_map'")
    # logger.info(f"5. If you do not like the default settings, edit 'ip_adapter_map' or 'controlnet_map'. \nIf you want to change the controlnet type, you need to replace the input image.")
    # logger.info(f"6. Change 'stylize_config' as needed. By default, it is generated twice: once for normal generation and once for upscaling.\nIf you don't need upscaling, delete the whole '1'.")
    # logger.info(f"7. Change 'output' as needed. Changing the 'fps' at this timing is not recommended as it will change the playback speed.\nIf you want to change the fps, specify it with the create-config option")


@stylize.command(no_args_is_help=True)
def generate(
    stylize_dir: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=False, dir_okay=True, exists=True, help="Path to stylize dir"),
    ] = ...,
    length: Annotated[
        int,
        typer.Option(
            "--length",
            "-L",
            min=-1,
            max=9999,
            help="Number of frames to generate. -1 means that the value in the config file is referenced.",
            rich_help_panel="Generation",
        ),
    ] = -1,
    frame_offset: Annotated[
        int,
        typer.Option(
            "--frame-offset",
            "-FO",
            min=0,
            max=999999,
            help="Frame offset at generation.",
            rich_help_panel="Generation",
        ),
    ] = 0,
):
    """Run video stylization"""
    from animatediff.cli import generate
    
    singapore_timezone = pytz.timezone('Asia/Singapore')
    time_str = datetime.now(singapore_timezone).strftime("%Y-%m-%d_%H-%M")
#    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    stylize_dir = Path(stylize_dir)
    config_org = stylize_dir.joinpath("prompt.json")

    model_config: ModelConfig = get_model_config(config_org)
    if length == -1:
        length = model_config.stylize_config["0"]["length"]

    model_config.stylize_config["0"]["length"] = min(model_config.stylize_config["0"]["length"] - frame_offset, length)
    if "1" in model_config.stylize_config:
        model_config.stylize_config["1"]["length"] = min(model_config.stylize_config["1"]["length"] - frame_offset, length)

    if frame_offset > 0:
        #controlnet
        org_controlnet_img_dir = data_dir.joinpath( model_config.controlnet_map["input_image_dir"] )
        new_controlnet_img_dir = org_controlnet_img_dir.parent / "00_tmp_controlnet_image"
        if new_controlnet_img_dir.is_dir():
            shutil.rmtree(new_controlnet_img_dir)
        new_controlnet_img_dir.mkdir(parents=True, exist_ok=True)

        for c in controlnet_dirs:
            src_dir = org_controlnet_img_dir.joinpath(c)
            dst_dir = new_controlnet_img_dir.joinpath(c)
            if src_dir.is_dir():
                dst_dir.mkdir(parents=True, exist_ok=True)

                frame_length = model_config.stylize_config["0"]["length"]

                src_imgs = sorted(glob.glob( os.path.join(src_dir, "[0-9]*.png"), recursive=False))
                for img in src_imgs:
                    n = int(Path(img).stem)
                    if n in range(frame_offset, frame_offset + frame_length):
                        dst_img_path = dst_dir.joinpath( f"{n-frame_offset:08d}.png" )
                        shutil.copy(img, dst_img_path)
        #img2img
        org_img2img_img_dir = data_dir.joinpath( model_config.img2img_map["init_img_dir"] )
        new_img2img_img_dir = org_img2img_img_dir.parent / "00_tmp_init_img_dir"
        if new_img2img_img_dir.is_dir():
            shutil.rmtree(new_img2img_img_dir)
        new_img2img_img_dir.mkdir(parents=True, exist_ok=True)

        src_dir = org_img2img_img_dir
        dst_dir = new_img2img_img_dir
        if src_dir.is_dir():
            dst_dir.mkdir(parents=True, exist_ok=True)

            frame_length = model_config.stylize_config["0"]["length"]

            src_imgs = sorted(glob.glob( os.path.join(src_dir, "[0-9]*.png"), recursive=False))
            for img in src_imgs:
                n = int(Path(img).stem)
                if n in range(frame_offset, frame_offset + frame_length):
                    dst_img_path = dst_dir.joinpath( f"{n-frame_offset:08d}.png" )
                    shutil.copy(img, dst_img_path)

        new_prompt_map = {}
        for p in model_config.prompt_map:
            n = int(p)
            if n in range(frame_offset, frame_offset + frame_length):
                new_prompt_map[str(n-frame_offset)]=model_config.prompt_map[p]

        model_config.prompt_map = new_prompt_map

        model_config.controlnet_map["input_image_dir"] = os.path.relpath(new_controlnet_img_dir.absolute(), data_dir)
        model_config.img2img_map["init_img_dir"] = os.path.relpath(new_img2img_img_dir.absolute(), data_dir)

        tmp_config_path = stylize_dir.joinpath("prompt_tmp.json")
        tmp_config_path.write_text(model_config.json(indent=4), encoding="utf-8")
        config_org = tmp_config_path

    print(f"length_generate:{length}")
    
    output_0_dir = generate(
        config_path=config_org,
        width=model_config.stylize_config["0"]["width"],
        height=model_config.stylize_config["0"]["height"],
        length=model_config.stylize_config["0"]["length"],
        context=model_config.stylize_config["0"]["context"],
        overlap=model_config.stylize_config["0"]["overlap"],
        stride=model_config.stylize_config["0"]["stride"],
        out_dir=stylize_dir
    )

    torch.cuda.empty_cache()

#    try:
#        # フォルダが存在する場合のみ削除
#        shutil.rmtree(output_0_dir.parent / f"{time_str}_{0:02d}")
#        print(f"Output Folder deleted successfully.")
#    except FileNotFoundError:
#        print(f"Output Folder does not exist.")
#    except Exception as e:
#        print(f"Error occurred while deleting Output folder: {e}")

    original_video_path = model_config.stylize_config.get("original_video", {}).get("path", "")
    # パスからファイル名部分を取得
#    video_name = os.path.basename(original_video_path)

    output_0_dir = output_0_dir.rename(output_0_dir.parent / f"{time_str}_{0:02d}")


    if "1" not in model_config.stylize_config:
        logger.info(f"Stylized results are output to {output_0_dir}")
        return

    logger.info(f"Intermediate files have been output to {output_0_dir}")

    output_0_img_dir = glob.glob( os.path.join(output_0_dir, "00-[0-9]*"), recursive=False)[0]

    interpolation_multiplier = 1
    if "interpolation_multiplier" in model_config.stylize_config["1"]:
        interpolation_multiplier = model_config.stylize_config["1"]["interpolation_multiplier"]

    if interpolation_multiplier > 1:
        from animatediff.rife.rife import rife_interpolate

        rife_img_dir = stylize_dir.joinpath(f"{1:02d}_rife_frame")
        if rife_img_dir.is_dir():
            shutil.rmtree(rife_img_dir)
        rife_img_dir.mkdir(parents=True, exist_ok=True)

        rife_interpolate(output_0_img_dir, rife_img_dir, interpolation_multiplier)
        model_config.stylize_config["1"]["length"] *= interpolation_multiplier

        if model_config.output:
            model_config.output["fps"] *= interpolation_multiplier
        if model_config.prompt_map:
            model_config.prompt_map = { str(int(i)*interpolation_multiplier): model_config.prompt_map[i] for i in model_config.prompt_map }

        output_0_img_dir = rife_img_dir


    controlnet_img_dir = stylize_dir.joinpath("01_controlnet_image")
    img2img_dir = stylize_dir.joinpath("01_img2img")
    img2img_dir.mkdir(parents=True, exist_ok=True)

    create_controlnet_dir(controlnet_img_dir)

    ip2p_for_upscale = model_config.stylize_config["1"]["controlnet_ip2p"]["enable"]
    ip_adapter_for_upscale = model_config.stylize_config["1"]["ip_adapter"]
    ref_for_upscale = model_config.stylize_config["1"]["reference"]

    shutil.copytree(output_0_img_dir, controlnet_img_dir.joinpath("controlnet_tile"), dirs_exist_ok=True)
    if ip2p_for_upscale:
        shutil.copytree(controlnet_img_dir.joinpath("controlnet_tile"), controlnet_img_dir.joinpath("controlnet_ip2p"), dirs_exist_ok=True)

    shutil.copytree(controlnet_img_dir.joinpath("controlnet_tile"), img2img_dir, dirs_exist_ok=True)

    model_config.controlnet_map["input_image_dir"] = os.path.relpath(controlnet_img_dir.absolute(), data_dir)

    model_config.controlnet_map["controlnet_tile"] = model_config.stylize_config["1"]["controlnet_tile"]
    model_config.controlnet_map["controlnet_ip2p"] = model_config.stylize_config["1"]["controlnet_ip2p"]

    if "controlnet_ref" in model_config.controlnet_map:
        model_config.controlnet_map["controlnet_ref"]["enable"] = ref_for_upscale

    model_config.ip_adapter_map["enable"] = ip_adapter_for_upscale
    for r in model_config.region_map:
        reg = model_config.region_map[r]
        if "condition" in reg:
            if "ip_adapter_map" in reg["condition"]:
                reg["condition"]["ip_adapter_map"]["enable"] = ip_adapter_for_upscale

    model_config.steps = model_config.stylize_config["1"]["steps"] if "steps" in model_config.stylize_config["1"] else model_config.steps
    model_config.guidance_scale = model_config.stylize_config["1"]["guidance_scale"] if "guidance_scale" in model_config.stylize_config["1"] else model_config.guidance_scale

    model_config.img2img_map["enable"] = model_config.stylize_config["1"]["img2img"]

    if model_config.img2img_map["enable"]:
        model_config.img2img_map["init_img_dir"] = os.path.relpath(Path(output_0_img_dir).absolute(), data_dir)

    save_config_path = stylize_dir.joinpath("prompt_01.json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")

    output_1_dir = generate(
        config_path=save_config_path,
        width=model_config.stylize_config["1"]["width"],
        height=model_config.stylize_config["1"]["height"],
        length=model_config.stylize_config["1"]["length"],
        context=model_config.stylize_config["1"]["context"],
        overlap=model_config.stylize_config["1"]["overlap"],
        stride=model_config.stylize_config["1"]["stride"],
        out_dir=stylize_dir
    )

 #   try:
        # フォルダが存在する場合のみ削除
 #       shutil.rmtree(output_0_dir.parent / f"{time_str}_{1:02d}")
 #       print(f"Output Folder deleted successfully.")
 #   except FileNotFoundError:
 #       print(f"Output Folder does not exist.")
 #   except Exception as e:
 #       print(f"Error occurred while deleting Output folder: {e}")

    output_1_dir = output_1_dir.rename(output_1_dir.parent / f"{time_str}_{1:02d}")

    logger.info(f"Stylized results are output to {output_1_dir}")
    logger.info(f"Stylize generate completed")



@stylize.command(no_args_is_help=True)
def interpolate(
    frame_dir: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=False, dir_okay=True, exists=True, help="Path to frame dir"),
    ] = ...,
    interpolation_multiplier: Annotated[
        int,
        typer.Option(
            "--interpolation_multiplier",
            "-m",
            min=1,
            max=10,
            help="interpolation_multiplier",
        ),
    ] = 1,
):
    """Interpolation with original frames. This function does not work well if the shape of the subject is changed from the original video. Large movements can also ruin the picture.(Since this command is experimental, it is better to use other interpolation methods in most cases.)"""

    try:
        import cupy
    except:
        logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.info(f"cupy is required to run interpolate")
        logger.info(f"Your CUDA version is {torch.version.cuda}")
        logger.info(f"Please find the installation method of cupy for your CUDA version from the following URL")
        logger.info(f"https://docs.cupy.dev/en/latest/install.html#installing-cupy-from-pypi")
        logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    prepare_softsplat()

#    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
#    time_str = "jjj"
    config_org = frame_dir.parent.joinpath("prompt.json")

    model_config: ModelConfig = get_model_config(config_org)
    time_str = model_config.name

    if "original_video" in model_config.stylize_config:
        org_video = Path(model_config.stylize_config["original_video"]["path"])
        offset = model_config.stylize_config["original_video"]["offset"]
        aspect_ratio = model_config.stylize_config["original_video"]["aspect_ratio"]
    else:
        logger.warn('!!! The following parameters are required !!!')
        logger.warn('"stylize_config": {')
        logger.warn('    "original_video": {')
        logger.warn('        "path": "C:\\my_movie\\test.mp4",')
        logger.warn('        "aspect_ratio": 0.6666,')
        logger.warn('        "offset": 0')
        logger.warn('    },')
        raise ValueError('model_config.stylize_config["original_video"] not found')


    save_dir = frame_dir.parent.joinpath(f"optflow_{time_str}")

    org_frame_dir = save_dir.joinpath("org_frame")
    org_frame_dir.mkdir(parents=True, exist_ok=True)

    stylize_frame = sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False))
    stylize_frame_num = len(stylize_frame)

    duration = int(stylize_frame_num / model_config.output["fps"]) + 1

    extract_frames(org_video, model_config.output["fps"] * interpolation_multiplier, org_frame_dir,aspect_ratio,duration,offset)

    W, H = Image.open(stylize_frame[0]).size

    org_frame = sorted(glob.glob( os.path.join(org_frame_dir, "[0-9]*.png"), recursive=False))

    for org in tqdm(org_frame, leave=True):
        img = get_resized_image(org, W, H)
        img.save(org)

    output_dir = save_dir.joinpath("warp_img")
    output_dir.mkdir(parents=True, exist_ok=True)

    from animatediff.softmax_splatting.run import estimate2

    for sty1,sty2 in tqdm(zip(stylize_frame,stylize_frame[1:]), total=len(stylize_frame[1:]), leave=True):
        sty1 = Path(sty1)
        sty2 = Path(sty2)

        head = int(sty1.stem)

        sty1_img = Image.open(sty1)
        sty2_img = Image.open(sty2)

        guide_frames=[org_frame_dir.joinpath(f"{g:08d}.png") for g in range(head*interpolation_multiplier, (head+1)*interpolation_multiplier)]

        guide_frames=[Image.open(g) for g in guide_frames]

        result = estimate2(sty1_img, sty2_img, guide_frames, "data/models/softsplat/softsplat-lf")

        shutil.copy( frame_dir.joinpath(f"{head:08d}.png"), output_dir.joinpath(f"{head*interpolation_multiplier:08d}.png"))

        offset = head*interpolation_multiplier + 1
        for i, r in enumerate(result):
            r.save( output_dir.joinpath(f"{offset+i:08d}.png") )


    from animatediff.generate import save_output


    frames = sorted(glob.glob( os.path.join(output_dir, "[0-9]*.png"), recursive=False))
    out_images = []
    for f in frames:
        out_images.append(Image.open(f))

    model_config.output["fps"] *= interpolation_multiplier

    out_file = save_dir.joinpath(f"01_{model_config.output['fps']}fps")
    save_output(out_images,output_dir,out_file,model_config.output,True,save_frames=None,save_video=None)

    out_file = save_dir.joinpath(f"00_original")
    save_output(out_images,org_frame_dir,out_file,model_config.output,True,save_frames=None,save_video=None)


@stylize.command(no_args_is_help=True)
def create_mask(
    stylize_dir: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=False, dir_okay=True, exists=True, help="Path to stylize dir"),
    ] = ...,
    frame_dir: Annotated[
        Path,
        typer.Option(
            "--frame_dir",
            "-f",
            path_type=Path,
            file_okay=False,
            help="Path to source frames directory. default is 'STYLIZE_DIR/00_img2img'",
        ),
    ] = None,
    bg_config: Annotated[
        Path,
        typer.Option(
            "--bg_config",
            "-bc",
            path_type=Path,
            file_okay=False,
            help="Path of bg prompt.json",
        ),
    ] = None,
    box_threshold: Annotated[
        float,
        typer.Option(
            "--box_threshold",
            "-b",
            min=0.0,
            max=1.0,
            help="box_threshold",
            rich_help_panel="create mask",
        ),
    ] = 0.3,
    text_threshold: Annotated[
        float,
        typer.Option(
            "--text_threshold",
            "-t",
            min=0.0,
            max=1.0,
            help="text_threshold",
            rich_help_panel="create mask",
        ),
    ] = 0.25,
    mask_padding: Annotated[
        int,
        typer.Option(
            "--mask_padding",
            "-mp",
            min=-100,
            max=100,
            help="padding pixel value",
            rich_help_panel="create mask",
        ),
    ] = 0,
    no_gb: Annotated[
        bool,
        typer.Option(
            "--no_gb",
            "-ng",
            is_flag=True,
            help="no green back",
            rich_help_panel="create mask",
        ),
    ] = False,
    no_crop: Annotated[
        bool,
        typer.Option(
            "--no_crop",
            "-nc",
            is_flag=True,
            help="no crop",
            rich_help_panel="create mask",
        ),
    ] = False,
    use_rembg: Annotated[
        bool,
        typer.Option(
            "--use_rembg",
            "-rem",
            is_flag=True,
            help="use [rembg] instead of [Sam+GroundingDINO]",
            rich_help_panel="create mask",
        ),
    ] = False,
    use_animeseg: Annotated[
        bool,
        typer.Option(
            "--use_animeseg",
            "-anim",
            is_flag=True,
            help="use [anime-segmentation] instead of [Sam+GroundingDINO]",
            rich_help_panel="create mask",
        ),
    ] = False,
    low_vram: Annotated[
        bool,
        typer.Option(
            "--low_vram",
            "-lo",
            is_flag=True,
            help="low vram mode",
            rich_help_panel="create mask/tag",
        ),
    ] = False,
    ignore_list: Annotated[
        Path,
        typer.Option(
            "--ignore-list",
            "-g",
            path_type=Path,
            dir_okay=False,
            exists=True,
            help="path to ignore token list file",
            rich_help_panel="create tag",
        ),
    ] = Path("config/prompts/ignore_tokens.txt"),
    predicte_interval: Annotated[
        int,
        typer.Option(
            "--predicte-interval",
            "-p",
            min=1,
            max=120,
            help="Interval of frames to be predicted",
            rich_help_panel="create tag",
        ),
    ] = 1,
    general_threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-th",
            min=0.0,
            max=1.0,
            help="threshold for general token confidence",
            rich_help_panel="create tag",
        ),
    ] = 0.35,
    character_threshold: Annotated[
        float,
        typer.Option(
            "--threshold2",
            "-th2",
            min=0.0,
            max=1.0,
            help="threshold for character token confidence",
            rich_help_panel="create tag",
        ),
    ] = 0.85,
    without_confidence: Annotated[
        bool,
        typer.Option(
            "--no-confidence-format",
            "-ncf",
            is_flag=True,
            help="confidence token format or not. ex. '(close-up:0.57), (monochrome:1.1)' -> 'close-up, monochrome'",
            rich_help_panel="create tag",
        ),
    ] = False,
    is_no_danbooru_format: Annotated[
        bool,
        typer.Option(
            "--no-danbooru-format",
            "-ndf",
            is_flag=True,
            help="danbooru token format or not. ex. 'bandaid_on_leg, short_hair' -> 'bandaid on leg, short hair'",
            rich_help_panel="create tag",
        ),
    ] = False,
):
    """Create mask from prompt"""
    from animatediff.utils.mask import (create_bg, create_fg, crop_frames,
                                        crop_mask_list, save_crop_info)
    from animatediff.utils.mask_animseg import animseg_create_fg
    from animatediff.utils.mask_rembg import rembg_create_fg
    import re
    
    print(f"no_crop:{no_crop}")
    
    is_danbooru_format = not is_no_danbooru_format
    with_confidence = not without_confidence

    if use_animeseg and use_rembg:
        raise ValueError("use_animeseg and use_rembg cannot be enabled at the same time")

    prepare_sam_hq(low_vram)
    prepare_groundingDINO()
    prepare_propainter()

    if use_animeseg:
        prepare_anime_seg()

#    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
#    time_str = "jjj"
    stylize_dir = Path(stylize_dir)
    config_org = stylize_dir.joinpath("prompt.json")

    model_config: ModelConfig = get_model_config(config_org)
    original_video_path = model_config.stylize_config["original_video"]["path"]
    match = re.search(r'(dance\d+)', original_video_path)
    video_name = match.group(1)
    time_str = video_name
    if bg_config is not None:
#    if not bg_config == 'NA':
        bg_model_config: ModelConfig = get_model_config(bg_config)

    if frame_dir is None:
        frame_dir = stylize_dir / "00_img2img"

    if not frame_dir.is_dir():
        raise ValueError(f'{frame_dir=} does not exist.')

    is_img2img = model_config.img2img_map["enable"] if "enable" in model_config.img2img_map else False


    create_mask_list = []
    if "create_mask" in model_config.stylize_config:
        create_mask_list = model_config.stylize_config["create_mask"]
    else:
        raise ValueError('model_config.stylize_config["create_mask"] not found')

    output_list = []

    stylize_frame = sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False))
    frame_len = len(stylize_frame)

    W, H = Image.open(stylize_frame[0]).size
    org_frame_size = (H,W)

    masked_area = [None for f in range(frame_len)]

    if use_rembg:
        create_mask_list = ["rembg"]
    elif use_animeseg:
        create_mask_list = ["anime-segmentation"]


    for i,mask_token in enumerate(create_mask_list):
        fg_dir = stylize_dir.joinpath(f"fg_{i:02d}_{time_str}")
        fg_dir.mkdir(parents=True, exist_ok=True)

        create_controlnet_dir( fg_dir / "00_controlnet_image" )

        fg_masked_dir = fg_dir / "00_img2img"
        fg_masked_dir.mkdir(parents=True, exist_ok=True)

        fg_mask_dir = fg_dir / "00_mask"
        fg_mask_dir.mkdir(parents=True, exist_ok=True)

        if use_animeseg:
            masked_area = animseg_create_fg(
                frame_dir=frame_dir,
                output_dir=fg_masked_dir,
                output_mask_dir=fg_mask_dir,
                masked_area_list=masked_area,
                mask_padding=mask_padding,
                bg_color=None if no_gb else (0,255,0),
            )
        elif use_rembg:
            masked_area = rembg_create_fg(
                frame_dir=frame_dir,
                output_dir=fg_masked_dir,
                output_mask_dir=fg_mask_dir,
                masked_area_list=masked_area,
                mask_padding=mask_padding,
                bg_color=None if no_gb else (0,255,0),
            )
        else:
            masked_area = create_fg(
                mask_token=mask_token,
                frame_dir=frame_dir,
                output_dir=fg_masked_dir,
                output_mask_dir=fg_mask_dir,
                masked_area_list=masked_area,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                mask_padding=mask_padding,
                sam_checkpoint= "data/models/SAM/sam_hq_vit_h.pth" if not low_vram else "data/models/SAM/sam_hq_vit_b.pth",
                bg_color=None if no_gb else (0,255,0),
            )

        if not no_crop:
            frame_size_hw = (masked_area[0].shape[1],masked_area[0].shape[2])
            cropped_mask_list, mask_pos_list, crop_size_hw = crop_mask_list(masked_area)

            logger.info(f"crop fg_masked_dir")
            crop_frames(mask_pos_list, crop_size_hw, fg_masked_dir)
            logger.info(f"crop fg_mask_dir")
            crop_frames(mask_pos_list, crop_size_hw, fg_mask_dir)
            save_crop_info(mask_pos_list, crop_size_hw, frame_size_hw, fg_dir / "crop_info.json")
        else:
            crop_size_hw = None
            logger.info(f"no crop fg_masked_dir!!!!!!!!!!!!!!!!!")

        logger.info(f"mask from [{mask_token}] are output to {fg_dir}")

#        if not is_img2img:
#            shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/controlnet_tile", dirs_exist_ok=True)
#        else:
#            shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/controlnet_openpose", dirs_exist_ok=True)

        shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/controlnet_ip2p", dirs_exist_ok=True)

        shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/controlnet_tile", dirs_exist_ok=True)
        shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/controlnet_openpose", dirs_exist_ok=True)
        shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/controlnet_lineart_anime", dirs_exist_ok=True)
        shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/controlnet_lineart", dirs_exist_ok=True)
        shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/controlnet_depth", dirs_exist_ok=True)
        shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/controlnet_scribble", dirs_exist_ok=True)
        shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/controlnet_normalbae", dirs_exist_ok=True)
        shutil.copytree(fg_masked_dir, fg_dir / "00_controlnet_image/animatediff_controlnet", dirs_exist_ok=True)

        if crop_size_hw:
            if crop_size_hw[0] == 0 or crop_size_hw[1] == 0:
                crop_size_hw = None

        output_list.append((fg_dir, crop_size_hw))

    torch.cuda.empty_cache()

    bg_dir = stylize_dir.joinpath(f"bg_{time_str}")
    bg_dir.mkdir(parents=True, exist_ok=True)
    create_controlnet_dir( bg_dir / "00_controlnet_image" )
    bg_inpaint_dir = bg_dir / "00_img2img"
    bg_inpaint_dir.mkdir(parents=True, exist_ok=True)


    create_bg(frame_dir, bg_inpaint_dir, masked_area,
              use_half = True,
              raft_iter = 20,
              subvideo_length=80 if not low_vram else 50,
              neighbor_length=10 if not low_vram else 8,
              ref_stride=10 if not low_vram else 8,
              low_vram = low_vram,
              )

    logger.info(f"background are output to {bg_dir}")

    shutil.copytree(bg_inpaint_dir, bg_dir / "00_controlnet_image/controlnet_tile", dirs_exist_ok=True)

    shutil.copytree(bg_inpaint_dir, bg_dir / "00_controlnet_image/controlnet_ip2p", dirs_exist_ok=True)

    output_list.append((bg_dir,None))

    torch.cuda.empty_cache()

    black_list = []
    if ignore_list.is_file():
        with open(ignore_list) as f:
            black_list = [s.strip() for s in f.readlines()]

    for output, size in output_list:

#        model_config.prompt_map = get_labels(
#            frame_dir= output / "00_img2img",
#            interval=predicte_interval,
#            general_threshold=general_threshold,
#            character_threshold=character_threshold,
#            ignore_tokens=black_list,
#            with_confidence=with_confidence,
#            is_danbooru_format=is_danbooru_format,
#            is_cpu = False,
#        )

        model_config.prompt_map = {
           "0": "best quality"
        }

        model_config.controlnet_map["input_image_dir"] = os.path.relpath((output / "00_controlnet_image" ).absolute(), data_dir)
        model_config.img2img_map["init_img_dir"] = os.path.relpath((output / "00_img2img" ).absolute(), data_dir)

        if size is not None:
            h, w = size
            height = 1024 * (h/(h+w))
            width = 1024 * (w/(h+w))
            height = int(height//8 * 8)
            width = int(width//8 * 8)

            model_config.stylize_config["0"]["width"]=width
            model_config.stylize_config["0"]["height"]=height
            if "1" in model_config.stylize_config:
                model_config.stylize_config["1"]["width"]=int(width * 1.25 //8*8)
                model_config.stylize_config["1"]["height"]=int(height * 1.25 //8*8)
        else:
            height, width = org_frame_size
            model_config.stylize_config["0"]["width"]=width
            model_config.stylize_config["0"]["height"]=height
            if "1" in model_config.stylize_config:
                model_config.stylize_config["1"]["width"]=int(width * 1.25 //8*8)
                model_config.stylize_config["1"]["height"]=int(height * 1.25 //8*8)

        save_config_path = output.joinpath("prompt.json")
#        if output == bg_dir and not bg_config == 'NA':
        if output == bg_dir and bg_config is not None:
            model_config.path = bg_model_config.path
            model_config.seed = bg_model_config.seed
            model_config.scheduler = bg_model_config.scheduler
            model_config.steps = bg_model_config.steps
            model_config.guidance_scale = bg_model_config.guidance_scale
            model_config.head_prompt = bg_model_config.head_prompt
            model_config.prompt_map = bg_model_config.prompt_map
            model_config.lora_map = bg_model_config.lora_map
            model_config.tail_prompt = bg_model_config.tail_prompt
            model_config.n_prompt = bg_model_config.n_prompt
            model_config.lora_map = bg_model_config.lora_map
            model_config.controlnet_map = bg_model_config.controlnet_map
            print("updated BG config")

        save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")
    logger.info(f"create mask completed")
        
        
@stylize.command(no_args_is_help=True)
def composite(
    stylize_dir: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=False, dir_okay=True, exists=True, help="Path to stylize dir"),
    ] = ...,
    box_threshold: Annotated[
        float,
        typer.Option(
            "--box_threshold",
            "-b",
            min=0.0,
            max=1.0,
            help="box_threshold",
            rich_help_panel="create mask",
        ),
    ] = 0.3,
    text_threshold: Annotated[
        float,
        typer.Option(
            "--text_threshold",
            "-t",
            min=0.0,
            max=1.0,
            help="text_threshold",
            rich_help_panel="create mask",
        ),
    ] = 0.25,
    mask_padding: Annotated[
        int,
        typer.Option(
            "--mask_padding",
            "-mp",
            min=-100,
            max=100,
            help="padding pixel value",
            rich_help_panel="create mask",
        ),
    ] = 0,
    use_rembg: Annotated[
        bool,
        typer.Option(
            "--use_rembg",
            "-rem",
            is_flag=True,
            help="use \[rembg] instead of \[Sam+GroundingDINO]",
            rich_help_panel="create mask",
        ),
    ] = False,
    use_animeseg: Annotated[
        bool,
        typer.Option(
            "--use_animeseg",
            "-anim",
            is_flag=True,
            help="use \[anime-segmentation] instead of \[Sam+GroundingDINO]",
            rich_help_panel="create mask",
        ),
    ] = False,
    low_vram: Annotated[
        bool,
        typer.Option(
            "--low_vram",
            "-lo",
            is_flag=True,
            help="low vram mode",
            rich_help_panel="create mask/tag",
        ),
    ] = False,
    is_simple_composite: Annotated[
        bool,
        typer.Option(
            "--simple_composite",
            "-si",
            is_flag=True,
            help="simple composite",
            rich_help_panel="composite",
        ),
    ] = False,
    bg_dir: Annotated[
        Path,
        typer.Option(
            "--bg_dir",
            "-bg",
            path_type=Path,
            file_okay=False, dir_okay=True, exists=True,
            help="Path to bg frames directory. default is 'STYLIZE_DIR/00_img2img'",
        ),
    ] = None,
    fg_dir: Annotated[
        Path,
        typer.Option(
            "--fg_dir",
            "-fg",
            path_type=Path,
            file_okay=False, dir_okay=True, exists=True,
            help="Path to fg frames directory. default is 'STYLIZE_DIR/fg_00_[model_name]/timestamp/00-timestamp'",
        ),
    ] = None,
) -> str:
    """composite FG and BG"""
    print("composite starts")
    from animatediff.utils.composite import composite, simple_composite
    from animatediff.utils.mask import (create_fg, load_frame_list,
                                        load_mask_list, restore_position)
    from animatediff.utils.mask_animseg import animseg_create_fg
    from animatediff.utils.mask_rembg import rembg_create_fg

    if use_animeseg and use_rembg:
        raise ValueError("use_animeseg and use_rembg cannot be enabled at the same time")

    prepare_sam_hq(low_vram)
    if use_animeseg:
        prepare_anime_seg()

#    time_str = "jjj"
    singapore_timezone = pytz.timezone('Asia/Singapore')
    time_str = datetime.now(singapore_timezone).strftime("%Y-%m-%d_%H-%M")

    config_org = stylize_dir.joinpath("prompt.json")

    model_config: ModelConfig = get_model_config(config_org)
 #   time_str = model_config.name


    composite_config = {}
    if "composite" in model_config.stylize_config:
        composite_config = model_config.stylize_config["composite"]
    else:
        raise ValueError('model_config.stylize_config["composite"] not found')

    save_dir = stylize_dir.joinpath(f"cp_{time_str}")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_config_path = save_dir.joinpath("prompt.json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")

    if not bg_dir:
        bg_dir = composite_config["bg_frame_dir"]
    bg_dir = Path(bg_dir)
    if not bg_dir.is_dir():
        raise ValueError('model_config.stylize_config["composite"]["bg_frame_dir"] not valid')

    frame_len = len(sorted(glob.glob( os.path.join(bg_dir, "[0-9]*.png"), recursive=False)))
    if fg_dir:
        fg_list = [{ "path": fg_dir,
                    "mask_path": " absolute path to mask dir (this is optional) ",
                    "mask_prompt": "person" }]
    else:
        fg_list = composite_config["fg_list"]

    for i, fg_param in enumerate(fg_list):
        mask_token = fg_param["mask_prompt"]
        frame_dir = Path(fg_param["path"])
        if not frame_dir.is_dir():
            logger.warn(f"{frame_dir=} not valid -> skip")
            continue

        mask_dir = Path(fg_param["mask_path"])
        if not mask_dir.is_dir():
            logger.info(f"{mask_dir=} not valid -> create mask")

            fg_tmp_dir = save_dir.joinpath(f"fg_{i:02d}_{time_str}")
            fg_tmp_dir.mkdir(parents=True, exist_ok=True)

            masked_area_list = [None for f in range(frame_len)]

            if use_animeseg:
                mask_list = animseg_create_fg(
                    frame_dir=frame_dir,
                    output_dir=fg_tmp_dir,
                    output_mask_dir=None,
                    masked_area_list=masked_area_list,
                    mask_padding=mask_padding,
                )
            elif use_rembg:
                mask_list = rembg_create_fg(
                    frame_dir=frame_dir,
                    output_dir=fg_tmp_dir,
                    output_mask_dir=None,
                    masked_area_list=masked_area_list,
                    mask_padding=mask_padding,
                )
            else:
                mask_list = create_fg(
                    mask_token=mask_token,
                    frame_dir=frame_dir,
                    output_dir=fg_tmp_dir,
                    output_mask_dir=None,
                    masked_area_list=masked_area_list,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    mask_padding=mask_padding,
                    sam_checkpoint= "data/models/SAM/sam_hq_vit_h.pth" if not low_vram else "data/models/SAM/sam_hq_vit_b.pth",
                )

        else:
            logger.info(f"use {mask_dir=} as mask")

            masked_area_list = [None for f in range(frame_len)]

            mask_list = load_mask_list(mask_dir, masked_area_list, mask_padding)

        mask_list = [ m.transpose([1,2,0]) if m is not None else m for m in mask_list]

        crop_info_path = frame_dir.parent.parent / "crop_info.json"
        crop_info={}
        if crop_info_path.is_file():
            with open(crop_info_path, mode="rt", encoding="utf-8") as f:
                crop_info = json.load(f)
            mask_list = restore_position(mask_list, crop_info)


        fg_list = [None for f in range(frame_len)]
        fg_list = load_frame_list(frame_dir, fg_list, crop_info)

        output_dir = save_dir.joinpath(f"bg_{i:02d}_{time_str}")
        output_dir.mkdir(parents=True, exist_ok=True)

  #      if is_simple_composite:
        print("do simple composite")
        simple_composite(bg_dir, fg_list, output_dir, mask_list)
 #       else:
#            print("std composite")
    #        composite(bg_dir, fg_list, output_dir, mask_list)

        bg_dir = output_dir

    from animatediff.generate import save_output

    frames = sorted(glob.glob( os.path.join(bg_dir, "[0-9]*.png"), recursive=False))
    out_images = []
    for f in frames:
        out_images.append(Image.open(f))

    singapore_timezone = pytz.timezone('Asia/Singapore')
    now = datetime.now(singapore_timezone).strftime("%Y-%m-%d_%H-%M-%S")
    out_file = save_dir.joinpath(f"composite{now}")
    save_output(out_images,bg_dir,out_file,model_config.output,True,save_frames=None,save_video=None)

    logger.info(f"output to {out_file}")
    return out_file

@stylize.command(no_args_is_help=True)
def create_region(
    stylize_dir: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=False, dir_okay=True, exists=True, help="Path to stylize dir"),
    ] = ...,
    frame_dir: Annotated[
        Path,
        typer.Option(
            "--frame_dir",
            "-f",
            path_type=Path,
            file_okay=False,
            help="Path to source frames directory. default is 'STYLIZE_DIR/00_img2img'",
        ),
    ] = None,
    box_threshold: Annotated[
        float,
        typer.Option(
            "--box_threshold",
            "-b",
            min=0.0,
            max=1.0,
            help="box_threshold",
            rich_help_panel="create mask",
        ),
    ] = 0.3,
    text_threshold: Annotated[
        float,
        typer.Option(
            "--text_threshold",
            "-t",
            min=0.0,
            max=1.0,
            help="text_threshold",
            rich_help_panel="create mask",
        ),
    ] = 0.25,
    mask_padding: Annotated[
        int,
        typer.Option(
            "--mask_padding",
            "-mp",
            min=-100,
            max=100,
            help="padding pixel value",
            rich_help_panel="create mask",
        ),
    ] = 0,
    use_rembg: Annotated[
        bool,
        typer.Option(
            "--use_rembg",
            "-rem",
            is_flag=True,
            help="use [rembg] instead of [Sam+GroundingDINO]",
            rich_help_panel="create mask",
        ),
    ] = False,
    use_animeseg: Annotated[
        bool,
        typer.Option(
            "--use_animeseg",
            "-anim",
            is_flag=True,
            help="use [anime-segmentation] instead of [Sam+GroundingDINO]",
            rich_help_panel="create mask",
        ),
    ] = False,
    low_vram: Annotated[
        bool,
        typer.Option(
            "--low_vram",
            "-lo",
            is_flag=True,
            help="low vram mode",
            rich_help_panel="create mask/tag",
        ),
    ] = False,
    ignore_list: Annotated[
        Path,
        typer.Option(
            "--ignore-list",
            "-g",
            path_type=Path,
            dir_okay=False,
            exists=True,
            help="path to ignore token list file",
            rich_help_panel="create tag",
        ),
    ] = Path("config/prompts/ignore_tokens.txt"),
    predicte_interval: Annotated[
        int,
        typer.Option(
            "--predicte-interval",
            "-p",
            min=1,
            max=120,
            help="Interval of frames to be predicted",
            rich_help_panel="create tag",
        ),
    ] = 1,
    general_threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-th",
            min=0.0,
            max=1.0,
            help="threshold for general token confidence",
            rich_help_panel="create tag",
        ),
    ] = 0.35,
    character_threshold: Annotated[
        float,
        typer.Option(
            "--threshold2",
            "-th2",
            min=0.0,
            max=1.0,
            help="threshold for character token confidence",
            rich_help_panel="create tag",
        ),
    ] = 0.85,
    without_confidence: Annotated[
        bool,
        typer.Option(
            "--no-confidence-format",
            "-ncf",
            is_flag=True,
            help="confidence token format or not. ex. '(close-up:0.57), (monochrome:1.1)' -> 'close-up, monochrome'",
            rich_help_panel="create tag",
        ),
    ] = False,
    is_no_danbooru_format: Annotated[
        bool,
        typer.Option(
            "--no-danbooru-format",
            "-ndf",
            is_flag=True,
            help="danbooru token format or not. ex. 'bandaid_on_leg, short_hair' -> 'bandaid on leg, short hair'",
            rich_help_panel="create tag",
        ),
    ] = False,
):
    """Create region from prompt"""
    from animatediff.utils.mask import create_bg, create_fg
    from animatediff.utils.mask_animseg import animseg_create_fg
    from animatediff.utils.mask_rembg import rembg_create_fg

    is_danbooru_format = not is_no_danbooru_format
    with_confidence = not without_confidence

    if use_animeseg and use_rembg:
        raise ValueError("use_animeseg and use_rembg cannot be enabled at the same time")

    prepare_sam_hq(low_vram)
    prepare_groundingDINO()
    prepare_propainter()

    if use_animeseg:
        prepare_anime_seg()

#    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
#    time_str = "jjj"
    config_org = stylize_dir.joinpath("prompt.json")

    model_config: ModelConfig = get_model_config(config_org)
    time_str = model_config.name

    if frame_dir is None:
        frame_dir = stylize_dir / "00_img2img"

    if not frame_dir.is_dir():
        raise ValueError(f'{frame_dir=} does not exist.')


    create_mask_list = []
    if "create_mask" in model_config.stylize_config:
        create_mask_list = model_config.stylize_config["create_mask"]
    else:
        raise ValueError('model_config.stylize_config["create_mask"] not found')

    output_list = []

    stylize_frame = sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False))
    frame_len = len(stylize_frame)

    masked_area = [None for f in range(frame_len)]

    if use_rembg:
        create_mask_list = ["rembg"]
    elif use_animeseg:
        create_mask_list = ["anime-segmentation"]


    for i,mask_token in enumerate(create_mask_list):
        fg_dir = stylize_dir.joinpath(f"r_fg_{i:02d}_{time_str}")
        fg_dir.mkdir(parents=True, exist_ok=True)

        fg_masked_dir = fg_dir / "00_tmp_masked"
        fg_masked_dir.mkdir(parents=True, exist_ok=True)

        fg_mask_dir = fg_dir / "00_mask"
        fg_mask_dir.mkdir(parents=True, exist_ok=True)

        if use_animeseg:
            masked_area = animseg_create_fg(
                frame_dir=frame_dir,
                output_dir=fg_masked_dir,
                output_mask_dir=fg_mask_dir,
                masked_area_list=masked_area,
                mask_padding=mask_padding,
                bg_color=(0,255,0),
            )
        elif use_rembg:
            masked_area = rembg_create_fg(
                frame_dir=frame_dir,
                output_dir=fg_masked_dir,
                output_mask_dir=fg_mask_dir,
                masked_area_list=masked_area,
                mask_padding=mask_padding,
                bg_color=(0,255,0),
            )
        else:
            masked_area = create_fg(
                mask_token=mask_token,
                frame_dir=frame_dir,
                output_dir=fg_masked_dir,
                output_mask_dir=fg_mask_dir,
                masked_area_list=masked_area,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                mask_padding=mask_padding,
                sam_checkpoint= "data/models/SAM/sam_hq_vit_h.pth" if not low_vram else "data/models/SAM/sam_hq_vit_b.pth",
                bg_color=(0,255,0),
            )

        logger.info(f"mask from [{mask_token}] are output to {fg_dir}")

        output_list.append((fg_dir, fg_masked_dir, fg_mask_dir))

    torch.cuda.empty_cache()

    bg_dir = stylize_dir.joinpath(f"r_bg_{time_str}")
    bg_dir.mkdir(parents=True, exist_ok=True)

    bg_inpaint_dir = bg_dir / "00_tmp_inpainted"
    bg_inpaint_dir.mkdir(parents=True, exist_ok=True)


    create_bg(frame_dir, bg_inpaint_dir, masked_area,
              use_half = True,
              raft_iter = 20,
              subvideo_length=80 if not low_vram else 50,
              neighbor_length=10 if not low_vram else 8,
              ref_stride=10 if not low_vram else 8,
              low_vram = low_vram,
              )

    logger.info(f"background are output to {bg_dir}")


    output_list.append((bg_dir,bg_inpaint_dir,None))

    torch.cuda.empty_cache()

    black_list = []
    if ignore_list.is_file():
        with open(ignore_list) as f:
            black_list = [s.strip() for s in f.readlines()]

    black_list.append("simple_background")
    black_list.append("green_background")

    region_map = {}

    for i, (output_root, masked_dir, mask_dir) in enumerate(output_list):

        prompt_map = get_labels(
            frame_dir= masked_dir,
            interval=predicte_interval,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            ignore_tokens=black_list,
            with_confidence=with_confidence,
            is_danbooru_format=is_danbooru_format,
            is_cpu = False,
        )

        if mask_dir:

            ipadapter_dir = output_root / "00_ipadapter"
            ipadapter_dir.mkdir(parents=True, exist_ok=True)

            region_map[str(i)]={
                "enable": True,
                "crop_generation_rate": 0.0,
                "mask_dir" : os.path.relpath(mask_dir.absolute(), data_dir),
                "save_mask": True,
                "is_init_img" : False,
                "condition" : {
                    "prompt_fixed_ratio": 0.5,
                    "head_prompt": "",
                    "prompt_map": prompt_map,
                    "tail_prompt": "",
                    "ip_adapter_map": {
                        "enable": True,
                        "input_image_dir": os.path.relpath(ipadapter_dir.absolute(), data_dir),
                        "prompt_fixed_ratio": 0.5,
                        "save_input_image": True,
                        "resized_to_square": False
                    }
                }
            }
        else:
            region_map["background"]={
                "is_init_img" : False,
                "hint" : "background's condition refers to the one in root"
            }

            model_config.prompt_map = prompt_map


        model_config.region_map =region_map


        config_org.write_text(model_config.json(indent=4), encoding="utf-8")


