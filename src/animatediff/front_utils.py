import gradio as gr
from animatediff.execute import execute
import sys
import io
import os
import time
import pytz
from pathlib import Path
from datetime import datetime
import glob
import re
import json
import yt_dlp
import shutil
import pytz
from PIL import Image

from animatediff import __version__, get_dir
from animatediff.stylize import create_config, create_mask, generate, composite
from animatediff.settings import ModelConfig, get_model_config
from animatediff.cli import refine
from animatediff.video_utils import create_video

def getNow() -> str:
    singapore_timezone = pytz.timezone('Asia/Singapore')
    time_str = datetime.now(singapore_timezone).strftime("%Y%m%d_%H%M")
    return time_str

def get_schedulers():
    return [("LCM", "lcm"),
        ("DDIM", "ddim"),
        ("PNDM", "pndm"),
        ("Heun", "heun"),
        ("UniPC", "unipc"),
        ("Euler", "euler"),
        ("Euler a", "euler_a"),
        ("LMS", "lms"),
        ("LMS Karras", "k_lms"),
        ("DPM2", "dpm_2"),
        ("DPM2 Karras", "k_dpm_2"),
        ("DPM2 a", "dpm_2_a"),
        ("DPM2 a Karras", "k_dpm_2_a"),
        ("DPM++ 2M", "dpmpp_2m"),
        ("DPM++ 2M Karras", "k_dpmpp_2m"),
        ("DPM++ SDE", "dpmpp_sde"),
        ("DPM++ SDE Karras", "k_dpmpp_sde"),
        ("DPM++ 2M SDE", "dpmpp_2m_sde"),
        ("DPM++ 2M SDE Karras", "k_dpmpp_2m_sde")]
    
def create_file_list(folder_path):
    file_list = []
    files = os.listdir(folder_path)
    files.sort(key=lambda x: (os.path.splitext(x)[0].lower(), x))
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_list.append(file_name)
    return file_list

def get_stylize_dir(video_name:str)-> Path:
    stylize_dir='stylize/' + video_name
    return Path(stylize_dir)

def get_fg_dir(video_name:str) -> Path:
    fg_folder_name = 'fg_00_'+video_name
    return get_stylize_dir(video_name) / fg_folder_name

def get_mask_dir(video_name:str) -> Path:
    return get_fg_dir(video_name) / '00_mask'

def get_bg_dir(video_name:str) -> Path:
    bg_folder_name = 'bg_' + video_name
    return get_stylize_dir(video_name) / bg_folder_name

def find_safetensor_files(folder, suffix=''):
    result_list = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".safetensors") or file.endswith(".ckpt"):
                file_path = os.path.join(root, file)
                folder_name = os.path.relpath(root, folder)
                file_name = os.path.splitext(file)[0]
                
                if folder_name != ".":
                    file_name = os.path.join(folder_name, file_name)
                
                result_name = f"{suffix}{file_name}"
                result_path = os.path.relpath(file_path, folder)
                if folder.startswith("data/"):
                    folder2 = folder[len("data/"):]
                result_list.append((result_name, folder2+'/'+result_path))

        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            subdir_suffix = f"{suffix}{subdir}/" if suffix else f"{subdir}/"
            result_list.extend(find_safetensor_files(subdir_path, subdir_suffix))
            
    result_list.sort(key=lambda x: x[0])  # file_name でソート
    return result_list
    
def find_last_folder_and_mp4_file(folder_path):
    subfolders = sorted([f.path for f in os.scandir(folder_path) if f.is_dir() and f.name[0].isdigit()], key=lambda x: os.path.basename(x))
    last_folder = subfolders[-1]
    mp4_files = glob.glob(os.path.join(last_folder, '*.mp4'))
    if mp4_files:
        return mp4_files[0]
    else:
        return None
    
def find_next_available_number(save_folder):
    existing_files = [f for f in os.listdir(save_folder) if f.startswith('dance') and f.endswith('.mp4')]
    existing_numbers = [int(file[5:10]) for file in existing_files]

    if existing_numbers:
        return max(existing_numbers) + 1
    else:
        return 1

def download_video(video_url, save_folder) -> Path:
    v_name = load_video_name(video_url)
    ydl_opts = {
        'outtmpl': os.path.join(save_folder, f'{v_name}.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(video_url, download=True)
        if 'entries' in result:
            for entry in result['entries']:
                if 'filename' in entry:
                    return saved_file_paths
                else:
                    # Alternative approach to determine file name
                    file_extension = entry.get('ext', 'mp4')
                    return os.path.join(save_folder, f'{v_name}.{file_extension}')
        else:
            if 'filename' in result:
                return result['filename']
            else:
                # Alternative approach to determine file name
                file_extension = result.get('ext', 'mp4')
                return os.path.join(save_folder, f'{v_name}.{file_extension}')
    
def download_videos(video_urls, save_folder):
    saved_file_paths = []
    for video_url in video_urls:
        v_name = load_video_name(video_url)
        ydl_opts = {
            'outtmpl': os.path.join(save_folder, f'{v_name}.%(ext)s'),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(video_url, download=True)
            if 'entries' in result:
                for entry in result['entries']:
                    if 'filename' in entry:
                        saved_file_paths.append(entry['filename'])
                    else:
                        # Alternative approach to determine file name
                        file_extension = entry.get('ext', 'mp4')
                        saved_file_paths.append(os.path.join(save_folder, f'{v_name}.{file_extension}'))
            else:
                if 'filename' in result:
                    saved_file_paths.append(result['filename'])
                else:
                    # Alternative approach to determine file name
                    file_extension = result.get('ext', 'mp4')
                    saved_file_paths.append(os.path.join(save_folder, f'{v_name}.{file_extension}'))
    return saved_file_paths

def find_and_get_composite_video(folder_path):
    folder_pattern = os.path.join(folder_path, 'cp_*')
    folders = glob.glob(folder_pattern)
    sorted_folders = sorted(folders)
    if sorted_folders:
        target_folder = sorted_folders[-1]
        mp4_files = glob.glob(os.path.join(target_folder, '*.mp4'))
        if mp4_files:
            return mp4_files[0]
    return None

def load_video_name(url):
    folder_path = './config/'
    file_path = os.path.join(folder_path, 'video_url.json')
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([], file, ensure_ascii=False, indent=2)
        data = []
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

    existing_entry = next((entry for entry in data if entry['url'] == url), None)
    if existing_entry:
        return existing_entry['video_name']
    else:
        count = len(data) + 1
        new_video_name = f'dance{count:05d}'
        new_entry = {'url': url, 'video_name': new_video_name}
        data.append(new_entry)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return new_video_name

def get_last_sorted_subfolder(base_folder):
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir() and f.name[0].isdigit()]
    sorted_subfolders = sorted(subfolders, key=lambda folder: os.path.basename(folder), reverse=True)
    last_sorted_subfolder = sorted_subfolders[0] if sorted_subfolders else None
    return last_sorted_subfolder

def get_first_sorted_subfolder(base_folder):
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir() and f.name[0].isdigit()]
    sorted_subfolders = sorted(subfolders, key=lambda folder: os.path.basename(folder), reverse=False)
    last_sorted_subfolder = sorted_subfolders[0] if sorted_subfolders else None
    return last_sorted_subfolder

def change_ip(enable):
    ip_ch= gr.Checkbox(value=enable)
    ip_image = gr.UploadButton(interactive=enable)
    ip_scale = gr.Slider(interactive=enable)
    ip_type = gr.Radio(interactive=enable)
    return ip_ch, ip_image, ip_scale, ip_type
    
def change_ad(enable):
    ad_ch = gr.Checkbox(value=enable)
    ad_scale = gr.Slider(interactive=enable)
    return ad_ch, ad_scale

def change_op(enable):
    op_ch = gr.Checkbox(value=enable)
    op_scale = gr.Slider(interactive=enable)
    return op_ch, op_scale

def change_dp(enable):
    dp_ch = gr.Checkbox(value=enable)
    dp_scale = gr.Slider(interactive=enable)
    return dp_ch, dp_scale

def change_la(enable):
    la_ch = gr.Checkbox(value=enable)
    la_scale = gr.Slider(interactive=enable)
    return la_ch, la_scale
    
def create_config_by_gui(
    now_str:str,
    video:str,
    stylize_dir: Path, 
    model: str, vae: str,
    motion_module: str, context:str, scheduler: str, 
    is_lcm: bool, is_hires: bool,
    step: int, cfg: float, seed:int, 
    head_prompt:str,
    neg_prompt:str,
    inp_lora1: str, inp_lora1_step: float,
    inp_lora2: str, inp_lora2_step: float,
    inp_lora3: str, inp_lora3_step: float,
    inp_lora4: str, inp_lora4_step: float,
    mo1_ch: str, mo1_scale: float,
    mo2_ch: str, mo2_scale: float,
    mask_target:str,
    ip_ch: bool, ip_image: Image, ip_scale: float, ip_type: str,
    ad_ch: bool, ad_scale: float, op_ch: bool, op_scale: float,
    dp_ch: bool, dp_scale:float, la_ch: bool, la_scale: float,
) -> Path:
    data_dir = get_dir("data")
    org_config='config/fix/real_base2.json'
    model_config: ModelConfig = get_model_config(org_config)
    # 引数とその値を表示
    print(f"data_dir: {data_dir}")
    print(f"now_str: {now_str}")
    print(f"video: {video}")
    print(f"stylize_dir: {stylize_dir}")
    print(f"model: {model}")
    print(f"vae: {vae}")
    print(f"motion_module: {motion_module}")
    print(f"context: {context}")
    print(f"scheduler: {scheduler}")
    print(f"is_lcm: {is_lcm}")
    print(f"is_hires: {is_hires}")
    print(f"step: {step}")
    print(f"cfg: {cfg}")
    print(f"Seed: {seed}")
    print(f"head_prompt: {head_prompt}")
    print(f"neg_prompt: {neg_prompt}")
    print(f"inp_lora1: {inp_lora1}")
    print(f"inp_lora1_step: {inp_lora1_step}")
    print(f"inp_lora2: {inp_lora2}")
    print(f"inp_lora2_step: {inp_lora2_step}")
    print(f"inp_lora3: {inp_lora3}")
    print(f"inp_lora3_step: {inp_lora3_step}")
    print(f"inp_lora4: {inp_lora4}")
    print(f"inp_lora4_step: {inp_lora4_step}")
    print(f"mo1_ch: {mo1_ch}")
    print(f"mo1_scale: {mo1_scale}")
    print(f"mo2_ch: {mo2_ch}")
    print(f"mo2_scale: {mo2_scale}")
    print(f"ip_ch: {ip_ch}")
    print(f"ip_image: {ip_image}")
    print(f"ip_scale: {ip_scale}")
    print(f"ip_type: {ip_type}")
    print(f"ad_ch: {ad_ch}")
    print(f"ad_scale: {ad_scale}")
    print(f"op_ch: {op_ch}")
    print(f"op_scale: {op_scale}")
    print(f"dp_ch: {dp_ch}")
    print(f"dp_scale: {dp_scale}")
    print(f"la_ch: {la_ch}")
    print(f"la_scale: {la_scale}")

    print(ip_image)
    
    model_config.name = now_str
    model_config.path = Path(model)
    model_config.motion_module = Path(motion_module)
    model_config.vae_path = vae if vae is not None else ""
    model_config.context_schedule = context
    model_config.steps = step
    model_config.guidance_scale = cfg
    model_config.scheduler = scheduler
    model_config.head_prompt = head_prompt
    model_config.n_prompt = [neg_prompt]
    model_config.seed = [seed]
    # model_config.n_prompt = [neg.strip() for neg in neg_prompt.split(',') if neg]
    model_config.lcm_map = {
        "enable": is_lcm,
        "start_scale": 0.15,
        "end_scale": 0.75,
        "gradient_start": 0.2,
        "gradient_end": 0.75
    }
    model_config.gradual_latent_hires_fix_map = {
        "enable": is_hires,
        "scale": {
            "0": 0.5,
            "0.7": 1.0
        },
        "reverse_steps": 5,
        "noise_add_count": 3
    }
    model_config.stylize_config = {
            "original_video": {
                "path": video,
                "aspect_ratio": -1,
                "offset": 0
            },
            "create_mask": [
                mask_target
            ],
            "composite": {
                "fg_list": [
                    {
                        "path": " absolute path to frame dir ",
                        "mask_path": " absolute path to mask dir (this is optional) ",
                        "mask_prompt": "person"
                    }
                ],
                "bg_frame_dir": "Absolute path to the BG frame directory",
                "hint": ""
            },
            "0": {
                "width": 512,
                "height": 904,
                "length": 16,
                "context": 16,
                "overlap": 4,
                "stride": 0
            }
        }
    model_config.lora_map = {}
    print(inp_lora1)
#    if inp_lora1 is not None:
    if len(inp_lora1) > 0:
        model_config.lora_map.update({inp_lora1 : {
            "region": ["0"],
            "scale": {"0": inp_lora1_step}
        }})
    if len(inp_lora2) > 0:
        model_config.lora_map.update({inp_lora2 : {
            "region": ["0"],
            "scale": {"0": inp_lora2_step}
        }})
    if len(inp_lora3) > 0:
        model_config.lora_map.update({inp_lora3 : {
            "region": ["0"],
            "scale": {"0": inp_lora3_step}
        }})
    if len(inp_lora4) > 0:
        model_config.lora_map.update({inp_lora4 : {
            "region": ["0"],
            "scale": {"0": inp_lora4_step}
        }})
    
    if mo1_ch is not None and len(mo1_ch) > 0:
        model_config.motion_lora_map[mo1_ch] = mo1_scale
    if mo2_ch is not None and len(mo2_ch) > 0:
        model_config.motion_lora_map[mo2_ch] = mo2_scale
    
    # model_config.controlnet_map["input_image_dir"] = stylize_dir/'00_controlnet_image'
    # model_config.controlnet_map["input_image_dir"] = os.path.relpath((stylize_dir/'00_controlnet_image').absolute(), data_dir)
    # model_config.img2img_map["init_img_dir"] = os.path.relpath((stylize_dir/'00_img2img').absolute(), data_dir)

    model_config.controlnet_map["input_image_dir"] = Path("..") / stylize_dir/'00_controlnet_image'
    model_config.img2img_map["init_img_dir"] = Path("..") / stylize_dir /'00_img2img'

    
    model_config.controlnet_map["max_samples_on_vram"] = 0
    model_config.controlnet_map["max_models_on_vram"] = 0
    model_config.controlnet_map["save_detectmap"] = True
    
    model_config.img2img_map["save_init_image"] = False
    model_config.ip_adapter_map["enable"] = ip_ch
    model_config.ip_adapter_map["input_image_dir"] = stylize_dir/'00_ipadapter'
    model_config.ip_adapter_map["scale"] = ip_scale
    model_config.ip_adapter_map["is_full_face"] = True if ip_type == "full_face" else False
    model_config.ip_adapter_map["is_plus_face"] = True if ip_type == "plus_face" else False
    model_config.ip_adapter_map["is_plus"] = True if ip_type == "plus" else False
    model_config.ip_adapter_map["is_light"] = True if ip_type == "light" else False
    model_config.ip_adapter_map["save_input_image"] = False
    save_image_to_path(ip_image, stylize_dir/'00_ipadapter'/'0.png')
    
    model_config.controlnet_map["animatediff_controlnet"]["enable"] = ad_ch
    model_config.controlnet_map["animatediff_controlnet"]["controlnet_conditioning_scale"] = ad_scale
    model_config.controlnet_map["controlnet_openpose"]["enable"] = op_ch
    model_config.controlnet_map["controlnet_openpose"]["controlnet_conditioning_scale"] = op_scale
    model_config.controlnet_map["controlnet_depth"]["enable"] = dp_ch
    model_config.controlnet_map["controlnet_depth"]["controlnet_conditioning_scale"] = dp_scale
    model_config.controlnet_map["controlnet_lineart"]["enable"] = la_ch
    model_config.controlnet_map["controlnet_lineart"]["controlnet_conditioning_scale"] = la_scale
    
    save_config_path = get_config_path(now_str)
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")

def save_image_to_path(image, file_path):
    if image is not None:
        try:
            # 保存前にフォルダ内のデータを削除
            folder_path = os.path.dirname(file_path)
            if os.path.exists(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path_to_delete = os.path.join(folder_path, file_name)
                    try:
                        if os.path.isfile(file_path_to_delete):
                            os.unlink(file_path_to_delete)
                        elif os.path.isdir(file_path_to_delete):
                            os.rmdir(file_path_to_delete)
                    except Exception as e:
                        print(f"Failed to delete {file_path_to_delete}: {e}")

            # イメージを指定したパスに保存
            image.save(file_path)
            print(f"Image saved successfully to {file_path}")
        except Exception as e:
            print(f"An error occurred while saving the image: {e}")
    
def get_config_path(now_str:str) -> Path:
    config_dir = Path("./config/from_ui")
    config_path = config_dir.joinpath(now_str+".json")
    return config_path
    
def update_config(now_str:str, video_name:str, mask_ch:bool):
    config_path = get_config_path(now_str)
    model_config: ModelConfig = get_model_config(config_path)
    stylize_dir = get_stylize_dir(video_name)
    stylize_fg_dir = get_fg_dir(video_name)
    
    img2img_dir = stylize_dir/"00_img2img"
    img = Image.open( img2img_dir.joinpath("00000000.png") )
    W, H = img.size
    gradual_latent_hires_fix = model_config.gradual_latent_hires_fix_map["enable"]
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
    model_config.stylize_config["0"]= {
                "width": width,
                "height": height,
                "length": length,
                "context": 16,
                "overlap": 4,
                "stride": 0
            }
    actual_config_path = stylize_fg_dir/'prompt.json' if mask_ch else stylize_dir/'prompt.json'
    actual_config_path.write_text(model_config.json(indent=4), encoding="utf-8")
    config_path.write_text(model_config.json(indent=4), encoding="utf-8")


# class ModelConfig(BaseSettings):
#     name: str = Field(...)  # Config name, not actually used for much of anything
#     path: Path = Field(...)  # Path to the model
#     vae_path: str = ""  # Path to the model
#     motion_module: Path = Field(...)  # Path to the motion module
#     context_schedule: str = "uniform"
#     lcm_map: Dict[str,Any]= Field({})
#     gradual_latent_hires_fix_map: Dict[str,Any]= Field({})
#     compile: bool = Field(False)  # whether to compile the model with TorchDynamo
#     tensor_interpolation_slerp: bool = Field(True)
#     seed: list[int] = Field([])  # Seed(s) for the random number generators
#     scheduler: DiffusionScheduler = Field(DiffusionScheduler.k_dpmpp_2m)  # Scheduler to use
#     steps: int = 25  # Number of inference steps to run
#     guidance_scale: float = 7.5  # CFG scale to use
#     unet_batch_size: int = 1
#     clip_skip: int = 1  # skip the last N-1 layers of the CLIP text encoder
#     prompt_fixed_ratio: float = 0.5
#     head_prompt: str = ""
#     prompt_map: Dict[str,str]= Field({})
#     tail_prompt: str = ""
#     n_prompt: list[str] = Field([])  # Anti-prompt(s) to use
#     is_single_prompt_mode : bool = Field(False)
#     lora_map: Dict[str,Any]= Field({})
#     motion_lora_map: Dict[str,float]= Field({})
#     ip_adapter_map: Dict[str,Any]= Field({})
#     img2img_map: Dict[str,Any]= Field({})
#     region_map: Dict[str,Any]= Field({})
#     controlnet_map: Dict[str,Any]= Field({})
#     upscale_config: Dict[str,Any]= Field({})
#     stylize_config: Dict[str,Any]= Field({})
#     output: Dict[str,Any]= Field({})
#     result: Dict[str,Any]= Field({})