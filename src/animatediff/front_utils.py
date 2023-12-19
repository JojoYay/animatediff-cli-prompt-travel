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
from animatediff.stylize import create_config, create_mask, generate, composite
from animatediff.settings import ModelConfig, get_model_config
from animatediff.cli import refine
from animatediff.video_utils import create_video


def validate_inputs(url):
    if not url:
        yield 'Error: URLs input is required.', None, None, None, None, gr.Button("Generate Video", scale=1, interactive=False)

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
    subfolders = sorted([f.path for f in os.scandir(folder_path) if f.is_dir()], key=lambda x: os.path.basename(x))
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
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    sorted_subfolders = sorted(subfolders, key=lambda folder: os.path.basename(folder), reverse=True)
    # print(f"sorted_subfolders: {sorted_subfolders}")
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
    model: str, 
    motion_module: str, 
    scheduler: str, 
    step: int, 
    cfg: float, 
    head_prompt:str,
    neg_prompt:str,
    inp_lora1: str, inp_lora1_step: float,
    inp_lora2: str, inp_lora2_step: float,
    inp_lora3: str, inp_lora3_step: float,
    inp_lora4: str, inp_lora4_step: float,
    ip_ch: bool, ip_image: Image, ip_scale: float, ip_type: str,
    ad_ch: bool, ad_scale: float, op_ch: bool, op_scale: float,
    dp_ch: bool, dp_scale:float, la_ch: bool, la_scale: float,
) -> Path:
    org_config='config/fix/real_base2.json'
    model_config: ModelConfig = get_model_config(org_config)
    print(f"inp_posi{head_prompt}")
    print(f"inp_lora1{inp_lora1}")
    print(f"inp_lora1_step{inp_lora1_step}")
    print(f"neg_prompt{neg_prompt}")
    print(ip_image)
    
    model_config.name = now_str
    model_config.path = Path(model)
    model_config.motion_module = Path(motion_module)
    model_config.steps = step
    model_config.guidance_scale = cfg
    model_config.scheduler = scheduler
    model_config.head_prompt = head_prompt
    model_config.n_prompt = [neg.strip() for neg in neg_prompt.split(',') if neg]
    model_config.stylize_config = {
            "original_video": {
                "path": video,
                "aspect_ratio": -1,
                "offset": 0
            },
            "create_mask": [
                "person"
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
    
    model_config.controlnet_map["input_image_dir"] = stylize_dir/'00_controlnet_image'
    model_config.img2img_map["init_img_dir"] = stylize_dir/'00_img2img'
    
    model_config.controlnet_map["max_samples_on_vram"] = 0
    model_config.controlnet_map["max_models_on_vram"] = 0
    model_config.controlnet_map["save_detectmap"] = False
    
    model_config.img2img_map["save_init_image"] = False
    
    model_config.ip_adapter_map["enable"] = ip_ch
    model_config.ip_adapter_map["input_image_dir"] = stylize_dir/'00_ipadapter'
    model_config.ip_adapter_map["scale"] = ip_scale
    model_config.ip_adapter_map["is_full_face"] = True if ip_type == "is_full_face" else False
    model_config.ip_adapter_map["is_plus_face"] = True if ip_type == "is_plus_face" else False
    model_config.ip_adapter_map["is_plus"] = True if ip_type == "is_plus" else False
    model_config.ip_adapter_map["is_light"] = True if ip_type == "is_light" else False
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
    
def update_config(now_str:str, stylize_dir, stylize_fg_dir):
    config_path = get_config_path(now_str)
    
    model_config: ModelConfig = get_model_config(config_path)
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
    fg_config_path = stylize_fg_dir/'prompt.json'
    fg_config_path.write_text(model_config.json(indent=4), encoding="utf-8")
    config_path.write_text(model_config.json(indent=4), encoding="utf-8")

