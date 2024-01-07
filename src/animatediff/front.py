import gradio as gr
from animatediff.execute import execute
from animatediff.front_utils import (get_schedulers, getNow, download_video, create_file_list,
                                    find_safetensor_files, find_last_folder_and_mp4_file, find_next_available_number,
                                    find_and_get_composite_video, load_video_name, get_last_sorted_subfolder,
                                    create_config_by_gui, get_config_path, update_config, change_ip, change_cn, change_ref, get_first_sorted_subfolder, get_stylize_dir, get_fg_dir,
                                    get_mask_dir, get_bg_dir, select_v2v, select_t2v, select_data, select_url, select_video, pick_video, generate_example, change_re, find_mp4_files)
from animatediff.settings import ModelConfig, get_model_config
from animatediff.video_utils import create_video
from animatediff.generate import save_output

from animatediff.stylize import generate, create_config, create_mask, composite
from animatediff.cli import refine
import traceback
import PIL

import io
import os
import time
from pathlib import Path
import shutil

# Define the function signature
def execute_wrapper(
      tab_select:str, tab_select2:str, url: str, dl_video: str, t_name: str, t_length:int, t_width:int, t_height:int, fps: int,
      inp_model: str, inp_vae: str, 
      inp_mm: str, inp_context: str, inp_sche: str, 
      inp_lcm: bool, inp_hires: bool, low_vr: bool,
      inp_step: int, inp_cfg: float, seed: int,
      single_prompt: bool, prompt_fixed_ratio: float, tensor_interpolation_slerp: bool,
      inp_posi: str, inp_pro_map: str, inp_neg: str, 
      inp_lora1: str, inp_lora1_step: float,
      inp_lora2: str, inp_lora2_step: float,
      inp_lora3: str, inp_lora3_step: float,
      inp_lora4: str, inp_lora4_step: float,
      mo1_ch: str, mo1_scale: float,
      mo2_ch: str, mo2_scale: float,
      ip_ch: bool, ip_image: PIL.Image.Image, ip_scale: float, ip_type: str, ip_image_ratio:float,
      mask_ch1: bool, mask_target:str, mask_type1: str, mask_padding1:int,
      ad_ch: bool, ad_scale: float, op_ch: bool, op_scale: float,
      dp_ch: bool, dp_scale: float, la_ch: bool, la_scale: float,
      me_ch: bool, me_scale: float, i2i_ch: bool, i2i_scale: float,
      ref_ch: bool, ref_image: PIL.Image.Image, ref_attention: float, ref_gn: float, ref_weight: float,
      is_refine: bool, re_scale: float, re_interpo: float,
      delete_if_exists: bool, is_test: bool, 
      progress=gr.Progress(track_tqdm=True)):
    yield 'generation Initiated...', None, None, None, None, None, None, None, None, None, None, gr.Button("Generating...", scale=1, interactive=False)
    # yield 'generation Initiated...', None, [], gr.Button("Generating...", scale=1, interactive=False)
    start_time = time.time()
    time_str = getNow()
    try:
        if url is None and tab_select == 'V2V' and tab_select2 == 'URL':
            yield 'Error: URL is required.', None, None, None, None, None, None, None, None,None, None, gr.Button("Generate Video", scale=1, interactive=True)
            # yield 'Error: URL is required.', None, [], gr.Button("Generate Video", scale=1, interactive=True) #キャプションをちゃんと更新(TODO)
            return
        if dl_video == [] and tab_select == 'V2V' and tab_select2 == 'Data':
            yield 'Error: Select Video', None, None, None, None, None, None, None, None,None, None, gr.Button("Generate Video", scale=1, interactive=True)
            # yield 'Error: URL is required.', None, [], gr.Button("Generate Video", scale=1, interactive=True)
            return
        if inp_model == []:
            yield 'Error: Select Model', None, None, None, None, None, None, None, None, None, None, gr.Button("Generate Video", scale=1, interactive=True)
            # yield 'Error: Select Model', None, [], gr.Button("Generate Video", scale=1, interactive=True)
            return
        if inp_mm == []:
            yield 'Error: Select Motion Module', None, None, None, None, None, None, None, None, None, None, gr.Button("Generate Video", scale=1, interactive=True)
            # yield 'Error: Select Motion Module', None, [], gr.Button("Generate Video", scale=1, interactive=True)
            return
        if inp_sche == []:
            yield 'Error: Select Sampling Method', None, None, None, None, None, None, None, None, None, None, gr.Button("Generate Video", scale=1, interactive=True)
            # yield 'Error: Select Sampling Method', None, [], gr.Button("Generate Video", scale=1, interactive=True)
            return
        if t_name is None and tab_select == 'T2V':
            yield 'Error: Video Name is required', None, None, None, None, None, None, None, None, None, None, gr.Button("Generate Video", scale=1, interactive=True)
            # yield 'Error: Video Name is required', None, [], gr.Button("Generate Video", scale=1, interactive=True)
            return
        
        if tab_select == 'T2V':
            mask_ch1 = False
            ad_ch = False
            dp_ch = False
            me_ch = False
            is_test = False
            delete_if_exists = True
        
        bg_config = None
        if tab_select == 'V2V':
            if tab_select2 == 'URL':
                save_folder = Path('data/video')
                saved_file = download_video(url, save_folder)

            else: #tab_select2 == "Data"
                saved_file = 'data/'+dl_video
                
            separator = os.path.sep
            video_name = os.path.splitext(os.path.normpath(saved_file.replace('/notebooks', separator)))[0].rsplit(separator, 1)[-1]
        else:
            video_name = t_name
            saved_file = None
        
        # video_name=saved_file.rsplit('.', 1)[0].rsplit('/notebooks', 1)[-1].rsplit('/', 1)[-1]
        stylize_dir= get_stylize_dir(video_name, str(fps))
        create_config_by_gui(
            now_str=time_str,
            video = saved_file,
            stylize_dir = stylize_dir, 
            model=inp_model, vae=inp_vae, fps=fps,
            motion_module=inp_mm, context=inp_context, scheduler=inp_sche, 
            is_lcm=inp_lcm, is_hires=inp_hires,
            step=inp_step, cfg=inp_cfg, seed=seed,
            single_prompt=single_prompt, prompt_fixed_ratio=prompt_fixed_ratio, tensor_interpolation_slerp=tensor_interpolation_slerp,
            head_prompt=inp_posi, inp_pro_map=inp_pro_map, neg_prompt=inp_neg,
            inp_lora1=inp_lora1, inp_lora1_step=inp_lora1_step,
            inp_lora2=inp_lora2, inp_lora2_step=inp_lora2_step,
            inp_lora3=inp_lora3, inp_lora3_step=inp_lora3_step,
            inp_lora4=inp_lora4, inp_lora4_step=inp_lora4_step,
            mo1_ch=mo1_ch, mo1_scale=mo1_scale,
            mo2_ch=mo2_ch, mo2_scale=mo2_scale,
            mask_target=mask_target,
            ip_ch=ip_ch, ip_image=ip_image, ip_scale=ip_scale, ip_type=ip_type,ip_image_ratio=ip_image_ratio,
            ad_ch=ad_ch, ad_scale=ad_scale, op_ch=op_ch, op_scale=op_scale,
            dp_ch=dp_ch, dp_scale=dp_scale, la_ch=la_ch, la_scale=la_scale,
            me_ch=me_ch, me_scale=me_scale, i2i_ch=i2i_ch, i2i_scale=i2i_scale,
            ref_ch=ref_ch, ref_image=ref_image, ref_attention=ref_attention, ref_gn=ref_gn, ref_weight=ref_weight,
            tab_select=tab_select, t_name=t_name, t_length=t_length, t_width=t_width, t_height=t_height
        )

        yield from execute_impl(tab_select=tab_select, fps=fps,now_str=time_str,video=saved_file, delete_if_exists=delete_if_exists,
                                is_test=is_test, is_refine=is_refine, re_scale=re_scale, re_interpo=re_interpo, 
                                bg_config=bg_config, mask_ch1=mask_ch1, mask_type=mask_type1, 
                                mask_padding1=mask_padding1, is_low=low_vr, t_name=t_name, ip_image=ip_image, ref_image=ref_image)
    except Exception as inst:
        # yield 'Runtime Error', None, [], gr.Button("Generate Video", scale=1, interactive=True)
        yield 'Runtime Error', None, None, None, None, None, None, None, None, None, None, gr.Button("Generating...", scale=1, interactive=True)
        # print(type(inst))    # the exception type
        # print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
        traceback.print_exc()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"実行時間: {execution_time}秒")

    
def execute_impl(tab_select:str, now_str:str, video: str, delete_if_exists: bool, is_test: bool, is_refine: bool, re_scale:float, re_interpo:float,
                 bg_config: str, fps:int, mask_ch1: bool, mask_type: str, mask_padding1:int, is_low:bool, t_name:str, ip_image:PIL.Image.Image, ref_image:PIL.Image.Image,):
    if tab_select == 'V2V':
        if video.startswith("/notebooks"):
            video = video[len("/notebooks"):]
        if bg_config is not None:
            if bg_config.startswith("/notebooks"):
                bg_config = bg_config[len("/notebooks"):]
        print(f"video1: {video}")
    else:
        mask_ch1 = False
        ad_ch = False
        dp_ch = False
        me_ch = False
        is_test = False
        delete_if_exists = True        
        
        video = None
        video_name = t_name
    try:
        original = video
        ip_adapter_pic = None
        mask_video = None
        depth_video = None
        lineart_video = None
        openpose_video = None
        media_face_video = None
        front_video = None
        front_refine = None
        composite_video = None
        final_video = None
        # video_paths = [original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video]
        yield 'generating config...', video, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video, gr.Button("Generating...", scale=1, interactive=False)
        # yield 'generating config...', pick_video(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), generate_example(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), gr.Button("Generating...", scale=1, interactive=False)
        if tab_select == 'V2V':
            separator = os.path.sep
            video_name = os.path.splitext(os.path.normpath(video.replace('/notebooks', separator)))[0].rsplit(separator, 1)[-1]

            # video_name=video.rsplit('.', 1)[0].rsplit('/notebooks', 1)[-1].rsplit('/', 1)[-1]
            video = Path(video).resolve()
        str_fps = str(fps)
        stylize_dir = get_stylize_dir(video_name, str_fps)
        stylize_fg_dir = get_fg_dir(video_name, str_fps)
        mask_dir = get_mask_dir(video_name, str_fps)
        stylize_bg_dir = get_bg_dir(video_name, str_fps)

        print(f"stylize_dir:{stylize_dir}")
        print(f"stylize_fg_dir:{stylize_fg_dir}")

        if bg_config is not None:
            bg_config = Path(bg_config)
            bg_model_config: ModelConfig = get_model_config(bg_config)

        if stylize_dir.exists() and not delete_if_exists:
            print(f"config already exists. skip create-config")
            # if mask_ch != "As is Base":
            if mask_ch1:
                mask_video = mask_dir/'mask.mp4'
        else:
            if stylize_dir.exists():
                print(f"Delete folder and create again")
                shutil.rmtree(stylize_dir)
            if tab_select == 'V2V':
                create_config(org_movie=video, fps=fps, low_vram=is_low)
                # !animatediff stylize create-config {video} -f {fps}

        # if not stylize_fg_dir.exists() and mask_ch1:
        if not mask_dir.exists() and mask_ch1:
            create_mask(stylize_dir=stylize_dir, bg_config=bg_config, no_crop=True, low_vram=is_low)
            # !animatediff stylize create-mask {stylize_dir} -mp {mask_padding} -nc　
            if mask_ch1:
                mask_video = mask_dir/'mask.mp4'
                save_output(
                    None,
                    mask_dir,
                    mask_video,
                    {"format":"mp4", "fps":fps},
                    False,
                    False,
                    None,
                )
        update_config(now_str, video_name, mask_ch1, tab_select, ip_image, ref_image, str_fps)
        config = get_config_path(now_str)
        model_config: ModelConfig = get_model_config(config)

        yield 'generating fg bg video...', video, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video, gr.Button("Generating...", scale=1, interactive=False)
        # yield 'generating fg bg video...', pick_video(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), generate_example(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), gr.Button("Generating...", scale=1, interactive=False)

        print(f"Start: stylize generate {stylize_fg_dir}")
        print(f"test: {is_test}")
        if is_test:
            # if mask_ch != "As is Base":
            if mask_ch1:
                generate(stylize_dir=stylize_fg_dir, length=16)
                # !animatediff stylize generate {stylize_fg_dir} -L 16
                if bg_config is not None:
                    generate(stylize_dir=stylize_bg_dir, length=16)
                    # !animatediff stylize generate {stylize_bg_dir} -L 16
                front_video = find_last_folder_and_mp4_file(stylize_fg_dir)
                detect_map = get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir))
            else:
                generate(stylize_dir=stylize_dir, length=16)
                # !animatediff stylize generate {stylize_dir} -L 16
                front_video = find_last_folder_and_mp4_file(stylize_dir)
                detect_map = get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_dir))
        else:
            # if mask_ch != "As is Base":
            if mask_ch1:
                generate(stylize_dir=stylize_fg_dir)
                # !animatediff stylize generate {stylize_fg_dir}
                if bg_config is not None:
                    generate(stylize_dir=stylize_bg_dir)
                    # !animatediff stylize generate {stylize_bg_dir}
                front_video = find_last_folder_and_mp4_file(stylize_fg_dir)
                detect_map = get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir))
            else:
                print(f"generate {stylize_dir} start")
                generate(stylize_dir=stylize_dir)
                # !animatediff stylize generate {stylize_dir}
                front_video = find_last_folder_and_mp4_file(stylize_dir)
                detect_map = get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_dir))
        print("###########################################################################################")

        print(f"video2: {front_video}")
        print(f"detect_map:{detect_map}")
        subfolders = [f.path for f in os.scandir(Path(detect_map)) if f.is_dir()]
        print(f"cn_folders: {subfolders}")
        for cn_folder in subfolders:
            # フォルダ名が "animate_diff" でない場合の処理
            print(f"cn_folder: {cn_folder}")
            if os.path.basename(cn_folder) != "animatediff_controlnet" and os.path.basename(cn_folder) != "controlnet_ref":
                filename = Path(cn_folder + '/' + os.path.basename(cn_folder)+'.mp4')
                save_output(
                    None,
                    Path(cn_folder),
                    filename,
                    {"format":"mp4","fps":fps},
                    False,
                    False,
                    None,
                )

                if os.path.basename(cn_folder) == "controlnet_depth":
                    depth_video = filename
                if os.path.basename(cn_folder) == "controlnet_lineart":
                    lineart_video = filename
                if os.path.basename(cn_folder) == "controlnet_openpose":
                    openpose_video = filename
                if os.path.basename(cn_folder) == "controlnet_mediapipe_face":
                    media_face_video = filename

        if is_refine:
            cur_width = model_config.stylize_config["0"]["width"]
            print(f"cur_width {cur_width}")
            new_width = int(float(cur_width) * float(1.5))
            print(f"refine width {new_width}")
            yield 'refining fg video', video, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video, gr.Button("Generating...", scale=1, interactive=False)
            # yield 'refining fg video', pick_video(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), generate_example(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), gr.Button("Generating...", scale=1, interactive=False)
            # if mask_ch != "As is Base":
            if mask_ch1:
                result_dir = get_first_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir))
                print(f"Start: Refine {result_dir} -width {new_width}")
                refine(frames_dir=result_dir, out_dir=stylize_fg_dir, config_path=config, width=new_width, tile_conditioning_scale=re_scale, interpolation_multiplier=re_interpo)
                # tile_upscale(frames_dir=result_dir, out_dir=stylize_fg_dir, config_path=config, width=new_width)
                # !animatediff refine {result_dir} -o {stylize_fg_dir} -c {config} -W {new_width}
                front_refine = find_last_folder_and_mp4_file(get_last_sorted_subfolder(stylize_fg_dir))
                print(f"front_video: {front_video}")
                fg_result = get_first_sorted_subfolder(get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir)))
                print(f"fg_result1{fg_result}")
                if mask_type == 'No Background': 
                # if mask_ch == 'Nothing Base':
                    semi_final_video = find_last_folder_and_mp4_file(get_last_sorted_subfolder(stylize_fg_dir))
                    front_refine = find_last_folder_and_mp4_file(get_last_sorted_subfolder(stylize_fg_dir))

            else:
                result_dir = get_first_sorted_subfolder(get_last_sorted_subfolder(stylize_dir))
                print(f"Start: Refine {result_dir} -width {new_width}")
                refine(frames_dir=result_dir, out_dir=stylize_fg_dir, config_path=config, width=new_width, tile_conditioning_scale=re_scale, interpolation_multiplier=re_interpo)
                # !animatediff refine {result_dir} -o {stylize_dir} -c {config} -W {new_width}
                print(f"front_video: {front_video}")
                fg_result = get_last_sorted_subfolder(get_last_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir)))
                # front_video = find_last_folder_and_mp4_file(get_last_sorted_subfolder(stylize_fg_dir))
                semi_final_video = find_last_folder_and_mp4_file(get_last_sorted_subfolder(stylize_fg_dir))
                front_refine = find_last_folder_and_mp4_file(get_last_sorted_subfolder(stylize_fg_dir))
        else:
            if mask_ch1:
            # if mask_ch != "As is Base":
                fg_result = get_first_sorted_subfolder(get_last_sorted_subfolder(stylize_fg_dir))
                # if mask_ch == 'Nothing Base':
                if mask_type == 'No Background': 
                    semi_final_video = find_last_folder_and_mp4_file(stylize_fg_dir)
            else:
                fg_result = get_first_sorted_subfolder(get_last_sorted_subfolder(stylize_dir))
                semi_final_video = find_last_folder_and_mp4_file(stylize_dir)

        # if mask_ch == "Original":
        if mask_ch1 and mask_type == 'Original': 
            yield 'composite video', video, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video, gr.Button("Generating...", scale=1, interactive=False)
            # yield 'composite video', pick_video(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), generate_example(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), gr.Button("Generating...", scale=1, interactive=False)
            bg_result = get_last_sorted_subfolder(stylize_bg_dir)

            print(f"fg_result:{fg_result}")
            if bg_config is not None:
                print(f"bg_dir: {bg_result}")
            else:
                print(f"bg_dir: {stylize_bg_dir/'00_img2img'}")

            if bg_config is not None:
                final_video_dir = composite(stylize_dir=stylize_dir, bg_dir=bg_result, fg_dir=fg_result)
                # !animatediff stylize composite {stylize_dir} -bg {bg_result} -fg {fg_result}  
            else:
                bg_result = stylize_bg_dir/'00_img2img'
                print(f"stylize_dir: {stylize_dir}")
                print(f"bg_result: {bg_result}")
                print(f"fg_result: {fg_result}")
                composite(stylize_dir=stylize_dir, bg_dir=bg_result, fg_dir=fg_result)
                # !animatediff stylize composite {stylize_dir} -bg {bg_result} -fg {fg_result}

            semi_final_video = find_and_get_composite_video(stylize_dir)
            composite_video = find_and_get_composite_video(stylize_dir)
            
        print(f"final_video_dir: {semi_final_video}")

        final_dir = os.path.dirname(semi_final_video)
        final_video = os.path.join(final_dir,  video_name + ".mp4")

    #    final_video_dir: stylize/dance00023/cp_2023-12-18_08-09/composite2023-12-18_08-09-41
        try:
            video = Path(video).as_posix()
            semi_final_video = Path(semi_final_video).as_posix()
            final_video = Path(final_video).as_posix()
            create_video(video, semi_final_video, final_video)
            print(f"new_file_path: {final_video}")

            yield 'video is ready!', video, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video, gr.Button("Generate Video", scale=1, interactive=True)
            # yield 'video is ready!', pick_video(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), generate_example(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), gr.Button("Generate Video", scale=1, interactive=True)
        except Exception as e:
            # print(f"error:{e}")
            traceback.print_exc()
            # print(type(e))    # the exception type
            # print(e.args)     # arguments stored in .args
            # print(e)          # __str__ allows args to be printed directly,
            final_video = semi_final_video
            yield 'video is ready!(no music added)', video, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video, gr.Button("Generate Video", scale=1, interactive=True)
            # yield 'video is ready!(no music added)', pick_video(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), generate_example(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), gr.Button("Generate Video", scale=1, interactive=True)

    except Exception as inst:
        yield 'Runtime Error', video, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video, gr.Button("Generate Video", scale=1, interactive=True)
        # yield 'Runtime Error', pick_video(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), generate_example(original, mask_video, depth_video, lineart_video, openpose_video, media_face_video, front_video, front_refine, composite_video, final_video), gr.Button("Generate Video", scale=1, interactive=True)
        # print(type(inst))    # the exception type
        # print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
        traceback.print_exc()


def launch():
    result_list = create_file_list("config/fix")
    safetensor_files = find_safetensor_files("data/sd_models")
    lora_files = find_safetensor_files("data/lora")
    mm_files = find_safetensor_files("data/motion_modules")
    schedulers = get_schedulers()
    ip_choice = ["full_face", "plus_face", "plus", "light"]
    ml_files = find_safetensor_files("data/motion_lora")
    # bg_choice = ["Original", "Nothing Base", "As is Base"]
    mask_type_choice = ["Original", "No Background"]
    context_choice = ["uniform", "composite"]
    vae_choice = find_safetensor_files("data/vae")
    video_files = find_mp4_files("data/video")
    
    with gr.Blocks() as iface:
        with gr.Row():
            gr.Markdown(
                """
                # AnimateDiff-Prompt-Travel-Extravaganza
                """, scale=8)
            btn = gr.Button("Generate V2V", scale=1)
            tab_select = gr.Textbox(lines=1, value="V2V", show_label=False, visible=False)
            tab_select2 = gr.Textbox(lines=1, value="Data", show_label=False, visible=False)
        with gr.Row():
            with gr.Column():
                with gr.Tab("V2V") as v2v_tab:
                    with gr.Tab("Existing Video") as data_tab:
                        dl_video = gr.Dropdown(choices=video_files, label="Videos")
                    with gr.Tab("Download from URL") as url_tab:
                        url = gr.Textbox(lines=1, value="https://www.tiktok.com/@ai_hinahina/video/7313863412541361426", show_label=False)
                with gr.Tab("T2V") as t2v_tab:
                    with gr.Group():
                        with gr.Row():
                            t_name = gr.Textbox(lines=1, label="Video Name")
                            t_length = gr.Slider(minimum=16, maximum=3840,  step=8, value=16, label="Length")
                        with gr.Row():
                            t_width = gr.Slider(minimum=384, maximum=1360,  step=8, value=512, label="Width")
                            t_height = gr.Slider(minimum=384, maximum=1360,  step=8, value=904, label="Height")
                    # key_prompts = gr.Textbox(lines=2, value='"0": "best quality"', label="Prompt")
                with gr.Group():
                    with gr.Group():
                        with gr.Row():
                            inp_model = gr.Dropdown(choices=safetensor_files, label="Model")
                            inp_vae = gr.Dropdown(choices=vae_choice, label="VAE")
                            fps = gr.Slider(minimum=8, maximum=64, step=1, value=16, label="fps")
                    with gr.Group():
                        with gr.Row():
                            inp_mm = gr.Dropdown(choices=mm_files, label="Motion Module")
                            inp_sche = gr.Dropdown(choices=schedulers, label="Sampling Method", value="k_dpmpp_sde")
                            inp_context = gr.Dropdown(choices=context_choice, label="Context", value="uniform")
                    with gr.Group():
                        with gr.Row():
                            inp_lcm = gr.Checkbox(label="LCM", value=True)
                            inp_hires = gr.Checkbox(label="gradual latent hires fix", value=False)
                            low_vr = gr.Checkbox(label="Low VRAM", value=False)
                    with gr.Group():
                        with gr.Row():
                            seed = gr.Number(value=-1, label="Seed")
                            inp_step = gr.Slider(minimum=1, maximum=50, step=1, value=8, label="Sampling Steps")
                            inp_cfg = gr.Slider(minimum=0.1, maximum=20, step=0.05,  value=1.8, label="CFG Scale")
                    with gr.Group():
                        with gr.Row():
                            single_prompt = gr.Checkbox(label="Single Prompt Mode", value=False, visible=False)
                            prompt_fixed_ratio = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="Prompt Fixed Ratio")
                            tensor_interpolation_slerp = gr.Checkbox(label="tensor_interpolation_slerp", value=True, visible=False)
                    inp_posi = gr.Textbox(lines=2, value="1girl, beautiful", placeholder="1girl, beautiful", label="Positive Prompt")
                    with gr.Accordion("Prompt Map", open=False):
                        inp_pro_map = gr.Textbox(lines=3, value='"0": "best quality",', show_label=False)
                    inp_neg = gr.Textbox(lines=2, value="low quality, low res,", placeholder="low quality, low res,", label="Negative Prompt")
                    with gr.Accordion("LoRAs", open=True):
                        with gr.Group():
                            with gr.Row():
                                inp_lora1 = gr.Dropdown(choices=lora_files, label="Lora1", scale=3)
                                inp_lora1_step = gr.Slider(minimum=0.1, maximum=3, step=0.05, value=1.0, label="LoRA1 Scale", scale=1)
                        with gr.Group():
                            with gr.Row():
                                inp_lora2 = gr.Dropdown(choices=lora_files, label="Lora2", scale=3)
                                inp_lora2_step = gr.Slider(minimum=0.1, maximum=3, step=0.05, value=1.0, label="LoRA2 Scale", scale=1)
                        with gr.Group():
                            with gr.Row():
                                inp_lora3 = gr.Dropdown(choices=lora_files, label="Lora3", scale=3)
                                inp_lora3_step = gr.Slider(minimum=0.1, maximum=3, step=0.05, value=1.0, label="LoRA3 Scale", scale=1)
                        with gr.Group():
                            with gr.Row():
                                inp_lora4 = gr.Dropdown(choices=lora_files, label="Lora4", scale=3)
                                inp_lora4_step = gr.Slider(minimum=0.1, maximum=3, step=0.05, value=1.0, label="LoRA4 Scale", scale=1)
                    with gr.Accordion("Motion Lora", open=False):
                        with gr.Row():
                            mo1_ch = gr.Dropdown(choices=ml_files, label="MotionLoRA1", scale=3)
                            mo1_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=0.8, label="Motion LoRA1 scale")
                        with gr.Row():
                            mo2_ch = gr.Dropdown(choices=ml_files, label="MotionLoRA2", scale=3)
                            mo2_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=0.8, label="Motion LoRA2 scale")
            with gr.Column():
                with gr.Group():
                    with gr.Accordion("Special Effects", open=True):
                        ip_ch = gr.Checkbox(label="IPAdapter", value=False)
                        with gr.Row():
                            ip_image = gr.Image(height=128, type="pil", interactive=False)
                            with gr.Column():
                                ip_scale = gr.Slider(minimum=0, maximum=2, step=0.1, value=1.0, label="scale", interactive=False)
                                ip_image_ratio = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.8, label="Image Fixed Ratio", interactive=False)
                                ip_type = gr.Radio(choices=ip_choice, label="Type", value="plus_face", interactive=False)

                        ref_ch = gr.Checkbox(label="Ref Only", value=False)
                        with gr.Row() as ref_grp:
                            ref_image = gr.Image(height=128, type="pil", interactive=False)
                            with gr.Column():
                                ref_attention = gr.Slider(minimum=0, maximum=1, step=0.1, value=1.0, label="attention auto machine weight", interactive=False)
                                ref_gn = gr.Slider(minimum=0, maximum=1, step=0.1, value=1.0, label="gn auto machine weight", interactive=False)
                                ref_weight = gr.Slider(minimum=0, maximum=1, step=0.1, value=1.0, label="Ref Only Weight", interactive=False)
                                
                        with gr.Row() as mask_grp:
                            with gr.Column():
                                with gr.Group():
                                    mask_ch1 = gr.Checkbox(label="Mask(Inpaint)", value=False)
                                    mask_target = gr.Textbox(lines=1, value="person", show_label=False)
                            mask_type1 = gr.Dropdown(choices=mask_type_choice, label="Type", value="Original" )
                            mask_padding1 = gr.Slider(minimum=-100, maximum=100, step=1, value=0, label="Mask Padding")
                        with gr.Row() as i2i_grp:
                            i2i_ch = gr.Checkbox(label="Image2Image", value=False)
                            i2i_scale = gr.Slider(minimum=0.05, maximum=5,  step=0.05, value=0.7, label="Denoising Strength")
                        with gr.Row() as ad_grp:
                            ad_ch = gr.Checkbox(label="AimateDiff Controlnet", value=True)
                            ad_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=0.25, label="AnimateDiff Controlnet Weight")
                        with gr.Row() as op_grp:
                            op_ch = gr.Checkbox(label="Open Pose", value=True)
                            op_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=0.9, label="Open Pose Weight")
                        with gr.Row() as dp_grp:
                            dp_ch = gr.Checkbox(label="Depth", value=False)
                            dp_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=0.5, label="Depth Weight", interactive=False)
                        with gr.Row() as la_grp:
                            la_ch = gr.Checkbox(label="Lineart", value=False)
                            la_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=0.5, label="Lineart Weight", interactive=False)
                        with gr.Row() as me_grp:
                            me_ch = gr.Checkbox(label="Mediapipe Face", value=False)
                            me_scale = gr.Slider(minimum=0, maximum=2,  step=0.05, value=0.5, label="Mediapipe Face Weight", interactive=False)
                        with gr.Row() as refine_grp:
                            refine = gr.Checkbox(label="Refine", value=False)
                            re_scale = gr.Slider(minimum=0.05, maximum=2,  step=0.05, value=0.75, label="Tile-upscale", interactive=False)
                            re_interpo = gr.Slider(minimum=1, maximum=3,  step=1, value=1, label="Interporation Mulitiplier", visible=False, interactive=False)

                 #   inp2 = gr.Dropdown(choices=result_list, info="please select", label="Config")
                    with gr.Row():
                        delete_if_exists = gr.Checkbox(label="Delete cache")
                        test_run = gr.Checkbox(label="Test Run", value=True)
                        
        with gr.Row():
            # with gr.Column():
            with gr.Group():
                o_status = gr.Label(value="Not Started Yet", label="Status", scale=5)
                # output=gr.Video(container=True)
                with gr.Row():
                # data_sets=gr.Dataset(components=[output], samples=[], label="Result Videos")
                    # examples = gr.Examples(examples=video_paths, inputs=input_hidden, outputs=output, cache_examples=False)
                    o_original = gr.Video(width=128, label="Original Video", scale=1)
                    o_mask = gr.Video(width=128, label="Mask", scale=1)
                    o_openpose = gr.Video(width=128, label="Open Pose", scale=1)
                    o_depth = gr.Video(width=128, label="Depth", scale=1)
                    o_lineart = gr.Video(width=128, label="Line Art", scale=1)
                with gr.Row():
                    o_mediaface = gr.Video(width=128, label="Mediapipe Face", scale=1)
                    o_front = gr.Video(width=128, label="Front Video", scale=1)
                    o_front_refine = gr.Video(width=128, label="Front Video (Refined)", scale=1)
                    o_composite = gr.Video(width=128, label="Composite Video", scale=1)
                    o_final = gr.Video(width=128, label="Generated Video", scale=1)             

        btn.click(fn=execute_wrapper,
                  inputs=[tab_select, tab_select2, url, dl_video, t_name, t_length, t_width, t_height, fps,
                          inp_model, inp_vae, 
                          inp_mm, inp_context, inp_sche, 
                          inp_lcm, inp_hires, low_vr,
                          inp_step, inp_cfg, seed,
                          single_prompt, prompt_fixed_ratio,tensor_interpolation_slerp,
                          inp_posi, inp_pro_map, inp_neg, 
                          inp_lora1, inp_lora1_step,
                          inp_lora2, inp_lora2_step,
                          inp_lora3, inp_lora3_step,
                          inp_lora4, inp_lora4_step,
                          mo1_ch, mo1_scale,
                          mo2_ch, mo2_scale,
                          ip_ch, ip_image, ip_scale, ip_type, ip_image_ratio,
                          mask_ch1, mask_target, mask_type1, mask_padding1,
                          ad_ch, ad_scale, op_ch, op_scale,
                          dp_ch, dp_scale, la_ch, la_scale,
                          me_ch, me_scale, i2i_ch, i2i_scale,
                          ref_ch, ref_image, ref_attention, ref_gn, ref_weight,
                          refine, re_scale, re_interpo,
                          delete_if_exists, test_run],
                  # outputs=[o_status, output, data_sets, btn])

                  outputs=[o_status, o_original, o_mask, o_lineart, o_depth, o_openpose, o_mediaface, o_front, o_front_refine, o_composite, o_final, btn])

        ip_ch.change(fn=change_ip, inputs=[ip_ch], outputs=[ip_ch, ip_image, ip_scale, ip_type, ip_image_ratio])        
        ad_ch.change(fn=change_cn, inputs=[ad_ch], outputs=[ad_ch, ad_scale])
        op_ch.change(fn=change_cn, inputs=[op_ch], outputs=[op_ch, op_scale])
        dp_ch.change(fn=change_cn, inputs=[dp_ch], outputs=[dp_ch, dp_scale])
        la_ch.change(fn=change_cn, inputs=[la_ch], outputs=[la_ch, la_scale])
        me_ch.change(fn=change_cn, inputs=[me_ch], outputs=[me_ch, me_scale])
        refine.change(fn=change_re, inputs=[refine], outputs=[refine, re_scale, re_interpo])
        i2i_ch.change(fn=change_cn, inputs=[i2i_ch], outputs=[i2i_ch, i2i_scale])
        ref_ch.change(fn=change_ref, inputs=[ref_ch], outputs=[ref_ch, ref_image, ref_attention, ref_gn, ref_weight])
        v2v_tab.select(fn=select_v2v, outputs=[tab_select, btn, mask_grp, i2i_grp, ad_grp, op_grp, dp_grp, la_grp, me_grp, test_run, delete_if_exists])
        t2v_tab.select(fn=select_t2v, outputs=[tab_select, btn, mask_grp, i2i_grp, ad_grp, op_grp, dp_grp, la_grp, me_grp, test_run, delete_if_exists])

        data_tab.select(fn=select_data, outputs=[tab_select2])
        url_tab.select(fn=select_url, outputs=[tab_select2])

        # data_sets.select(fn=select_video, outputs=[output])
        
    iface.queue()
    iface.launch(share=True)

    while True:
        pass

launch()


