from moviepy.editor import VideoFileClip

def extract_audio(input_video, output_audio):
    # 動画から音声を抽出
    video_clip = VideoFileClip(input_video)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio, codec='pcm_s16le', ffmpeg_params=["-ac", "2"])

def create_video_with_audio(input_video, audio_file, output_video):
    # 動画と音声を結合して新しい動画を作成
    video_clip = VideoFileClip(input_video)
    audio_clip = AudioFileClip(audio_file)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_video, codec='libx264', audio_codec='aac')

def create_video(org_video, new_video, output_video_path):
    org_video_clip = VideoFileClip(org_video)
    video_clip = VideoFileClip(new_video)
    org_video_clip.audio = org_video_clip.audio.subclip(0, video_clip.duration)
    video_clip = video_clip.set_audio(org_video_clip.audio)
    video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', logger=None)

