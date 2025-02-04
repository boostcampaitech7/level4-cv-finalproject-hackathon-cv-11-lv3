import os
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from tqdm import tqdm
import cv2

def process_videos_in_folder(folder_path, output_folder, threshold=30.0, trim_last_seconds=30):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(folder_path, video_file)
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_seconds = total_frames / fps if fps > 0 else 0
        cap.release()

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()

        trimmed_scene_list = []
        for start_time, end_time in scene_list:
            start_sec = start_time.get_seconds()
            end_sec = end_time.get_seconds()

            if end_sec <= duration_seconds - trim_last_seconds:
                trimmed_scene_list.append((start_time, end_time))

        split_video_ffmpeg(
            input_video_path=video_path,
            scene_list=trimmed_scene_list,
            output_dir=output_folder,
            show_progress=True,
            output_file_template=f"{video_id}_$SCENE_NUMBER.mp4"
        )

        print(f"Processing completed for: {video_file}")

input_folder = "/data/ephemeral/home/videos_movieclips_461"
output_folder = "./videos"

process_videos_in_folder(input_folder, output_folder)
