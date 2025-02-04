import streamlit as st
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import random


def model_inference(input_data):
    return "This is the generated text from the model based on the video or frame input."



def cut_video_moviepy(input_path, start_time, end_time, output_path):
    try:
        print(f"Processing video file: {input_path}")
        with VideoFileClip(input_path) as video:
            print(f"Video duration: {video.duration} seconds")
            print(f"Video FPS: {video.fps}")
            print(f"Video size: {video.size}")

            if start_time >= video.duration or end_time > video.duration:
                raise ValueError("Start or end time exceeds video duration.")
            if start_time >= end_time:
                raise ValueError("Start time must be less than end time.")

            subclip = video.subclipped(start_time, end_time)
            subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    except Exception as e:
        raise RuntimeError(f"MoviePy failed: {e}")

tab1, tab2 = st.tabs(["Video-2-Text", "Text-2-Frame"])

# Video-2-Text
with tab1:
    st.header("Video-2-Text")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    start_time_str = st.text_input("Start timestamp (HH:MM:SS:FF)", value="00:00:10:00")
    end_time_str = st.text_input("End timestamp (HH:MM:SS:FF)", value="00:00:30:00")

    if "clipped_video_path" not in st.session_state:
        st.session_state.clipped_video_path = None

    if video_file and st.button("Process Video"):
        try:
            start_time = time_to_seconds(start_time_str)
            end_time = time_to_seconds(end_time_str)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(video_file.read())
                temp_video_path = temp_file.name

            clipped_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            cut_video_moviepy(temp_video_path, start_time, end_time, clipped_temp_file.name)

            st.session_state.clipped_video_path = clipped_temp_file.name
            st.video(clipped_temp_file.name)
            st.success("Video clipped successfully!")

        except Exception as e:
            st.error(f"Error processing video: {e}")

    if st.session_state.clipped_video_path and st.button("Run Inference on Clipped Video"):
        try:
            video_manager = VideoManager([st.session_state.clipped_video_path])
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector())

            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()

            result_text = model_inference(scene_list)

            st.success("Inference Completed!")
            st.text_area("Generated Text:", result_text, height=200)
        except Exception as e:
            st.error(f"Error during inference: {e}")

# Text-2-Frame
with tab2:
    st.header("Text-2-Frame")
    input_text = st.text_input("Enter a description or text prompt:")
