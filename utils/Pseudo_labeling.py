#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import shutil
import logging
import warnings
import argparse
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def parse_args():
    parser = argparse.ArgumentParser(description="Pseudo Labeling & Outlier Re-Inference Script")

    parser.add_argument("--inference", action="store_true",
                        help="슈도 레이블링(Pseudo Labeling)을 실행합니다.")
    parser.add_argument("--re_inference", action="store_true",
                        help="이미 생성된 JSON 중 outlier(r)가 있는 비디오만 재추론합니다.")

    parser.add_argument("--video_dir", type=str, default="./videos",
                        help="비디오가 저장되어있는 폴더 경로")
    parser.add_argument("--done_dir", type=str, default="./done",
                        help="처리 완료 후 비디오를 옮길 경로")
    parser.add_argument("--output_dir", type=str, default="./json_output",
                        help="추론 결과 JSON 파일을 저장할 폴더 경로")
    parser.add_argument("--merge_json_dir", type=str, default="./json_output",
                        help="기존에 생성된 JSON 폴더(Outlier 탐색용) 혹은 Merge 대상 폴더")
    parser.add_argument("--merge_output_file", type=str, default="merged_data.json",
                        help="JSON 병합 결과를 저장할 경로 (옵션)")

    parser.add_argument("--exclude_last_seconds", type=int, default=30,
                        help="씬 검출 시 마지막 N초 구간을 제외 (기본=30)")
    parser.add_argument("--pyscene_threshold", type=float, default=30.0,
                        help="PySceneDetect ContentDetector threshold (기본=30)")
    parser.add_argument("--num_segments", type=int, default=8,
                        help="한 씬에서 프레임을 몇 덩어리로 샘플링할지 (기본=8)")
    parser.add_argument("--input_size", type=int, default=448,
                        help="모델에 들어갈 이미지 크기 (기본=448)")

    parser.add_argument("--duration_mode", type=str, default="subtract",
                        choices=["scene","subtract","full"],
                        help=("duration을 계산하는 방식 선택:"
                              " scene=마지막 씬 종료시점,"
                              " subtract=전체 길이에서 exclude_last_seconds 뺀 값,"
                              " full=영상 전체 길이"))
    
    args = parser.parse_args()

    if not args.inference and not args.re_inference:
        parser.error("하나 이상의 작업(--inference, --re_inference)을 지정해야 합니다.")

    return args

def read_json(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return None
        return json.loads(content)


def find_outlier(data):
    if isinstance(data, dict):
        if not data:
            return "Empty object"
        outliers = [find_outlier(value) for value in data.values()]
        return next((outlier for outlier in outliers if outlier), None)
    
    elif isinstance(data, list):
        if not data:
            return "Empty list"
        outliers = [find_outlier(value) for value in data]
        return next((outlier for outlier in outliers if outlier), None)
    
    elif isinstance(data, str):
        if data == "r":
            return "'r' found"
    
    return None

def get_videos(directory, extensions=('.mp4','.avi','.mkv','.mov')):
    video_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                path = os.path.join(root, file)
                video_paths.append(path)
    logging.info(f"Found {len(video_paths)} videos in {directory}")
    return video_paths

def detect_scenes(video_path, threshold=30.0, exclude_last_seconds=30):
    """
    PySceneDetect로 씬을 검출, 마지막 exclude_last_seconds 구간은 무시.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        total_duration = total_frames / fps

        valid_scenes = []
        for start, end in scene_list:
            start_sec = start.get_seconds()
            end_sec   = end.get_seconds()
            if end_sec <= total_duration - exclude_last_seconds:
                valid_scenes.append((start_sec, end_sec))

        return valid_scenes, total_duration
    except Exception as e:
        logging.error(f"Error detecting scenes: {e}")
        return [], 0.0

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_index(start_sec, end_sec, fps, num_segments=8):
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec * fps)
    seg_size = float(end_frame - start_frame) / num_segments
    indices = [
        int(start_frame + (seg_size / 2) + round(seg_size * idx))
        for idx in range(num_segments)
    ]
    return indices

def load_frames(video_path, start_sec, end_sec, fps, num_segments=8, input_size=448):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        transform = build_transform(input_size)
        frame_indices = get_index(start_sec, end_sec, fps, num_segments)

        pixel_values_list = []
        for idx in frame_indices:
            if idx < 0 or idx >= len(vr):
                continue
            frame = Image.fromarray(vr[idx].asnumpy())
            frame = transform(frame)
            pixel_values_list.append(frame)

        if not pixel_values_list:
            return None
        return torch.stack(pixel_values_list, dim=0)
    except Exception as e:
        logging.error(f"Error loading frames: {e}")
        return None

def load_model(model_path="OpenGVLab/InternVL2_5-8B-MPO"):
    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        model = model.eval().cuda()
        logging.info("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)

def generate_caption(pixel_values, question, generation_config, model, tokenizer):
    try:
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        frames_text = ''.join([f'Frame{i+1}: <image>\n' for i in range(pixel_values.size(0))])
        query = frames_text + question

        response, history = model.chat(
            tokenizer,
            pixel_values,
            query,
            generation_config,
            history=None,
            return_history=True
        )
        return response
    except Exception as e:
        logging.error(f"Error generating caption: {e}")
        return ""
    finally:
        del pixel_values
        torch.cuda.empty_cache()

def compute_duration_ms(scenes, total_duration, exclude_last_seconds, mode="subtract"):
    if not scenes:
        return 0

    if mode == "scene":
        last_end_sec = scenes[-1][1]
        return int(last_end_sec * 1000)
    elif mode == "subtract":
        adjusted = total_duration - exclude_last_seconds
        if adjusted < 0:
            adjusted = 0
        return int(adjusted * 1000)
    else:
        return int(total_duration * 1000)

def build_json_data(video_id, scenes, captions, total_duration, exclude_last_seconds, duration_mode):
    """
    scenes   : [(start_sec, end_sec), ...]
    captions : ["...", "..."]
    """
    if not scenes:
        return {
            video_id: {
                "duration": 0,
                "timestamps": [],
                "sentences": []
            }
        }
    
    timestamps = []
    for (start_sec, end_sec) in scenes:
        start_ms = int(start_sec * 1000)
        end_ms   = int(end_sec   * 1000)
        timestamps.append([start_ms, end_ms])
   
    duration_ms = compute_duration_ms(
        scenes=scenes,
        total_duration=total_duration,
        exclude_last_seconds=exclude_last_seconds,
        mode=duration_mode
    )

    return {
        video_id: {
            "duration": duration_ms,
            "timestamps": timestamps,
            "sentences": captions
        }
    }

def pseudo_label_video(
    video_path,
    model,
    tokenizer,
    question,
    output_dir,
    done_dir,
    pyscene_threshold=30.0,
    exclude_last_seconds=30,
    num_segments=8,
    input_size=448,
    generation_config=None,
    duration_mode="scene"
):
    if generation_config is None:
        generation_config = {
            "max_new_tokens": 256,
            "do_sample": True,
            "top_k": 30,
            "top_p": 0.9,
            "temperature": 0.1
        }

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_id = video_name[-11:] if len(video_name) >= 11 else video_name

    scenes, total_duration = detect_scenes(video_path,
                                           threshold=pyscene_threshold,
                                           exclude_last_seconds=exclude_last_seconds)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())

    captions = []
    for (start_sec, end_sec) in tqdm(scenes, desc=f"Processing {video_id} Scenes"):
        pixel_values = load_frames(
            video_path, start_sec, end_sec,
            fps, num_segments=num_segments, input_size=input_size
        )
        if pixel_values is None:
            captions.append("")
            continue

        cap = generate_caption(pixel_values, question, generation_config, model, tokenizer)
        captions.append(cap)

    final_json = build_json_data(
        video_id=video_id,
        scenes=scenes,
        captions=captions,
        total_duration=total_duration,
        exclude_last_seconds=exclude_last_seconds,
        duration_mode=duration_mode
    )
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{video_id}.json")
    with open(json_path, "w", encoding='utf-8') as jf:
        json.dump(final_json, jf, indent=4, ensure_ascii=False)
    logging.info(f"Saved JSON: {json_path}")

    os.makedirs(done_dir, exist_ok=True)
    shutil.move(video_path, os.path.join(done_dir, os.path.basename(video_path)))
    logging.info(f"Moved video to {done_dir}: {video_path}")

    return json_path

def merge_json_files(input_dir, output_path):
    merged_data = {}
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    for jf in json_files:
        path = os.path.join(input_dir, jf)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                merged_data.update(data)
        except Exception as e:
            logging.warning(f"Cannot parse {path}: {e}")

    with open(output_path, 'w', encoding='utf-8') as outf:
        json.dump(merged_data, outf, indent=4, ensure_ascii=False)
    logging.info(f"Merged JSON saved to {output_path}")

def re_inference(json_dir, video_dir, output_dir, done_dir,
                 question, model, tokenizer,
                 pyscene_threshold=30.0, exclude_last_seconds=30,
                 num_segments=8, input_size=448, generation_config=None,
                 duration_mode="scene"):
    if generation_config is None:
        generation_config = {
            "max_new_tokens": 256,
            "do_sample": True,
            "top_k": 30,
            "top_p": 0.9,
            "temperature": 0.1
        }

    problem_files = []
    for fn in os.listdir(json_dir):
        if not fn.endswith(".json"):
            continue
        jp = os.path.join(json_dir, fn)
        data = read_json(jp)
        if not data:
            continue
        problem = find_outlier(data)
        if problem:
            problem_files.append((jp, problem))

    outlier_videos = []
    for (json_path, issue) in problem_files:
        if issue:
            base = os.path.basename(json_path)
            video_id = base.replace(".json","")
            for vf in os.listdir(video_dir):
                if vf.endswith(".mp4"):
                    if vf[-15:-4] == video_id:
                        outlier_videos.append(os.path.join(video_dir,vf))
                    elif vf[-11:] == video_id:
                        outlier_videos.append(os.path.join(video_dir,vf))

    if not outlier_videos:
        logging.info("No outlier videos found.")
        return

    for vid in tqdm(outlier_videos, desc="Re-inference outlier videos"):
        pseudo_label_video(
            video_path=vid,
            model=model,
            tokenizer=tokenizer,
            question=question,
            output_dir=output_dir,
            done_dir=done_dir,
            pyscene_threshold=pyscene_threshold,
            exclude_last_seconds=exclude_last_seconds,
            num_segments=num_segments,
            input_size=input_size,
            generation_config=generation_config,
            duration_mode=duration_mode
        )


def main():
    args = parse_args()

    question = (
    "Please carefully watch the following video scene and describe it in as much detail as possible. "
    "Focus on the following aspects:\n"
    "1) If there are any people present, provide a thorough description of them (physical appearance, clothing, expressions, etc.), but do not describe them in list form.\n"
    "2) If there are no people, do NOT apologize or disclaim; instead, describe the environment or objects in thorough detail as part of a flowing narrative.\n"
    "3) Describe each person's actions or movements in detail: gestures, body language, eye contact, posture, and any physical interactions with objects, ensuring they are seamlessly integrated into the overall description.\n"
    "4) Provide context for the situation and setting (indoors/outdoors, lighting, time of day, weather, etc.) and explain what is happening, making it part of a unified story.\n"
    "5) Include any relevant objects, environmental details, or interactions that add meaning. For example, mention notable items and how they are positioned or used, incorporating them into the broader narrative.\n"
    "6) Avoid repetitive phrases like 'the scene...' at the beginning of every sentence. Instead, aim for a natural flow of description.\n"
    "7) Be thorough and vivid, but clarify if certain details are assumptions rather than visible facts.\n"
    "8) Finally, combine all details into a single, cohesive, and natural narrative. Avoid listing information as separate points and ensure that the description reads like a story or article."
    )

    generation_config = {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 3,
        "early_stopping": True,     
        "no_repeat_ngram_size": 3,  
        "length_penalty": 1.0,
        "top_k": 50,
        "top_p": 0.7,
        "temperature": 0.4,
    }

    model, tokenizer = load_model()

    if args.inference:
        videos = get_videos(args.video_dir)
        for v in tqdm(videos, desc="Inference videos"):
            pseudo_label_video(
                video_path=v,
                model=model,
                tokenizer=tokenizer,
                question=question,
                output_dir=args.output_dir,
                done_dir=args.done_dir,
                pyscene_threshold=args.pyscene_threshold,
                exclude_last_seconds=args.exclude_last_seconds,
                num_segments=args.num_segments,
                input_size=args.input_size,
                generation_config=generation_config,
                duration_mode=args.duration_mode
            )

        if args.merge_output_file:
            merge_json_files(args.output_dir, args.merge_output_file)

    if args.re_inference:
        re_inference(
            json_dir=args.merge_json_dir,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            done_dir=args.done_dir,
            question=question,
            model=model,
            tokenizer=tokenizer,
            pyscene_threshold=args.pyscene_threshold,
            exclude_last_seconds=args.exclude_last_seconds,
            num_segments=args.num_segments,
            input_size=args.input_size,
            generation_config=generation_config,
            duration_mode=args.duration_mode
        )

        if args.merge_output_file:
            merge_json_files(args.output_dir, args.merge_output_file)

if __name__ == "__main__":
    main()
