import os
import time
import streamlit            as     st
from   models.analyze       import AnalyzeVideo
from   models.add_embedding import EmbeddingProcessor

VIDEO_STORAGE_PATH = "/data/ephemeral/home/videos"

def save_uploaded_file(uploaded_file, save_dir = VIDEO_STORAGE_PATH):
    
    os.makedirs(save_dir, exist_ok = True)
    
    base_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
    
    if len(base_name) < 20:
        video_id = base_name
        
    else:
        video_id = base_name[-15:-4]
        
    new_filename         = f"{video_id}.mp4"
    save_path            = os.path.join(save_dir, new_filename)
    
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
        
    print(f"📂 저장된 파일: {uploaded_file.name} -> {save_path}")
    return save_path, new_filename

class VideoPreprocessingPage:
    def run(self):
        st.title("🎬 Video PreProcessing")
        st.write("새로운 비디오를 넣어주세요!")
        
        uploaded_files = st.file_uploader("📂 mp4 비디오 파일 업로드",
                                          type=["mp4"],
                                          accept_multiple_files=True,
                                          )
        
        if uploaded_files and len(uploaded_files) <= 10:
            if st.button("⏳ 비디오 프로세싱 시작"):
                status_text        = st.empty()
                video_paths        = []
                original_filenames = {}
                
                status_text.text("📂 비디오 저장중")
                
                for uploaded in uploaded_files:
                    save_path, new_filename       = save_uploaded_file(uploaded)
                    
                    video_paths.append(save_path)
                    
                    original_filenames[save_path] = new_filename
                    
                    print(f"📂 원본 파일명: {uploaded.name} -> 저장된 파일명: {new_filename}")
                    
                time.sleep(1)
                
                st.write("📜 비디오 추론 시작")
                status_text.text("추론 중... 잠시만 기다려 주세요!")
                
                with AnalyzeVideo(use_audio  = False, num_seg    = 3, batch_size = 11) as av:
                    for idx, video_path in enumerate(video_paths):
                        status_text.text(f"📊 분석 진행 중: {idx+1}/{len(video_paths)} - {original_filenames[video_path]}")
                        
                        av.fast_batch_analyze(video_paths = [video_path],
                                              output_path = "/data/ephemeral/home/json_output")
                        
                status_text.text("✅ 모든 비디오가 처리되었습니다!")
                
                status_text.text("📜 NPZ 파일 병합 시작")
                embedding = EmbeddingProcessor(existing_npz_path = "/data/ephemeral/home/movie_clip_AnglE_UAE_Large_V1_features.npz", 
                                               json_folder       = "/data/ephemeral/home/json_output", 
                                               updated_npz_path  = "/data/ephemeral/home/movie_clip_AnglE_UAE_Large_V1_features.npz")
                status_text.text("✅ NPZ 파일 병합 완료")
                st.success("🎉 모든 비디오의 전처리가 완료되었습니다!")
                