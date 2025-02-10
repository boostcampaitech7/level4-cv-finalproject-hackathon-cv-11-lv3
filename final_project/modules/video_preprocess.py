import streamlit     as st
import tempfile
import os
import time
from   models.analyze       import AnalyzeVideo
from   models.add_embedding import EmbeddingProcessor

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
                
                temp_dir           = tempfile.gettempdir()
                
                for uploaded in uploaded_files:
                    origin_filename = uploaded.name
                    temp_path       = os.path.join(temp_dir, origin_filename)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                        
                    video_paths.append(temp_path)
                    original_filenames[temp_path] = origin_filename
                    
                time.sleep(1)
                
                st.write("📜 비디오 추론 시작")
                status_text.text("추론 중... 잠시만 기다려 주세요!")
                
                with AnalyzeVideo(use_audio  = False, 
                                  num_seg    = 3, 
                                  batch_size = 11) as av:
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
                