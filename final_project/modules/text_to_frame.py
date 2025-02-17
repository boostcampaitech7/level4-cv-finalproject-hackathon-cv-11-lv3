import streamlit            as     st
from   models.translation   import Translator
from   models.frame_extract import FrameExtractor 

VIDEO_DIR          = "/data/ephemeral/home/videos_movieclips"
NPZ_FILE           = "/data/ephemeral/home/movie_clip_AnglE_UAE_Large_V1_features.npz"
OUTPUT_DIR         = "/data/ephemeral/home/extracted_frames"
VIDEO_STORAGE_PATH = "/data/ephemeral/home/videos"

class Text2FramePage:
    def run(self):
        st.title("🔎 Text-2-Frame")
        st.write("찾고 싶은 영상의 설명을 입력하고, '프레임 추출 시작' 버튼을 눌러주세요")
        
        input_text = st.text_input("🔠 찾고 싶은 영상의 설명을 한글로 입력해주세요:")

        mode       = st.radio("번역 모드를 선택하세요:", ("API", "DL"))
        st.write("💡 API에서 문제가 발생하면 DL 모드를 사용해보세요! (20~30초 정도 더 소요됩니다.)")
        
        if st.button("⏳ 프레임 추출 시작"):
            status_text = st.empty()
            
            with Translator(kr2en = True, mode  = mode) as t:
                translated_text = t.translate(input_text)
            print(translated_text)
            
            frame_extractor = FrameExtractor(video_dir1        = VIDEO_DIR,
                                             video_dir2        = VIDEO_STORAGE_PATH,
                                             npz_file          = NPZ_FILE,
                                             output_dir        = OUTPUT_DIR,
                                             top_k             = 5,
                                             sampling_interval = 500)
            
            status_text.text("🔍 프레임 추출 중...")
            final_results = frame_extractor.extract_frames(translated_text)
            status_text.text("✅ 추천 장면 추출 완료!")
            st.success("🎉 프레임 추출이 완료되었습니다!")
        
            if final_results:
                st.write("## 추천 프레임")
                for idx, result in enumerate(final_results, start=1):
                    st.markdown(f"### {idx}. Movie ID: **{result['movie_id']}**")
                    st.write(f"**시간 범위:** {result['time_range']}")
                    st.write(f"**최적 시간:** {result['best_time']/1000:.1f} 초")
                    st.write(f"**유사도:** {result['angle_similarity']:.4f}")
                    st.image(result["output_frame_path"])
                    st.markdown("---")
            else:
                st.write("추출된 결과가 없습니다.")
            
