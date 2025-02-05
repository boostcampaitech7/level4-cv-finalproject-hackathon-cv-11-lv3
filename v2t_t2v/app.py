from   v2t_models.translation_model import (DL_translation,
                                            API_translator)
from   v2t_models.v2t_model         import  analyze_video
from   video_utils                  import (get_video_duration,
                                            get_default_times,
                                            time_to_seconds,
                                            cut_video_moviepy)
import streamlit as st

if "clipped_video_path" not in st.session_state:
    st.session_state.clipped_video_path = None

if "last_uploaded_video" not in st.session_state:
    st.session_state.last_uploaded_video = None

tab1, tab2 = st.tabs(["Video-2-Text", "Text-2-Frame"])

# Video-2-Text
with tab1:
    st.header("🎬 Video-2-Text")
    video_file = st.file_uploader("📂 mp4 비디오 파일 업로드", type=[".mp4"])
    
    if video_file:
        if st.session_state.last_uploaded_video != video_file:
            st.session_state.clipped_video_path  = None
            st.session_state.last_uploaded_video = video_file

        temp_video_path, video_duration          = get_video_duration(video_file)
        if video_duration:
            video_duration_int = int(video_duration)

            start_time         = st.slider("⏳ 시작 시간 (초)", 
                                           min_value=0, 
                                           max_value=video_duration_int, 
                                           value=0, step=1)
            end_time           = st.slider("⏳ 종료 시간 (초)", 
                                           min_value=0, 
                                           max_value=video_duration_int, 
                                           value=video_duration_int, step=1)
            
            if st.button("✂️ 비디오 처리하기"):
                try:
                    # cut_video_moviepy에서 조정된 end_time 받을 수 있음
                    clipped_video_path, used_end_time   = cut_video_moviepy(temp_video_path,
                                                                           start_time,
                                                                           end_time)
                    
                    # ✅ 비디오 클립을 유지 (새로운 비디오가 업로드되기 전까지)
                    st.session_state.clipped_video_path = clipped_video_path
                    st.success("✅ 비디오 클립이 성공적으로 생성되었습니다!")
                    
                    if used_end_time != end_time:
                        st.warning(f"영상 길이를 초과하여 종료 시간을 {used_end_time}초로 조정했습니다.")

                except Exception as e:
                    st.error(f"❌ 비디오 처리 오류: {e}")

    # ✅ 기존 비디오 클립 유지
    if st.session_state.clipped_video_path:
        st.video(st.session_state.clipped_video_path)

    if st.session_state.clipped_video_path and st.button("📜 비디오 내용 분석하기"):
        with st.spinner("추론 중... 잠시만 기다려 주세요!"):
            try:
                result_text     = analyze_video(st.session_state.clipped_video_path)
                translated_text = API_translator(response = result_text, 
                                                 kr2en = False)
                st.success("✅ 추론 완료!")
                st.text_area("📝 생성된 텍스트\n", translated_text, height=250)

            except Exception as e:
                st.error(f"❌ 추론 중 오류 발생: {e}")

# Text-2-Frame
with tab2:
    st.header("🔎 Text-2-Frame")
    input_text       = st.text_input("🔠 찾고싶은 영상의 설명을 입력해주세요!:")
    translated_input = API_translator(response = input_text,
                                      kr2en    = True)
    st.text_area("번역, token_갯수",translated_input)
    video_files      = st.file_uploader("📂 mp4 비디오 파일 업로드",
                                        type                  = [".mp4"],
                                        accept_multiple_files = True)

    st.write("🚧 이 기능은 아직 구현되지 않았습니다.")

