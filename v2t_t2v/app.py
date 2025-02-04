import streamlit as st
from   v2t_models.v2t_model   import  analyze_video
from   video_utils            import (get_video_duration,
                                      get_default_times,
                                      time_to_seconds,
                                      cut_video_moviepy)


tab1, tab2 = st.tabs(["Video-2-Text", "Text-2-Frame"])

# Video-2-Text
with tab1:
    st.header("🎬 Video-2-Text")
    video_file                      = st.file_uploader("📂 mp4 비디오 파일 업로드", 
                                                       type = [".mp4"])
    temp_video_path, video_duration = get_video_duration(video_file) \
                                      if video_file else (None, None)
                                      
    start_time_str , end_time_str   = get_default_times(video_duration)
    start_time_str                  = st.text_input("⏳ 시작 시간 (HH:MM:SS:FF)",
                                                    value=start_time_str)
    end_time_str                    = st.text_input("⏳ 종료 시간 (HH:MM:SS:FF)", 
                                                    value=end_time_str)
    
    if "clipped_video_path" not in st.session_state:
        st.session_state.clipped_video_path = None
        
    if video_file and st.button("✂️ 비디오 처리하기"):
        try:
            start_time         = time_to_seconds(start_time_str)
            end_time           = time_to_seconds(end_time_str)
            
            clipped_video_path = cut_video_moviepy(temp_video_path,
                                                   start_time,
                                                   end_time)
            
            st.session_state.clipped_video_path = clipped_video_path
            
            st.video(clipped_video_path)
            st.success("✅ 비디오 클립이 성공적으로 생성되었습니다!")
            
        except Exception as e:
            st.error(f"❌ 비디오 처리 오류: {e}")
            
    if st.session_state.clipped_video_path and st.button("📜 비디오 내용 분석하기"):
        try:
            result_text = analyze_video(st.session_state_clipped_video_path)
            
            st.success("✅ 추론 완료!")
            st.text_area("📝 생성된 텍스트\n", result_text, height=250)
            
        except Exception as e:
            st.error(f"❌ 추론 중 오류 발생: {e}")
            
# Text-2-Frame
with tab2:
    st.header("🔎 Text-2-Frame")
    input_text = st.text_input("🔠 설명 또는 텍스트 프롬프트 입력:")
    st.write("🚧 이 기능은 아직 구현되지 않았습니다.")
