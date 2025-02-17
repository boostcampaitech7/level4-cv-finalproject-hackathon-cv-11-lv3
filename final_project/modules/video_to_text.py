import tempfile
import ffmpeg
import streamlit          as     st
from   models.analyze     import AnalyzeVideo
from   models.translation import Translator

def cut_video_ffmpeg(video_path : str, start_time : int, end_time : int):
    try:
        print(f"🎬 비디오 파일 처리 중: {video_path}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out:
            output_path = temp_out.name
        (
            ffmpeg
            .input(video_path, ss=start_time, to=end_time)
            .output(output_path, codec="copy")
            .run(quiet=True, overwrite_output=True)
        )

        print("✅ 비디오 잘라내기 완료!")
        return output_path, end_time

    except Exception as e:
        print(f"❌ 비디오 컷팅 실패: {e}")
        return None, None

def get_video_duration_ffmpeg(video_file : str):
    if isinstance(video_file, str):
        temp_video_path = video_file
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            temp_video_path = temp_file.name

    try:
        probe = ffmpeg.probe(temp_video_path)
        duration = float(probe["format"]["duration"])
        return temp_video_path, duration

    except Exception as e:
        print(f"❌ 비디오 정보를 가져오는 중 오류 발생: {e}")
        return temp_video_path, 60 

def save_video_file(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        return temp_video.name

class Video2TextPage:
    def run(self):
        st.header("🎬 Video-2-Text")
        
        video_file  = st.file_uploader("📂 mp4 비디오 파일 업로드", type=[".mp4"])
        mode        = st.radio("번역 모드를 선택하세요:", ("API", "DL"))
        
        st.write("💡 API에서 문제가 발생하면 DL 모드를 사용해보세요! (20~30초 정도 더 소요됩니다.)")
        
        if video_file:
            print(video_file.name)
            temp_video_path = save_video_file(video_file)
            temp_video_path, video_duration = get_video_duration_ffmpeg(temp_video_path)

            if video_duration is None:
                st.error("❌ 비디오 길이를 가져오지 못했습니다. 올바른 파일을 업로드하세요.")
                return 

            start_time = st.slider("⏳ 시작 시간 (초)", 0, int(video_duration), 0)
            end_time   = st.slider("⏳ 종료 시간 (초)", 0, int(video_duration), int(video_duration))

            if start_time >= end_time:
                st.error("❌ 시작 시간이 종료 시간과 같거나 더 큽니다. 다시 설정하세요.")
                return
            
            if st.button("✂️ 비디오 처리하기"):
                if start_time == 0 and end_time == video_duration:
                    clipped_video_path = temp_video_path  
                    used_start_time = 0
                    used_end_time = video_duration
                    print("🔄 원본 비디오 그대로 사용")
                    
                else:
                    clipped_video_path, used_end_time = cut_video_ffmpeg(temp_video_path, 
                                                                         start_time,
                                                                         end_time)
                    used_start_time = start_time 

                if clipped_video_path:
                    clipped_duration = used_end_time - used_start_time
                    st.session_state.clipped_video_path = clipped_video_path
                    st.success(f"✅ 비디오 클립 생성 완료! ({used_start_time}초 ~ {used_end_time}초)")
                    st.session_state.clipped_duration = clipped_duration
                    
                    if used_end_time != end_time:
                        st.warning(f"⚠️ 영상 길이를 초과하여 종료 시간을 {used_end_time}초로 조정했습니다.")
                        
                else:
                    st.error("❌ 비디오 자르기 실패. 다시 시도해주세요.")

        if "clipped_video_path" in st.session_state:
            st.video(st.session_state.clipped_video_path)

            if st.button("📜 비디오 내용 분석하기"):
                clipped_duration = st.session_state.get("clipped_duration", 0)
                use_audio = False if clipped_duration > 90 else True
                print(f"🎵 **사용된 오디오 여부:** {'✅ 사용됨' if use_audio else '❌ 미사용'}")
                
                with st.spinner("추론 중... 잠시만 기다려 주세요!"):
                    try:
                        with AnalyzeVideo(use_audio = use_audio, num_seg = 32) as av:
                            response = av.analyze(video_path = st.session_state.clipped_video_path)
                        
                        with Translator(kr2en = False, mode  = mode) as t:
                            translated_text = t.translate(response = response)
                            print(translated_text)
                            
                        st.success("✅ 추론 완료!")
                        st.text_area("📝 생성된 텍스트", translated_text, height=350)

                    except Exception as e:
                        st.error(f"❌ 추론 중 오류 발생: {e}")