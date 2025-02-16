import os
import re
import time
import streamlit                        as     st
from   models.translation               import Translator
from   distribute.flask_video_processor import FlaskVideoProcessor

TIMEOUT_SECONDS = 550

class Text2FrameDistributedInference:
    def __init__(self):
        self.processor = FlaskVideoProcessor()

    @staticmethod
    def contains_korean(text: str) -> bool:
        return bool(re.search('[가-힣]', text))

    def run(self):
        st.title("🔎 Text-2-Frame 분산 추론 🚧")
        st.write("영상 설명을 입력하고, 비디오 파일을 업로드한 후 **결과 보기** 버튼을 누르거나 자동 실행을 기다려주세요.")

        input_text = st.text_input("🔠 영상 설명 (한글로 입력)")
        
        if not input_text:
            st.warning("영상 설명을 입력하세요.")
            return
        
        if not self.contains_korean(input_text):
            st.warning("설명에 한글이 포함되어야 합니다.")
            return
        
        st.success("영상 설명 입력이 확인되었습니다!")

        if "prev_input_text" not in st.session_state or st.session_state.prev_input_text != input_text:
            try:
                with Translator(kr2en=True, mode="API") as t:
                    translated_text = t.translate(input_text)
                    
                st.session_state.translated_text = translated_text
                st.session_state.prev_input_text = input_text
                
            except Exception as e:
                st.error(f"번역 오류: {e}")
                return
        else:
            translated_text = st.session_state.translated_text
            
        st.write(f"번역된 텍스트: {translated_text}")

        uploaded_videos = st.file_uploader("비디오 파일 업로드 (mp4)", type=["mp4"], accept_multiple_files=True)
        
        if not uploaded_videos:
            st.info("비디오 파일을 업로드하세요.")
            return

        file_save_info = self.processor.save_videos(uploaded_videos)
        
        for info in file_save_info:
            st.success(f"저장 완료: {info['folder']}/{info['filename']}")

        if st.button("업로드 완료 알림 전송"):
            response = self.processor.notify_server(file_save_info)
            
            for server, resp in response.items():
                
                if resp.status_code == 200:
                    st.success(f"{server}에 알림 전송 성공")
                    
                else:
                    st.error(f"{server} 알림 전송 실패 (상태 코드: {resp.status_code})")

        st.header("2. 후속 작업 결과 (process_results)")

        expected_count = len(file_save_info)
        existing_count = len([f for f in os.listdir(self.processor.RESULT_DIR) if f.endswith('.json')])
        
        st.write(f"JSON 파일 도착: {existing_count} / {expected_count}")

        if "check_start_time" not in st.session_state:
            st.session_state.check_start_time = time.time()

        elapsed_time = time.time() - st.session_state.check_start_time
        st.write(f"경과 시간: {int(elapsed_time)} 초 / {TIMEOUT_SECONDS} 초")

        status_placeholder = st.empty()  
        timeout_reached    = False  

        while True:
            existing_count = len([f for f in os.listdir(self.processor.RESULT_DIR) if f.endswith('.json')])
            status_placeholder.info(f"현재 JSON 파일 수: {existing_count} / {expected_count}")
            
            if existing_count >= expected_count:
                status_placeholder.success("모든 JSON 파일이 도착했습니다!")
                break
            if time.time() - st.session_state.check_start_time >= TIMEOUT_SECONDS:
                timeout_reached = True
                status_placeholder.warning("시간 초과: JSON 파일이 모두 도착하지 않았습니다. 진행합니다.")
                break
            time.sleep(1)

        if "extraction_run" not in st.session_state:
            if timeout_reached:
                st.info("시간이 초과되었으므로, 도착하지 않은 JSON 파일을 무시하고 진행합니다.")
                
            else:
                st.info("모든 JSON 파일이 도착하여 후속 프로세스를 실행합니다.")
                
            try:
                frame_results = self.processor.run_frame_extracting(translated_text)
                st.session_state.extraction_run = True
                st.success("후속 프로세스가 완료되었습니다!")
                
                if frame_results:
                    st.write("## 추천 프레임")
                    for idx, result in enumerate(frame_results, start=1):
                        st.markdown(f"### {idx}. Movie ID: **{result['movie_id']}**")
                        st.write(f"**시간 범위:** {result['time_range']}")
                        st.write(f"**최적 시간:** {result['best_time']/1000:.1f} s")
                        st.write(f"**유사도:** {result['angle_similarity']:.4f}")
                        st.image(result["output_frame_path"])
                        st.markdown("---")
                else:
                    st.write("후속 프로세스 결과가 없습니다.")
            except Exception as e:
                st.error(f"결과 처리 중 오류 발생: {e}")
        else:
            if st.button("결과 보기"):
                try:
                    frame_results = self.processor.run_frame_extracting(translated_text)
                    st.success("후속 프로세스가 완료되었습니다!")
                    if frame_results:
                        st.write("## 추천 프레임")
                        for idx, result in enumerate(frame_results, start=1):
                            st.markdown(f"### {idx}. Movie ID: **{result['movie_id']}**")
                            st.write(f"**시간 범위:** {result['time_range']}")
                            st.write(f"**최적 시간:** {result['best_time']/1000:.1f} s")
                            st.write(f"**유사도:** {result['angle_similarity']:.4f}")
                            st.image(result["output_frame_path"])
                            st.markdown("---")
                    else:
                        st.write("후속 프로세스 결과가 없습니다.")
                except Exception as e:
                    st.error(f"결과 처리 중 오류 발생: {e}")
