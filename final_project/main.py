import streamlit                      as st
from   modules.video_to_text          import Video2TextPage
from   modules.text_to_frame          import Text2FramePage
from   modules.video_preprocess       import VideoPreprocessingPage
from   modules.flask_video_preprocess import Text2FrameDistributedInference

class AppManager:
    def __init__(self):
        self.pages = {
            "🎬 Video to Text"                : Video2TextPage(),
            "🎥 Video Preprocessing"          : VideoPreprocessingPage(),
            "🔎 Text to Frame"                : Text2FramePage(),
            "🔎 Text to Frame(분산 추론) 🚧"  : Text2FrameDistributedInference()
        }

    def run(self):
        st.sidebar.title("📌 메뉴")
        page_name = st.sidebar.radio("이동할 페이지를 선택하세요:", list(self.pages.keys()))
        self.pages[page_name].run()
        
if __name__ == "__main__":
    app = AppManager()
    app.run()