#  장면 탐색을 위한 Video to Text / Text to Frame 모델

## 🥇 팀 구성원
<table>
    <tr>
        <th align="center"><a href="https://github.com/DrunkLee">이상진</a></th>
        <th align="center"><a href="https://github.com/youhs1125">유희석</a></th>
        <th align="center"><a href="https://github.com/JJhun26">정지훈</a></th>
        <th align="center"><a href="https://github.com/chunyudong">천유동</a></th>
        <th align="center"><a href="https://github.com/subsup98">임용섭</a></th>
        <th align="center"><a href="https://github.com/ssujaewoo">박재우</a></th>
    </tr>
    <tr>
        <td align="center"><img src="https://github.com/user-attachments/assets/a95e8208-6cd7-4e9e-8268-1bc696bd56f7" width="100"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/328adcf5-4a22-48d6-983c-732202b529b0" width="100"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/c813d969-5442-4a52-80c4-f00d26dcd379" width="100"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/8d5fdffb-a81a-4da6-8c70-35b4d8402264" width="100"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/153869a0-abe4-4892-96aa-875a496f296d" width="100"></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/c03fb329-2690-493c-9e01-4ef03dbb3d17" width="100"></td>
    </tr>
    <tr>
    <td align="center">
        <strong>👑 팀장</strong><br>
        <small>🔹 Video 2 Text</small><br>
        <small>🔹 VLM Fine-Tuning</small><br>
        <small>🔹 최적화</small><br>
        <small>🔹 Prompt Engineering</small><br>
        <small>🔹 Streamlit</small>
    </td>
    <td align="center">
        <small>🔹 Video 2 Text</small><br>
        <small>🔹 VLM Fine-Tuning</small><br>
        <small>🔹 Data Cleansing</small><br>
        <small>🔹 Data Labeling</small><br>
        <small>🔹 최적화</small>
    </td>
    <td align="center">
        <small>🔹 Text 2 Frame</small><br>
        <small>🔹 Translation</small><br>
        <small>🔹 Data Cleansing</small><br>
        <small>🔹 Backend</small>
    </td>
    <td align="center">
        <small>🔹 Text 2 Frame</small><br>
        <small>🔹 VLM Test</small><br>
        <small>🔹 Data Cleansing</small><br>
        <small>🔹 Data Labeling</small>
    </td>
    <td align="center">
        <small>🔹 Text 2 Frame</small><br>
        <small>🔹 Backend</small><br>
        <small>🔹 Streamlit</small><br>
        <small>🔹 VLM Fine-Tuning</small>
    </td>
    <td align="center">
        <small>🔹 Video 2 Text</small><br>
        <small>🔹 VLM Test</small><br>
        <small>🔹 Audio</small><br>
        <small>🔹 Transcription</small>
    </td>
</tr>

</table>



## 📅 프로젝트 일정
2025.01.10(금) ~ 2025.02.12(수)

## 🔍 프로젝트 소개
본 프로젝트는 `부스트캠프 AI Tech` 에서 진행한 해커톤 주제 중 하나인 `TVING`의 **장면 탐색을 위한 Video-to-Text & Text-to-Frame** 주제 입니다.

비디오 탐색을 위한 **Video-to-Text 및 Text-to-Frame 모델**을 구축하여, 비디오 내용을 텍스트로 변환하고, **특정 텍스트와 가장 일치하는 장면**을 찾아주는 기능을 제공합니다.

## 📊 데이터셋
- Youtube-8M (`Tag : Movieclips`)
- Movie Clips 총 비디오 개수 : `1683개`
- Movie Clips 접근 가능 개수 : `1216개`
- **캡션 정보 없음**

## 🏗️ 프로젝트 구조

<details>
<summary><span style="font-size: 20px; font-weight: bold">Project Structure</span></summary>

```plaintext
📦 project-root
├── 📄 README.md
├── 
├── 📁 final_project
│   │
│   ├── 📜 main.py
│   │
│   ├── 📁 distribute
│   │   ├── 📜 flask_video_processor.py
│   │   ├── 📜 mainserver_flask.py
│   │   └── 📜 subserver_flask.py
│   │
│   ├── 📁 models
│   │   ├── 📜 add_embedding.py
│   │   ├── 📜 analyze.py
│   │   ├── 📜 angle_similarity.py
│   │   ├── 📜 audio_model.py
│   │   ├── 📜 clip_similarity.py
│   │   ├── 📜 frame_extract.py
│   │   ├── 📜 translation.py
│   │   └── 📜 video_processor.py
│   │
│   ├── 📁 modules
│   │   ├── 📜 flask_video_preprocess.py
│   │   ├── 📜 text_to_frame.py
│   │   ├── 📜 video_preprocess.py
│   │   └── 📜 video_to_text.py
│   │
│   ├── 📜 requirements.txt
│   └── 🛠️ setup.sh
│   
├── 📁 lora_train_json
│   ├── 📜 test_split.jsonl
│   ├── 📜 train_split.jsonl
│   └── 📜 val_split.jsonl
│   
├── 📁 model_test
│   ├── 📜 CogVLM2-llama2-caption.ipynb
│   ├── 📜 InternVL2.5-8B-MPO.ipynb
│   ├── 📜 InternVideo2-chat-8B.ipynb
│   ├── 📜 LLaVA-NeXt-Video-7B-hf.ipynb
│   ├── 📜 LLaVA-Video-7B-Qwen2.ipynb
│   ├── 📜 LongVU_Qwen2_7B.ipynb
│   ├── 📜 qwen2_test.ipynb
│   ├── 📜 videoMAE2-giant.ipynb
│   └── 📜 blip.ipynb
│   
├── 📁 utils
│   ├── 📜 Pseudo_labeling.py
│   ├── 📜 scene_split.py
│   └── 📜 split_videos.ipynb
│   
└── 📁 wrap_up_report
    └── 📜 최종_보고서_CV_프로젝트(11조).pdf
```

</details>

## ⚙️ Settings
본 프로젝트는 다음과 같은 환경에서 실행됩니다.
- **운영체제** : `Linux`
- **GPU** : `Tesla V100 32GB`
- **PyTorch** : `2.1.0`
- **CUDA** : `11.8`
- **cuDNN** : `8.7.0`
- **NVCC** : `11.8`
- **Python** : `3.10`

### 🔧 Setup
추론에 필요한 `LoRA Weight`와 `.NPZ DB`, 패키지들을 설치합니다.
``` bash
chmod +x setup.sh
sh setup.sh
```

### 🚀 Streamlit
```bash
streamlit run main.py
```
## 🛠 사용 방법
### 1. Streamlit
<p>
<img src="https://github.com/user-attachments/assets/80944e5e-7a67-4501-8800-efb1ab5b750a">
</p>
커맨드 창을 이용해 streamlit을 실행시켜 줍니다.

### 2. Video to Text
<p>
<img src="https://github.com/user-attachments/assets/4ac4c52c-2b53-4989-b77c-0c1fd9c03a04">
</p>

1. 추론하고 싶은 파일을 스트림릿에 올려줍니다.

2. 슬라이드바를 이용하여 비디오를 잘라줍니다.

3. 추론 버튼을 누릅니다

### 3. Video Preprocess
<p>
<img src="https://github.com/user-attachments/assets/b3cacc4e-4da3-4fe9-80ce-7e60160f2ff6">
</p>

1. VectorDB에 저장할 새로운 비디오 10개를 넣어줍니다.

<p>
<img src="https://github.com/user-attachments/assets/8bfd0bab-7ab6-4462-bc31-0786b3288111">
</p>

2. 스트림릿에 동영상이 모두 로드 되면 '비디오 프로세싱 시작'이라는 버튼이 나타나며 프로세싱을 시작합니다, 최대 10분 가량 걸리는 작업이니 인내심을 가지고 기다려주세요!

<p>
<img src="https://github.com/user-attachments/assets/123f7a21-f687-4396-aae7-3adc12ef9779">
</p>

3. 프로세싱이 마무리되면 '모든 비디오의 전처리가 완료되었습니다!' 라는 문구와 함께 VectorDB에 입력으로 들어온 비디오 임베딩이 저장됩니다!

### 4. Text to Frame

<p>
<img src="https://github.com/user-attachments/assets/97d3413f-f599-43e0-966c-ded520b6a0a1">
</p>

찾고 싶은 영상의 설명을 한글로 작성 후 '프레임 추출' 버튼을 누르고 잠시 뒤 VectorDB에서 유사도를 검사하여 Top 5의 결과를 보여줍니다!


## :computer:서비스 아키텍쳐
![image](https://github.com/user-attachments/assets/06bf1ad7-9145-4d8e-a342-f1840cbef735)



# LoRA Weight File Link
[LoRA_Weight](https://drive.google.com/file/d/1ZAWyN1aPXgWKbyCACnHz8LS9qE8Wqs7B/view?usp=drive_link)

# Youtube-8M Movie Clip AnglE features
[movie_clip_AnglE_UAE_Large_V1_features](https://drive.google.com/file/d/1mwfAh37wVEA3hJCLs1eXDjTe9vElefxB/view?usp=drive_link)
