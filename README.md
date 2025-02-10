#  장면 탐색을 위한 Video to Text / Text to Frame 모델

## 🥇 팀 구성원
 
### 이상진, 유희석, 정지훈, 천유동, 임용섭, 박재우

## 📅 프로젝트 일정
프로젝트 전체 일정

- 2025.01.10(금) ~ 2025.02.12(수)

## 📊 데이터셋
Youtube-8M에서 제공하는 Movie Clip 데이터셋으로 다음과 같은 구성을 따릅니다. 

- 전체 비디오 개수 : 1216개

## 🚀 빠른 시작
### Launch Streamlit
1. Aistage의 server4로 접속
2. /data/ephemeral/home/level4-cv-finalproject-hackathon-cv-11-lv3/final_project 폴더로 이동 후
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

### Video Preprocess
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

### Text to Frame

<p>
<img src="https://github.com/user-attachments/assets/123f7a21-f687-4396-aae7-3adc12ef9779">
</p>

찾고 싶은 영상의 설명을 한글로 작성 후 '프레임 추출' 버튼을 누르고 잠시 뒤 VectorDB에서 유사도를 검사하여 Top 5의 결과를 보여줍니다!


## :computer:서비스 아키텍쳐
![image](https://github.com/user-attachments/assets/06bf1ad7-9145-4d8e-a342-f1840cbef735)




# LoRA Weight File Link
[LoRA_Weight](https://drive.google.com/file/d/1ZAWyN1aPXgWKbyCACnHz8LS9qE8Wqs7B/view?usp=drive_link)

# Youtube-8M Movie Clip AnglE features
[movie_clip_AnglE_UAE_Large_V1_features](https://drive.google.com/file/d/1mwfAh37wVEA3hJCLs1eXDjTe9vElefxB/view?usp=drive_link)
