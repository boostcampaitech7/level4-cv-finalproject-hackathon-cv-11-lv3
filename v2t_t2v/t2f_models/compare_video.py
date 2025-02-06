import os
import json
import shutil

json_file_path = '희석이가 올려준 이름이랑 경로랑 매칭된 json의 위치'  
new_video_folder = '새로운 비디오들이 저장된 폴더 폴더의 위치를 선택'       
target_folder = '새로운 비디오들만 저장해놓을 곳으로' # 혹은 각자의 이름을 가진 폴더를 만들게 하거나?             

os.makedirs(target_folder, exist_ok=True)

if os.path.exists(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        video_mapping = json.load(f)
else:
    video_mapping     = {}
existing_video_ids    = set(video_mapping.keys())
print(f"기존 JSON에 등록된 비디오 ID 수: {len(existing_video_ids)}")

for video_file in os.listdir(new_video_folder):
    video_path = os.path.join(new_video_folder, video_file)
    if not os.path.isfile(video_path):
        continue

    file_name = os.path.splitext(video_file)[0]
    
    if len(file_name) >= 11:
        video_id_candidate = file_name[-11:]
    else:
        video_id_candidate = file_name

    if video_id_candidate in existing_video_ids:
        print(f"[이미 등록] '{video_file}' (ID: {video_id_candidate})")
        continue

    destination_path = os.path.join(target_folder, video_file)
    try:
        shutil.move(video_path, destination_path)
        print(f"[이동 완료] '{video_file}' (ID: {video_id_candidate}) -> {destination_path}")
        
        video_mapping[video_id_candidate] = destination_path
        existing_video_ids.add(video_id_candidate) 
    except Exception as e:
        print(f"[오류] '{video_file}' 이동 실패: {e}")

with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(video_mapping, f, indent=4, ensure_ascii=False)

print("JSON 파일 업데이트 완료!")
#마지막으로 new 폴더를 비워주면 끝일듯?
