from   flask import Flask, request, jsonify
import os
import requests
import json
from v2t_model import AnalyzeVideo
#from models.analyze import AnalyzeVideo
app = Flask(__name__)

MAIN_SERVER_BASE_URL = "https://parish-beliefs-norwegian-household.trycloudflare.com"
FOLDER_NAME          = "folder_2"
DOWNLOAD_DIR         = "/data/ephemeral/home/ys/folder_2"
RESULT_ENDPOINT      = f"{MAIN_SERVER_BASE_URL}/receive_result"
SERVER_NAME          = "server1"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@app.route('/trigger_download', methods=['POST'])
def trigger_download():
    list_url     = f"{MAIN_SERVER_BASE_URL}/{FOLDER_NAME}/list"
    try:
        response = requests.get(list_url)
        if response.status_code == 200:
            file_list = response.json().get("files", [])
            
            if not file_list:
                return jsonify({'message': '폴더에 파일이 없습니다.'}), 400
            print(f"Main Server {FOLDER_NAME}의 파일 목록: {file_list}")
        
        else:
            return jsonify({
                            'error': 'Failed to fetch file list',
                            'status_code': response.status_code
                           }), 500
    except Exception as e:
        return jsonify({'error': f"Error during file list fetch: {str(e)}"}), 500

    for filename in file_list:
        download_url    = f"{MAIN_SERVER_BASE_URL}/download/{FOLDER_NAME}/{filename}"
        local_file_path = os.path.join(DOWNLOAD_DIR, filename)    
        try:
            resp        = requests.get(download_url, stream=True)
            if resp.status_code == 200:
                with open(local_file_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"{SERVER_NAME} 파일 다운로드 성공: {local_file_path}")
            else:
                print(f"{SERVER_NAME} 파일 다운로드 실패: {filename}. 상태 코드: {resp.status_code}")
                continue
        except Exception as e:
            print(f"{SERVER_NAME} 파일 다운로드 중 오류 발생: {filename}, {str(e)}")
            continue


    local_file_dir   = DOWNLOAD_DIR
    local_files_path = [
                        os.path.join(local_file_dir, file)
                        for file in os.listdir(local_file_dir)
                        if file.lower().endswith('.mp4')
                       ]
    print(f"{SERVER_NAME} 분석할 파일 목록: {local_files_path}")


    JSON_OUTPUT_DIR = os.path.join(DOWNLOAD_DIR, "json_output")
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

    try:
        with AnalyzeVideo(use_audio = True,
                            num_seg = 3, 
                         batch_size = 7) as analyzer:
                   analysis_result  = analyzer.batch_analyze(video_paths=local_files_path, output_path=JSON_OUTPUT_DIR)
        print(f"{SERVER_NAME} 동영상 분석 완료. 분석 결과: {analysis_result}")
    
    except Exception as e:
        print(f"{SERVER_NAME} 동영상 분석 중 오류 발생: {str(e)}")
        return jsonify({'error': f"동영상 분석 중 오류 발생: {str(e)}"}), 500

    results = []
    for json_filename in os.listdir(JSON_OUTPUT_DIR):
        json_file_path = os.path.join(JSON_OUTPUT_DIR, json_filename)
        try:
            with open(json_file_path, 'rb') as json_file:
                files           = {'file': json_file}
                data            = {'server_id': f'{SERVER_NAME}'}
                result_response = requests.post(RESULT_ENDPOINT, files=files, data=data)
            
            if result_response.status_code == 200:
                print(f"{SERVER_NAME} 결과 파일 전송 성공: {json_file_path}")
                results.append(json_file_path)
            
            else:
                print(f"{SERVER_NAME} 결과 파일 전송 실패: {json_file_path}. 상태 코드: {result_response.status_code}")
        except Exception as e:
            print(f"{SERVER_NAME} 결과 파일 전송 중 오류 발생: {str(e)}")

    return jsonify({
        'message': f'{FOLDER_NAME}의 모든 파일 다운로드 및 처리 완료',
        'sent': results
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
