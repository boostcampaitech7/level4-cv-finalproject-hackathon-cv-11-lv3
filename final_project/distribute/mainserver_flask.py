from   flask import Flask, request, jsonify, send_from_directory, abort, send_file
import os
import zipfile
import io
import threading
import time
import json

app = Flask(__name__)

BASE_DIR   = '/data/ephemeral/ys/test_videos'
FOLDERS    = ['folder_1', 'folder_2', 'folder_3']
UPLOAD_DIR = '/data/ephemeral/ys/test_jsons'
os.makedirs(UPLOAD_DIR, exist_ok=True)
for folder in FOLDERS:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

EXPECTED_SERVERS      = {"server1", "server2", "server3"}
received_results      = {} 
failed_servers        = set()
RESULT_TIMEOUT        = 180
timeout_timer         = None
start_time            = None

@app.route('/download/<folder_name>/<filename>', methods=['GET'])
def download_file(folder_name, filename):
    if folder_name not in FOLDERS:
        abort(404, description="Invalid folder name")
    folder_path = os.path.join(BASE_DIR, folder_name)
    file_path   = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        return send_from_directory(folder_path, filename, as_attachment=True)
    else:
        abort(404, description="File not found")

@app.route('/<folder_name>/list', methods=['GET'])
def list_files(folder_name):
    folder_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder not found'}), 404
    files       = os.listdir(folder_path)
    return jsonify({'files': files}), 200

def start_timeout_timer():
    global timeout_timer
    timeout_timer = threading.Timer(RESULT_TIMEOUT, on_timeout)
    timeout_timer.start()
    print(f"[Timeout Timer] {RESULT_TIMEOUT}초 후 타임아웃 처리 시작.")

def on_timeout():
    global failed_servers
    pending = EXPECTED_SERVERS - set(received_results.keys()) - failed_servers
    if pending:
        print(f"[Timeout] 타임아웃 발생: 아직 응답 없는 서버 {pending} 를 실패 처리합니다.")
        failed_servers.update(pending)
    process_results()

def process_results():
    global received_results, failed_servers, timeout_timer, start_time
    if timeout_timer is not None:
        timeout_timer.cancel()
    print("==== 최종 응답 집계 ====")
    print("성공 응답:", received_results)
    print("실패 서버:", failed_servers)
    print("후속 작업을 실행합니다...")
    for server_id, data in received_results.items():
        print(f"- {server_id}로부터 받은 데이터: {data}")
    received_results.clear()
    failed_servers.clear()
    start_time = None

@app.route('/receive_result', methods=['POST'])
def receive_result():
    global start_time, timeout_timer

    server_id = request.form.get('server_id')
    if not server_id:
        return jsonify({'error': 'Server ID not provided'}), 400

    if start_time is None:
        start_time = time.time()
        start_timeout_timer()

    if 'file' in request.files:
        file      = request.files['file']
        filename  = file.filename
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            file.save(file_path)
            print(f"[수신] {server_id}로부터 파일 저장 완료: {file_path}")
            received_results[server_id] = {
                                             'file_saved': True,
                                             'file_path': file_path
                                          }
        except Exception as e:
            failed_servers.add(server_id)
            print(f"[오류] {server_id} 파일 저장 중 오류: {str(e)}")
            return jsonify({'error': '파일 저장 실패', 'details': str(e)}), 500
    else:
        try:
            if request.is_json:
                      data = request.get_json()
            else:
                data_field = request.form.get('data')
                if data_field:
                      data = json.loads(data_field)
                else:
                      data = dict(request.form)
            received_results[server_id] = data
            print(f"[수신] {server_id}로부터 응답을 받았습니다: {data}")
        except Exception as e:
            failed_servers.add(server_id)
            print(f"[오류] {server_id} 응답 처리 중 오류: {str(e)}")
            return jsonify({'error': str(e)}), 500

    total_received     = len(received_results) + len(failed_servers)
    if total_received >= len(EXPECTED_SERVERS):
        process_results()

    return jsonify({'message': f"{server_id}의 응답 처리 완료"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
