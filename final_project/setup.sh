pip install gdown
gdown 11xU9Is8gd5cIMNzLlBcPr7lLzjFyZKpd
tar -xvf videos_movieclips.tar

gdown 1ZAWyN1aPXgWKbyCACnHz8LS9qE8Wqs7B
tar -xvzf epoch4.tar.gz

gdown 1mwfAh37wVEA3hJCLs1eXDjTe9vElefxB

conda update conda
conda update --all

conda install -c nvidia cudatoolkit=11.8
conda install -c nvidia cuda-nvcc=11.8

apt update
apt-get update
apt-get install -y libjpeg-dev libpng-dev
apt-get install -y libgl1-mesa-glx
apt install tmux
apt install git
apt install ffmpeg
apt install cloudflared

pip install -r requirements.txt

git clone https://github.com/boostcampaitech7/level4-cv-finalproject-hackathon-cv-11-lv3.git
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-1B-MPO --local-dir InternVL2_5-1B-MPO