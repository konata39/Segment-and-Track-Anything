## Environment setting
#set conda environment\
conda create -n {env_name} python=3.9\
conda activate {env_name}\
\
#Install SAM\
cd sam; pip install -e .\
cd -\
\
#Install Grounding-Dino\
pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO \
\
#Install conda for cuda 11.8\
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 –index-url https://download.pytorch.org/whl/cu118 \
\
#Install other lib\
pip install numpy opencv-python pycocotools matplotlib Pillow==9.2.0 scikit-image \
pip install gradio==3.39.0 gdown ffmpeg==1.4\
#Install needed lab for zip\
pip install -r zip_requirements.txt\
pip install zip\
\
#Install Pytorch Correlation \
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git \
cd Pytorch-Correlation-extension \
pip install -e . \
cd - \
\
#Install ckpt\
sh ./script/download_ckpt.sh

## 使用說明
https://hackmd.io/9osxi6z-THiEexabUZHVMQ
