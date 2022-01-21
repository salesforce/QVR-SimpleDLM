pip --no-cache-dir install pandas==1.0.1 && \
pip --no-cache-dir install sklearn==0.21.3 && \
pip --no-cache-dir install nltk==3.4.5 && \
pip --no-cache-dir install scikit-learn==0.21.3 && \
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 python=3.6 -c pytorch -y && conda clean --all -y && \
pip --no-cache-dir install opencv-python==4.5.5 -U && \
pip --no-cache-dir install tensorboard==2.6.0 && \
pip --no-cache-dir install pyyaml==3.12 --ignore-installed && \
pip --no-cache-dir install tensorboardX==2.0 && \
pip --no-cache-dir install cython==0.27.3

apt-get update && apt-get install -y locales && \
locale-gen en_US.UTF-8 && \
export LANG=en_US.UTF-8 && \
export LANGUAGE=en_US:en && \
export LC_ALL=en_US.UTF-8 && \
apt-get update && apt-get install psmisc