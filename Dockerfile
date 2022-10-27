FROM mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda11.0-cudnn8.1

RUN sudo ln -fs /usr/bin/python3.6 /usr/bin/python
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install subword_nmt sacremoses
RUN pip3 install pypinyin
RUN pip3 install tensorboardX
RUN git clone https://github.com/yc1999/fairseq.git && \
    cd fairseq && \
    pip3 install --editable ./