FROM mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda11.0-cudnn8.1

RUN sudo ln -fs /usr/bin/python3.6 /usr/bin/python
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install subword_nmt sacremoses
RUN pip3 install pypinyin
RUN pip3 install tensorboardX
RUN wget https://github.com/facebookresearch/fairseq/archive/refs/tags/v0.10.2.tar.gz && \
    tar -xzvf v0.10.2.tar.gz && \
    cd fairseq-0.10.2 && \
    pip3 install --editable ./
RUN pip3 install sacrebleu==1.4.14