FROM openvino/ubuntu20_dev:2021.4.2_20210416
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE=/home/openvino/app

ENV INTEL_OPENVINO_DIR=/opt/intel/openvino
USER root

RUN apt-get -yqq update \ 
    && apt-get install -yq --no-install-recommends \
      git \
      yasm libx264-dev \
      vim \
      make \
      g++ \
      wget \
      npm \
      libzmq3-dev \
      libkrb5-dev \
      ffmpeg \
    && apt-get clean \  
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt $WORKSPACE/

RUN python3 -m pip install --upgrade pip \
    && pip3 --no-cache-dir install -r $WORKSPACE/requirements.txt \
    && pip3 --no-cache-dir install -r ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/requirements.txt

###### Configure the Docker Image with access to GPU
# https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_docker_linux.html
# Used image version https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_docker_linux.html
# Use `20.35.17767` for 10th generation Intel® Core™ processor (formerly Ice Lake) 
# or 11th generation Intel® Core™ processor (formerly Tiger Lake)
ARG INTEL_OPENCL=19.41.14441
ARG DEVICE
RUN if [[ "${DEVICE,,}" == "gpu" ]] ; then \
  ${INTEL_OPENVINO_DIR}/install_dependencies/install_NEO_OCL_driver.sh --no_numa -y --install_driver ${INTEL_OPENCL};\
    rm -rf /var/lib/apt/lists/* ; \
  fi

COPY people-counter ${WORKSPACE}/people-counter
COPY resources ${WORKSPACE}/resources
COPY models ${WORKSPACE}/models

# Upgrade npm
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash \
    && apt-get install -y nodejs \
    &&  npm -v \
    && npm update npm -g \
    && npm -v

#  http://www.tiernok.com/posts/2019/faster-npm-installs-during-ci/
RUN cd $WORKSPACE/people-counter/webservice/server \
  && cp .npmrc $HOME/ \ 
  && npm install -g npm \
  && npm init --yes \
  && npm install \
  && npm dedupe \
  && cd $WORKSPACE/people-counter/webservice/ui \
  && npm cache clean --force \
  && npm install \
  && npm dedupe \
  && rm -rf /tmp/npm* \
  && chown -R openvino $WORKSPACE

# We need to check out an old version of FFmpeg to have FFserver as FFmpeg no longer bundles it
RUN git clone -c http.sslverify=false https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg \
  && cd /tmp/ffmpeg \
  && git checkout 2ca65fc7b74444edd51d5803a2c1e05a801a6023 \
  && ./configure \
  && make \
  && make install \
  && rm -rf /tmp/ffmpeg

WORKDIR ${WORKSPACE}/people-counter
USER openvino
RUN echo "source ${INTEL_OPENVINO_DIR}/bin/setupvars.sh -pyver `python3 --version | cut -d '.' -f1,2`" >> $HOME/.bashrc

EXPOSE 3000 3002 3004

CMD /bin/bash