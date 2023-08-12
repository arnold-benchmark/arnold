# FROM public.ecr.aws/nvidia/isaac-sim:2022.1.1

ARG DEBIAN_FRONTEND=noninteractive
ARG BASE_DIST=ubuntu20.04
ARG CUDA_VERSION=11.4.2
ARG ISAACSIM_VERSION=2022.1.1

FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION} as isaac-sim
FROM nvidia/cudagl:${CUDA_VERSION}-base-${BASE_DIST}

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y neofetch unrar net-tools iputils-ping software-properties-common curl zip python3 python3-pip tmux git htop build-essential aria2 openssh-server xrdp xfce4* sudo vim libtf-dev && apt-get clean
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatomic1 \
    libegl1 \
    libglu1-mesa \
    libgomp1 \
    libsm6 \
    libxi6 \
    libxrandr2 \
    libxt6 \
    wget \
    vulkan-utils \
&& apt-get -y autoremove \
&& apt-get clean autoclean \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN echo 'root:root' | chpasswd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo "xfce4-session" > /root/.xsession
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN cd /root && aria2c -q "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz" && tar xvf ngrok-v3-stable-linux-amd64.tgz && rm ngrok-v3-stable-linux-amd64.tgz
USER root
RUN cd /root && aria2c -q "https://go.microsoft.com/fwlink/?LinkID=760868" && DEBIAN_FRONTEND="noninteractive" apt install -y ./*.deb && rm *.deb && apt-get clean
RUN curl https://get.docker.com/ | bash
RUN usermod -aG docker root
RUN sed -ri 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config
EXPOSE 1-65535

ARG VULKAN_SDK_VERSION=1.3.224.1
# Download the Vulkan SDK and extract the headers, loaders, layers and binary utilities
RUN wget -q --show-progress \
    --progress=bar:force:noscroll \
    https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \
    -O /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \ 
    && echo "Installing Vulkan SDK ${VULKAN_SDK_VERSION}" \
    && mkdir -p /opt/vulkan \
    && tar -xf /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz -C /opt/vulkan \
    && mkdir -p /usr/local/include/ && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/include/* /usr/local/include/ \
    && mkdir -p /usr/local/lib && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/* /usr/local/lib/ \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/libVkLayer_*.so /usr/local/lib \
    && mkdir -p /usr/local/share/vulkan/explicit_layer.d \
    && cp /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/etc/vulkan/explicit_layer.d/VkLayer_*.json /usr/local/share/vulkan/explicit_layer.d \
    && mkdir -p /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/share/vulkan/registry/* /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/bin/* /usr/local/bin \
    && ldconfig \
    && rm /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz && rm -rf /opt/vulkan

# Setup the required capabilities for the container runtime    
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Open ports for live streaming
EXPOSE 47995-48012/udp \
       47995-48012/tcp \
       49000-49007/udp \
       49000-49007/tcp \
       49100/tcp \
       8011/tcp \
       8012/tcp \
       8211/tcp \
       8899/tcp \
       8891/tcp

ENV OMNI_SERVER http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2022.1
# ENV OMNI_SERVER omniverse://localhost/NVIDIA/Assets/Isaac/2022.1
# ENV OMNI_USER admin
# ENV OMNI_PASS admin
ENV MIN_DRIVER_VERSION 470.141.03

COPY --from=isaac-sim /isaac-sim /isaac-sim

WORKDIR /isaac-sim

RUN ln -s exts/omni.isaac.examples/omni/isaac/examples extension_examples

RUN echo '' > vulkan_check.sh



# move the file here
RUN cd /root/ && ls && git clone git@github.com:arnold-benchmark/arnold.git
RUN cd /root/ && mkdir .code
# need to activate nvidia runtime during docker build, https://medium.com/@jgleeee/building-docker-images-that-require-nvidia-runtime-environment-1a23035a3a58
RUN cd /root/arnold/ && /isaac-sim/python.sh -m pip install num2words==0.5.12 opencv-python==4.8.0.76 tqdm==4.66.1 ftfy==6.1.1 regex==2023.8.8 transformers==4.30.0 einops==0.6.1 tensorboard==2.11.2 sentencepiece==0.1.99 clip-by-openai==1.1 kornia==0.5.11
RUN cd /root/arnold/ && /isaac-sim/python.sh -m pip install Cython==3.0.0
RUN cd /root/arnold/ && /isaac-sim/python.sh -m pip install hydra-core==1.3.2
RUN cd /root/arnold/ && /isaac-sim/python.sh -m pip install --upgrade cython==3.0.0
RUN cd /root/arnold/utils && /isaac-sim/python.sh setup.py build_ext --inplace
RUN cd /root/arnold/ && /isaac-sim/python.sh -m pip install typing_extensions==4.3.0 -U
#&& cp /root/temp/cliport/utils/compute_points.cpython-*-linux-gnu.so /root/project_amazon/cliport/cliport/utils/ && rm -rf /root/temp/

ENTRYPOINT []
CMD service docker start; /usr/sbin/sshd -D
