# get baseimage
FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get upgrade -y
# reinstall certificates, otherwise git clone command might result in an error
RUN apt-get install --reinstall ca-certificates -y

# set timezone for packages
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
# install developer dependencies
RUN apt-get install -y git build-essential --no-install-recommends curl libcurl4-openssl-dev libb64-dev unzip zip wget tar

# install cmake
RUN wget -qO- "https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# install vcpkg package manager
RUN git clone https://github.com/microsoft/vcpkg
RUN vcpkg/bootstrap-vcpkg.sh

# install crow package
RUN /vcpkg/vcpkg install crow

# install Triton Client libraries
RUN mkdir trt-client && cd trt-client \
    && wget https://github.com/triton-inference-server/server/releases/download/v2.29.0/v2.29.0_ubuntu2004.clients.tar.gz \
    && tar xzf v2.29.0_ubuntu2004.clients.tar.gz

RUN apt-get install -y libopencv-dev

# copy files from local directory to container
COPY . /trt-client-app

# compile with CMake 
RUN cd trt-client-app && mkdir build && cd build && bash -c "cmake .. && cmake --build ."

# run executable (name has to match with CMakeLists.txt file)
ENTRYPOINT [ "/trt-client-app/build/image_client" ]