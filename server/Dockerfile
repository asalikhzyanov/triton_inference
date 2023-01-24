FROM nvcr.io/nvidia/tritonserver:22.12-py3

ADD ./model_repository /models
ADD ./fetch_models.sh ./
RUN  sh ./fetch_models.sh
RUN rm ./fetch_models.sh

ENTRYPOINT [ "tritonserver" ]