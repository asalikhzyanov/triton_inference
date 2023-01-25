# Simple Inference Server with Triton

This repository contains a source code for a C++ server that accepts POST requests. In a POST request, you can send a picture that the server classifies and gives the class name.

![Diagram](https://github.com/asalikhzyanov/triton_inference/blob/main/content/diagram.png)

## Run

### Server with client app
**NB**: To run the whole app you will need to pull `nvcr.io/nvidia/tritonserver:22.12-py3`. Docker image description: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch. 

```
git clone https://github.com/asalikhzyanov/triton_inference.git
cd /path/to/triton_inference
docker-compose up --build
```

The app will open an entrypoint http://localhost:18080/image to accept an image to classify.

### POST request
```
python3 simple_request.py  -i /path/to/image -u http://ip:18080/image
```

## #TODOs
### Server
* Add new models to repository
* Shrink a tritonserver image to exclude non-needed constituents
* Increase security by creating a docker network for server and client-apps

### Client apps
#### image-client
* Fetch and set model parameters automatically
* Make tritonserver ip a configurable parameter 
* Make topk a parameter settable from outside
* Verbose request responses
* Handle exceptions
* Add batchinng
#### detection-app
* The whole app :)

## References
* https://github.com/triton-inference-server/client
* https://crowcpp.org/master/
* https://stackoverflow.com/questions/73332642/c-with-crow-cmake-and-docker