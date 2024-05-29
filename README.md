# Overhead MNIST


## Dataset info
* The dataset used in this project is the Overhead-MNIST dataset and it can be found [here](https://www.kaggle.com/datasets/datamunge/overheadmnist/)
* The dataset contains 10 classes


## Project info
* This repo contains an end-to-end deep learning project deployment for overhead image classification
* The purpose of this project is to try multiple new things. The following new things that have been explored
  * In this project, the recently popular [Kolmogorov-Arnold Networks (KAN)](https://kindxiaoming.github.io/pykan/index.html) is also explored
  * Simple and Complex CNNs using the KAN layer is developed and tested
  * Weights and Biases has been used for MLOps including its new feature of model registry
  * For training, the Automatic Mixed Precision (AMP) has been explored by using `float16` for some layer weights to speed up training
  * For deployment, an API has been developed and deployed using FastAPI and docker
* For the training, the dataset is split into 95% - 5% for train and validation sets respectively
* The python packages are listed in [requirements.txt](requirements.txt)
* The docker container can be deployed using [Dockerfile](Dockerfile)
* For training and logging the model, use the [modeling/train.py](modeling/train.py) script
* The FastAPI app deployment code is in [app.py](app.py) script
* To test the deployed FastAPI app on a local machine, the [test_post_request.py](test_post_request.py) script can be used
* Some sample test images are available in [sample_test_images](sample_test_images)


## Training using docker image
* Use the following command for training using the docker image
```
docker run --rm -it --init   --gpus=all   --ipc=host   --user="$(id -u):$(id -g)"   --volume="$PWD:/app"   my_pytorch python3 modeling/train.py --dir_dataset /app/dir_dataset/
```


## Docker deployment instructions on a local machine
* For deployment [requirements_deployment.txt](requirements_deployment.txt) needs to be used
* To build the container, run the following command
```
docker build -t fastapi_overhead_mnist .
```
* To the run the container, run the following command
```
docker run -p 7860:7860 -t fastapi_overhead_mnist
```


## HuggingFace deployment
* The FastAPI application has also been deployed to [HuggingFace](https://huggingface.co/spaces/abhishekrs4/Overhead_MNIST)
* To test the deployed FastAPI app on HuggingFace, use the [test_post_request.py](https://huggingface.co/spaces/abhishekrs4/Overhead_MNIST/blob/main/test_post_request.py) script in the HuggingFace repo since the endpoint is different


## Docs
* The docs generated with sphinx can be found in [_build/html/index.html](_build/html/index.html)


## Sample test images
![Sample test image 1](sample_test_images/68162.jpg?raw=true)
![Sample test image 2](sample_test_images/68164.jpg?raw=true)
![Sample test image 3](sample_test_images/68166.jpg?raw=true)
![Sample test image 4](sample_test_images/68172.jpg?raw=true)
![Sample test image 5](sample_test_images/68186.jpg?raw=true)
![Sample test image 6](sample_test_images/76675.jpg?raw=true)
![Sample test image 7](sample_test_images/76679.jpg?raw=true)
![Sample test image 8](sample_test_images/76680.jpg?raw=true)
![Sample test image 9](sample_test_images/76685.jpg?raw=true)
![Sample test image 10](sample_test_images/76686.jpg?raw=true)


## References
* [Kolmogorov-Arnold Networks (KAN) Paper](https://arxiv.org/abs/2404.19756)
* [Kolmogorov-Arnold Networks (KAN) Github](https://github.com/KindXiaoming/pykan)
* [Kolmogorov-Arnold Networks (KAN) Docs](https://kindxiaoming.github.io/pykan/index.html)
