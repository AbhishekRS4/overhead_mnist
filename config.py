from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Overhead MNIST API"
    version: str = "2024.03.02"
    num_classes: int = 10
    model_type: str = "medium_resnet"
    # one of the following - simple_cnn, simple_resnet, medium_resnet, deep_resnet
    device: str = "cpu"


settings = Settings()
