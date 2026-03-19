from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://triton:triton@localhost:5432/triton"
    redis_url: str = "redis://localhost:6379/0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    upload_dir: str = "/data/tmp"

    model_config = {"env_file": ".env"}


settings = Settings()
