from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    model_path: str = "models/pneumonia_resnet18.pth"
    rag_index_dir: str = "medical_documents/index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
