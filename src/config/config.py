from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic import DirectoryPath

from sqlalchemy import create_engine

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = 'config/.env', env_file_encoding = 'utf-8')
    #data_file_name: FilePath
    model_path: DirectoryPath
    model_name: str
    db_conn_str: str
    rent_apart_table_name: str

settings = Settings()

engine = create_engine(settings.db_conn_str)