import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

class Settings:
    def __init__(self):
        self.allow_start_without_db = os.getenv('ALLOW_START_WITHOUT_DB', '0') in ('1', 'true', 'TRUE')
        self.skip_db_migration = os.getenv('DB_SKIP_MIGRATION', '0') in ('1', 'true', 'TRUE')
        # SSL 验证配置（可选）
        self.ca_bundle = os.getenv('SSL_CA_BUNDLE')  # 自定义 CA 证书路径
        self.ssl_verify = os.getenv('SSL_VERIFY', 'true').lower() not in ('0', 'false', 'no')
        # CORS (optional): comma-separated origins; same-origin setup can leave this empty
        allow_origins_str = os.getenv('ALLOW_ORIGINS', '')
        self.allow_origins = [o.strip() for o in allow_origins_str.split(',') if o.strip()]
        self.allow_credentials = os.getenv('ALLOW_CREDENTIALS', 'false').lower() in ('1', 'true', 'yes')
        self.http_timeout_seconds = float(os.getenv('HTTP_TIMEOUT_SECONDS', '8'))
        self.faq_db_path = os.getenv('FAQ_DB_PATH', './data/faq.db')

settings = Settings()
