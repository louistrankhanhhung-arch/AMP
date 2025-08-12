from dataclasses import dataclass
import os

@dataclass
class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHANNEL_ID: int = int(os.getenv("TELEGRAM_CHANNEL_ID", "0"))
    KUCOIN_KEY: str = os.getenv("KUCOIN_KEY", "")
    KUCOIN_SECRET: str = os.getenv("KUCOIN_SECRET", "")
    KUCOIN_PASSPHRASE: str = os.getenv("KUCOIN_PASSPHRASE", "")
    TOP_K: int = int(os.getenv("TOP_K", "8"))
    RISK_PCT: float = float(os.getenv("RISK_PCT", "0.8"))
    ATR_SL_K: float = float(os.getenv("ATR_SL_K", "1.2"))
    TP_MIN_SPACING_ATR: float = float(os.getenv("TP_MIN_SPACING_ATR", "0.5"))

settings = Settings()
