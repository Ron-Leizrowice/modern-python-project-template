import os
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum

import httpx
import logfire
import ollama
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent

# ------------------------ Observability ------------------------------
logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

# ------------------------- Agent -------------------------------------
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"

MODEL = "gpt-oss:20b"

try:
    ollama.pull(MODEL)
except Exception as exc:
    raise RuntimeError("ollama not installed on your system.") from exc


agent = Agent(
    model=f"ollama:{MODEL}",
    instructions=("Be concise."),
)


# ----------------------- Tools -------------------------------------
class Market(Enum):
    FTSE_100 = "ftse_100"
    SNP_500 = "s&p_500"
    DAX = "dax"
    HANG_SENG = "hang_seng"
    STRAITS_TIMES = "strait_times"
    NIKKEI = "nikkei"

    @property
    def symbol(self) -> str:
        """Returns the symbol for the market."""
        match self:
            case Market.FTSE_100:
                return "^FTSE"
            case Market.SNP_500:
                return "^GSPC"
            case Market.DAX:
                return "^GDAXI"
            case Market.HANG_SENG:
                return "^HSI"
            case Market.STRAITS_TIMES:
                return "^STI"
            case Market.NIKKEI:
                return "^N225"

        raise ValueError("Unknown Market")


class City(StrEnum):
    NEW_YORK = "New York"
    LONDON = "London"
    TOKYO = "Tokyo"
    SINGAPORE = "Singapore"
    FRANKFURT = "Frankfurt"
    HONG_KONG = "Hong Kong"


class MarketMetaData(BaseModel):
    currency: str
    symbol: str
    exchangeName: str
    fullExchangeName: str
    instrumentType: str
    timezone: str
    regularMarketPrice: float
    fiftyTwoWeekHigh: float
    fiftyTwoWeekLow: float
    regularMarketDayHigh: float
    regularMarketDayLow: float
    longName: str


class MarketIndicators(BaseModel):
    low: list[float]
    close: list[float]
    volume: list[int]
    close: list[float]


class AdjIndicators(BaseModel):
    adjclose: list[float]


class Indicators(BaseModel):
    quote: list[MarketIndicators]
    adjclose: list[AdjIndicators]


class StockMarketData(BaseModel):
    meta: MarketMetaData
    timestamp: list[str]
    indicators: Indicators

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamps(cls, timestamp: list[int]) -> list[str]:
        dt = [datetime.fromtimestamp(ts, tz=UTC) for ts in timestamp]
        return [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in dt]


class DailyUnits(BaseModel):
    temperature_2m_max: str
    temperature_2m_min: str
    uv_index_max: str
    sunshine_duration: str
    daylight_duration: str
    rain_sum: str
    showers_sum: str
    snowfall_sum: str
    wind_speed_10m_max: str
    wind_gusts_10m_max: str


class DailyWeatherData(BaseModel):
    time: list[str]
    temperature_2m_max: list[float]
    temperature_2m_min: list[float]
    uv_index_max: list[float]
    sunshine_duration: list[float]
    daylight_duration: list[float]
    rain_sum: list[float]
    showers_sum: list[float]
    wind_speed_10m_max: list[float]
    wind_gusts_10m_max: list[float]


class WeatherData(BaseModel):
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    timezone_abbreviation: str
    elevation: float
    daily: DailyWeatherData


# ------------------------ Tools ---------------------------------------------
@agent.tool_plain
async def get_market_data(market: Market, timerange: int) -> StockMarketData:
    """Return market data for a given market.

    Args:
        market: The market to query.
        timerange: The number of days to look back.
        interval: Time interval
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
    }

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=timerange)
    start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
    query_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{market.symbol}?period1={start_ts}&period2={end_ts}&interval=1d"
    async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
        r = await client.get(query_url)
    r.raise_for_status()
    data = r.json()

    return StockMarketData.model_validate(data["chart"]["result"][0])


def _geocode_city(name: str) -> tuple[float, float]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": name, "count": 1, "language": "en", "format": "json"}
    with httpx.Client(timeout=10) as x:
        r = x.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    if not j.get("results"):
        raise ValueError(f"city not found: {name}")
    lat = float(j["results"][0]["latitude"])
    lon = float(j["results"][0]["longitude"])
    return lat, lon


@agent.tool_plain
async def city_weather(city: City, timerange: int) -> WeatherData:
    """Return recent weather for a city."""

    latitude, longitude = _geocode_city(city)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "sunshine_duration",
            "rain_sum",
            "showers_sum",
            "wind_speed_10m_max",
        ],
        "past_days": timerange,
        "forecast_days": 0,
    }
    with httpx.Client(timeout=10) as x:
        r = x.get(url, params=params)
        r.raise_for_status()
        j = r.json()

    return WeatherData.model_validate(j)


# ------------------------ Runner --------------------------------------------
async def run_agent(prompt: str) -> str:
    """Run the agent with a given prompt."""
    result = await agent.run(prompt)
    return result.output
