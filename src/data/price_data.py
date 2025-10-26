"""
price_data.py
Price data retrieval and caching using eodhd.com API with yfinance fallback.
Handles caching, rate limiting, and data normalization.
"""
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from dotenv import load_dotenv

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None

# Load environment variables
load_dotenv()


class PriceDataClient:
    """
    Client for fetching and caching price data from eodhd.com and yfinance.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        rate_limit_delay: float = 0.1,
        use_fallback: bool = True
    ):
        """
        Initialize price data client.
        
        Args:
            api_key: eodhd.com API key (or from EODHD_API_KEY env var)
            cache_dir: Directory for caching price data
            rate_limit_delay: Delay between API calls in seconds
            use_fallback: Whether to use yfinance as fallback
        """
        self.api_key = api_key or os.getenv("EODHD_API_KEY")
        self.cache_dir = cache_dir or os.getenv("PRICE_CACHE_DIR", "data/price_cache")
        self.rate_limit_delay = rate_limit_delay
        self.use_fallback = use_fallback and HAS_YFINANCE
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup requests session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # In-memory cache
        self._memory_cache: Dict[str, pd.DataFrame] = {}
        
        # Load existing cache
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load index of cached files."""
        cache_index_path = os.path.join(self.cache_dir, "cache_index.parquet")
        if os.path.exists(cache_index_path):
            try:
                self.cache_index = pd.read_parquet(cache_index_path)
            except Exception:
                self.cache_index = pd.DataFrame(columns=["ticker", "last_updated", "source"])
        else:
            self.cache_index = pd.DataFrame(columns=["ticker", "last_updated", "source"])
    
    def _save_cache_index(self):
        """Save cache index."""
        cache_index_path = os.path.join(self.cache_dir, "cache_index.parquet")
        self.cache_index.to_parquet(cache_index_path, index=False)
    
    def _get_cache_path(self, ticker: str) -> str:
        """Get cache file path for a ticker."""
        return os.path.join(self.cache_dir, f"{ticker}.parquet")
    
    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load price data from cache."""
        # Check memory cache
        if ticker in self._memory_cache:
            return self._memory_cache[ticker]
        
        # Check disk cache
        cache_path = self._get_cache_path(ticker)
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").reset_index(drop=True)
                self._memory_cache[ticker] = df
                return df
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, ticker: str, df: pd.DataFrame, source: str):
        """Save price data to cache."""
        cache_path = self._get_cache_path(ticker)
        df.to_parquet(cache_path, index=False)
        self._memory_cache[ticker] = df
        
        # Update cache index
        if ticker in self.cache_index["ticker"].values:
            self.cache_index.loc[self.cache_index["ticker"] == ticker, "last_updated"] = datetime.now()
            self.cache_index.loc[self.cache_index["ticker"] == ticker, "source"] = source
        else:
            new_row = pd.DataFrame([{
                "ticker": ticker,
                "last_updated": datetime.now(),
                "source": source
            }])
            self.cache_index = pd.concat([self.cache_index, new_row], ignore_index=True)
        
        self._save_cache_index()
    
    def fetch_from_eodhd(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        exchange: str = "US"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch price data from eodhd.com API.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchange: Exchange code (default: US)
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adjusted_close, Volume
        """
        if not self.api_key:
            return None
        
        # Default date range: last 5 years
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        
        # Construct API URL
        symbol = f"{ticker}.{exchange}"
        url = f"https://eodhd.com/api/eod/{symbol}"
        params = {
            "api_token": self.api_key,
            "from": start_date,
            "to": end_date,
            "fmt": "json"
        }
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                return None
            
            df = pd.DataFrame(data)
            
            # Normalize column names
            df = df.rename(columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adjusted_close": "Adjusted_close",
                "volume": "Volume"
            })
            
            # Ensure Date column
            if "Date" not in df.columns:
                return None
            
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            
            # Add ticker column
            df["ticker"] = ticker
            
            return df
            
        except Exception as e:
            print(f"Error fetching from eodhd for {ticker}: {e}")
            return None
    
    def fetch_from_yfinance(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch price data from yfinance (fallback).
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adjusted_close, Volume
        """
        if not HAS_YFINANCE:
            return None
        
        # Default date range: last 5 years
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=False)
            
            if df.empty:
                return None
            
            # Reset index to get Date as column
            df = df.reset_index()
            
            # Normalize column names
            df = df.rename(columns={
                "Adj Close": "Adjusted_close"
            })
            
            # Ensure required columns
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required_cols):
                return None
            
            # Add Adjusted_close if missing
            if "Adjusted_close" not in df.columns:
                df["Adjusted_close"] = df["Close"]
            
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            
            # Add ticker column
            df["ticker"] = ticker
            
            return df
            
        except Exception as e:
            print(f"Error fetching from yfinance for {ticker}: {e}")
            return None
    
    def get_price_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
        exchange: str = "US"
    ) -> Optional[pd.DataFrame]:
        """
        Get price data with caching. Tries eodhd first, then yfinance.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: Force refresh from API
            exchange: Exchange code for eodhd (default: US)
            
        Returns:
            DataFrame with price data or None
        """
        # Try cache first
        if not force_refresh:
            cached_df = self._load_from_cache(ticker)
            if cached_df is not None:
                # Filter by date range if specified
                if start_date or end_date:
                    if start_date:
                        cached_df = cached_df[cached_df["Date"] >= start_date]
                    if end_date:
                        cached_df = cached_df[cached_df["Date"] <= end_date]
                return cached_df
        
        # Try eodhd
        df = self.fetch_from_eodhd(ticker, start_date, end_date, exchange)
        if df is not None and not df.empty:
            self._save_to_cache(ticker, df, "eodhd")
            return df
        
        # Fallback to yfinance
        if self.use_fallback:
            df = self.fetch_from_yfinance(ticker, start_date, end_date)
            if df is not None and not df.empty:
                self._save_to_cache(ticker, df, "yfinance")
                return df
        
        return None
    
    def get_forward_return(
        self,
        ticker: str,
        as_of_date: str,
        forward_days: int = 5,
        price_col: str = "Close"
    ) -> Optional[float]:
        """
        Calculate forward return for a ticker from a specific date.
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Start date (YYYY-MM-DD)
            forward_days: Number of trading days forward
            price_col: Price column to use (Close or Adjusted_close)
            
        Returns:
            Forward return as decimal (e.g., 0.05 for 5%) or None
        """
        df = self.get_price_data(ticker)
        if df is None or df.empty:
            return None
        
        try:
            df_dates = pd.to_datetime(df["Date"])
            t0 = pd.to_datetime(as_of_date)
            
            # Find first trading day >= as_of_date
            idx = df_dates.searchsorted(t0)
            if idx >= len(df):
                return None
            
            # Forward date index
            idx1 = idx + forward_days
            if idx1 >= len(df):
                return None
            
            p0 = df.loc[idx, price_col]
            p1 = df.loc[idx1, price_col]
            
            if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
                return None
            
            return float((p1 - p0) / p0)
            
        except Exception:
            return None
    
    def batch_get_forward_returns(
        self,
        samples: List[Tuple[str, str]],
        forward_days: int = 5,
        price_col: str = "Close"
    ) -> Dict[Tuple[str, str], Optional[float]]:
        """
        Batch calculate forward returns for multiple ticker/date pairs.
        
        Args:
            samples: List of (ticker, as_of_date) tuples
            forward_days: Number of trading days forward
            price_col: Price column to use
            
        Returns:
            Dictionary mapping (ticker, date) -> forward_return
        """
        results = {}
        
        # Group by ticker
        ticker_dates = {}
        for ticker, date in samples:
            if ticker not in ticker_dates:
                ticker_dates[ticker] = []
            ticker_dates[ticker].append(date)
        
        # Process each ticker
        for ticker, dates in ticker_dates.items():
            df = self.get_price_data(ticker)
            if df is None or df.empty:
                for date in dates:
                    results[(ticker, date)] = None
                continue
            
            for date in dates:
                ret = self._calculate_forward_return_from_df(
                    df, date, forward_days, price_col
                )
                results[(ticker, date)] = ret
        
        return results
    
    def _calculate_forward_return_from_df(
        self,
        df: pd.DataFrame,
        as_of_date: str,
        forward_days: int,
        price_col: str
    ) -> Optional[float]:
        """Helper to calculate forward return from pre-loaded dataframe."""
        try:
            df_dates = pd.to_datetime(df["Date"])
            t0 = pd.to_datetime(as_of_date)
            
            idx = df_dates.searchsorted(t0)
            if idx >= len(df):
                return None
            
            idx1 = idx + forward_days
            if idx1 >= len(df):
                return None
            
            p0 = df.loc[idx, price_col]
            p1 = df.loc[idx1, price_col]
            
            if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
                return None
            
            return float((p1 - p0) / p0)
            
        except Exception:
            return None


# Convenience functions
def get_price_client(**kwargs) -> PriceDataClient:
    """Get a configured price data client."""
    return PriceDataClient(**kwargs)


def get_forward_return(
    ticker: str,
    as_of_date: str,
    forward_days: int = 5,
    **kwargs
) -> Optional[float]:
    """Convenience function to get a single forward return."""
    client = get_price_client(**kwargs)
    return client.get_forward_return(ticker, as_of_date, forward_days)
