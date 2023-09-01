import matplotlib.pyplot as plt
import pandas as pd

from liqmap.download import BinanceAggTradesDownload, BybitAggTradesDownloader
from liqmap.exceptions import ExchangeNotSupportedError


class HistoricalDraw:
    """
    Draw liquidation map from historical data.
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        exchange: str,
        **kwargs,
    ) -> None:
        """
        :param symbol: Trading symbol
        :param start_date: Start date for drawing
        :param end_date: End date for drawing
        :param exchange: Exchange name
        :param kwargs: Other parameters
        """
        self._symbol = symbol
        self._start_date = start_date
        self._end_date = end_date
        self._exchange = exchange
        self._kwargs = kwargs

    def _download(self) -> None:
        """
        download historical aggTrades data
        :return: None
        """
        # Download aggTrades
        if self._exchange == "binance":
            aggtrades = BinanceAggTradesDownload(
                destination_dir=".",
                download_symbol=self._symbol,
                start_date=self._start_date,
                end_date=self._end_date,
            )
        elif self._exchange == "bybit":
            aggtrades = BybitAggTradesDownloader(
                destination_dir=".",
                download_symbol=self._symbol,
                start_date=self._start_date,
                end_date=self._end_date,
            )
        else:
            raise ExchangeNotSupportedError(
                f"Exchange {self._exchange} is not supported."
            )
        aggtrades.download_aggtrades()

    # format aggTrades
    def _format_aggtrade_dataframe(self, filepath: str) -> pd.DataFrame:
        """
        Format aggTrades DataFrame
        :param filepath: Historical Data path
        :return: None
        """
        # Merge aggTrades into one dataframe
        aggtrades = pd.read_csv(filepath)
        if self._exchange == "binance":
            aggtrades["transact_time"] = pd.to_datetime(
                aggtrades["transact_time"], unit="ms"
            )
            aggtrades["price"] = aggtrades["price"].astype(float)
            aggtrades["quantity"] = aggtrades["quantity"].astype(float)
            aggtrades["is_buyer_maker"] = aggtrades["is_buyer_maker"].astype(bool)
            df = aggtrades[["transact_time", "price", "quantity", "is_buyer_maker"]]
        elif self._exchange == "bybit":
            aggtrades["transact_time"] = pd.to_datetime(
                aggtrades["transact_time"], unit="ms"
            )
            aggtrades["price"] = aggtrades["price"].astype(float)
            aggtrades["quantity"] = aggtrades["quantity"].astype(float)
            aggtrades["is_buyer_maker"] = aggtrades["is_buyer_maker"].astype(bool)
            df = aggtrades[["transact_time", "price", "quantity", "is_buyer_maker"]]
        else:
            raise ExchangeNotSupportedError(
                f"Exchange {self._exchange} is not supported."
            )

        return df

    def liquidation_map_from_historical(self, mode="", threshold=0):
        # Downloading historical data
        self._download()

        # Formatting historical data

        # Visualize liquidation map
