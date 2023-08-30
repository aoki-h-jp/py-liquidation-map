from binance_bulk_downloader.downloader import BinanceBulkDownloader
import os
import pandas as pd
from rich.progress import track
from concurrent.futures import ThreadPoolExecutor


class BinanceAggTradesDownload(BinanceBulkDownloader):
    def __init__(self, destination_dir='.', download_symbol='BTCUSDT', start_date='2021-01-01', end_date='2021-01-31') -> None:
        """
        :param destination_dir: Destination directory for downloaded files
        :param download_symbol: download symbol
        :param start_date: Start date for download
        :param end_date: End date for download
        """
        super().__init__(destination_dir=destination_dir, data_type='aggTrades', asset='um', timeperiod_per_file='daily')
        self.super = super()
        self._symbol = download_symbol
        self._start_date = start_date
        self._end_date = end_date

    def _make_filename(self, date: str) -> str:
        """
        make filename for aggTrades
        :param date: date string
        :return: filename
        """
        return f'{self._symbol}-aggTrades-{date}.zip'

    def _make_date_range(self) -> list:
        """
        make date range for aggTrades
        :return: date range
        """
        return pd.date_range(start=self._start_date, end=self._end_date, freq='D').strftime('%Y-%m-%d').tolist()

    def download_aggtrades(self) -> None:
        """
        download aggTrades concurrently
        :return: None
        """
        download_prefixes = ["/".join([self.super._build_prefix(), self._symbol, self._make_filename(date=dt)]) for dt in self._make_date_range()]
        for prefix in track(self.super.make_chunks(download_prefixes, self.super._CHUNK_SIZE), description="Downloading aggTrades"):
            with ThreadPoolExecutor() as executor:
                executor.map(self.super._download, prefix)
