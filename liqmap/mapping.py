# import standard libraries
import datetime
import math
import warnings

# import third-party libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from download import BinanceAggTradesDownload, BybitAggTradesDownloader
from exceptions import ExchangeNotSupportedError, InvalidParamError
from rich.progress import track

warnings.filterwarnings("ignore")


class HistoricalMapping:
    """
    Map liquidation map from historical data.
    """

    def __init__(
        self,
        symbol: str,
        start_datetime: str,
        end_datetime: str,
        exchange: str,
    ) -> None:
        """
        :param symbol: Trading symbol
        :param start_datetime: Start datetime for drawing
        :param end_datetime: End datetime for drawing
        :param exchange: Exchange name
        """
        self._symbol = symbol
        self._start_datetime = start_datetime
        self._end_datetime = end_datetime
        self._exchange = exchange
        self._downloaded_list = []

    def _make_start_date(self) -> str:
        """
        Make start date from string
        :return: datetime
        """
        return datetime.datetime.strptime(
            self._start_datetime, "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y-%m-%d")

    def _make_end_date(self) -> str:
        """
        Make end date from string
        :return: datetime
        """
        return datetime.datetime.strptime(
            self._end_datetime, "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y-%m-%d")

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
                start_date=self._make_start_date(),
                end_date=self._make_end_date(),
            )
        elif self._exchange == "bybit":
            aggtrades = BybitAggTradesDownloader(
                destination_dir=".",
                download_symbol=self._symbol,
                start_date=self._make_start_date(),
                end_date=self._make_end_date(),
            )
        else:
            raise ExchangeNotSupportedError(
                f"Exchange {self._exchange} is not supported."
            )
        aggtrades.download_aggtrades()

    def _make_prefix_list(self) -> list:
        """
        Make prefix list
        :return: list of prefix
        """
        prefix_list = []
        if self._exchange == "binance":
            for date in pd.date_range(self._make_start_date(), self._make_end_date()):
                prefix_list.append(
                    f"data/futures/um/daily/aggTrades/{self._symbol}/{self._symbol}-aggTrades-{date.strftime('%Y-%m-%d')}.csv"
                )
        elif self._exchange == "bybit":
            for date in pd.date_range(self._make_start_date(), self._make_end_date()):
                prefix_list.append(
                    f"bybit_data/trading/{self._symbol}/{self._symbol}{date.strftime('%Y-%m-%d')}.csv"
                )
        else:
            raise ExchangeNotSupportedError(
                f"Exchange {self._exchange} is not supported."
            )

        return prefix_list

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
            headers = [
                "agg_trade_id",
                "price",
                "quantity",
                "first_trade_id",
                "last_trade_id",
                "transact_time",
                "is_buyer_maker",
            ]
            if aggtrades.columns.tolist() != headers:
                aggtrades = pd.read_csv(filepath, header=None)
                aggtrades.columns = headers
            aggtrades["timestamp"] = pd.to_datetime(
                aggtrades["transact_time"], unit="ms"
            )
            aggtrades["price"] = aggtrades["price"].astype(float)
            aggtrades["size"] = aggtrades["quantity"].astype(float)
            aggtrades["is_buyer_maker"] = aggtrades["is_buyer_maker"].astype(bool)
            aggtrades["side"] = aggtrades["is_buyer_maker"].apply(
                lambda x: "Buy" if x is False else "Sell"
            )
            aggtrades["amount"] = aggtrades["price"] * aggtrades["size"]
        elif self._exchange == "bybit":
            aggtrades["timestamp"] = pd.to_datetime(
                aggtrades["timestamp"] * 1000, unit="ms"
            )
            aggtrades["price"] = aggtrades["price"].astype(float)
            aggtrades["size"] = aggtrades["size"].astype(float)
            aggtrades["side"] = aggtrades["side"].astype(str)
            aggtrades["amount"] = aggtrades["price"] * aggtrades["size"]
        else:
            raise ExchangeNotSupportedError(
                f"Exchange {self._exchange} is not supported."
            )
        df = aggtrades[["timestamp", "price", "size", "side", "amount"]]
        df = df.sort_values(by="timestamp")

        return df

    @staticmethod
    def human_format(x, pos):
        if x < 1e6:
            return str(x)
        elif x < 1e9:
            return "{:.1f}M".format(x * 1e-6)
        else:
            return "{:.1f}B".format(x * 1e-9)

    def _make_merged_dataframe(self) -> pd.DataFrame:
        """
        Make merged dataframe
        :return:
        """
        # Formatting historical data
        df_merged = pd.DataFrame()
        for prefix in track(self._make_prefix_list(), description="Formatting data"):
            df_prefix = self._format_aggtrade_dataframe(prefix)
            df_merged = pd.concat([df_merged, df_prefix])

        df_merged = df_merged.sort_values(by="timestamp")
        df_merged = df_merged.reset_index(drop=True)
        df_merged = df_merged[df_merged["timestamp"] <= self._end_datetime]
        df_merged = df_merged[df_merged["timestamp"] >= self._start_datetime]

        return df_merged


    def _make_buy_dataframe(self) -> pd.DataFrame:
        """
        Make buy dataframe
        :return: pd.DataFrame
        """
        df_merged = self._make_merged_dataframe()
        df_buy = df_merged[df_merged["side"] == "Buy"]

        df_buy["LossCut100x"] = df_buy["price"] * 0.99
        df_buy["LossCut50x"] = df_buy["price"] * 0.98
        df_buy["LossCut25x"] = df_buy["price"] * 0.96
        df_buy["LossCut10x"] = df_buy["price"] * 0.90

        return df_buy

    def _make_sell_dataframe(self) -> pd.DataFrame:
        """
        Make buy dataframe
        :return: pd.DataFrame
        """
        df_merged = self._make_merged_dataframe()
        df_sell = df_merged[df_merged["side"] == "Sell"]

        df_sell["LossCut100x"] = df_sell["price"] * 1.01
        df_sell["LossCut50x"] = df_sell["price"] * 1.02
        df_sell["LossCut25x"] = df_sell["price"] * 1.04
        df_sell["LossCut10x"] = df_sell["price"] * 1.10

        return df_sell


    def liquidation_map_from_historical(
        self, mode="gross_value", threshold_gross_value=100000, threshold_top_n=100, threshold_portion=0.01
    ) -> None:
        """
        Draw liquidation map from historical data
        :param mode: draw mode "gross_value", "top_n", "portion" is available.
        "gross_value": draw liquidation map from above threshold gross value trades
        "top_n": draw liquidation map from top n trades
        "portion": draw liquidation map from top n% trades
        example:
        threshold_gross_value=100000 means draw liquidation map from trades whose gross value is above 100000 USDT.
        threshold_top_n=100 means draw liquidation map from top 100 large trades.
        threshold_portion=0.01 means draw liquidation map from top 1% large trades.
        :param threshold_gross_value: threshold for gross value
        :param threshold_top_n: threshold for top n
        :param threshold_portion: threshold for top n%
        :return:
        """
        # Downloading historical data
        self._download()

        # Formatting historical data
        df_merged = self._make_merged_dataframe()
        df_buy = self._make_buy_dataframe()
        df_sell = self._make_sell_dataframe()

        # mode: gross_value
        if mode == "gross_value":
            df_buy = df_buy[df_buy["amount"] >= threshold_gross_value]
            df_sell = df_sell[df_sell["amount"] >= threshold_gross_value]
        elif mode == "top_n":
            print("passed")
            df_buy = df_buy.sort_values(by="amount", ascending=False)
            df_buy = df_buy.iloc[:threshold_top_n]
            df_sell = df_sell.sort_values(by="amount", ascending=False)
            df_sell = df_sell.iloc[:threshold_top_n]
        elif mode == "portion":
            df_buy = df_buy.sort_values(by="amount", ascending=False)
            df_buy = df_buy.iloc[:int(len(df_buy) * threshold_portion)]
            df_sell = df_sell.sort_values(by="amount", ascending=False)
            df_sell = df_sell.iloc[:int(len(df_sell) * threshold_portion)]
        else:
            raise InvalidParamError(f"mode {mode} is not supported.")

        # Visualize liquidation map
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(9, 9))
        # draw price on ax1
        for i, dt in enumerate(df_buy["timestamp"]):
            label = "large BUY LONG" if i == 0 else None
            ax1.scatter(
                dt,
                df_buy.iloc[i, 1],
                s=100,
                facecolor="None",
                edgecolors="b",
                label=label,
            )
        for i, dt in enumerate(df_sell["timestamp"]):
            label = "large SELL SHORT" if i == 0 else None
            ax1.scatter(
                dt,
                df_sell.iloc[i, 1],
                s=100,
                facecolor="None",
                edgecolors="r",
                label=label,
            )

        ax1.plot(df_merged["timestamp"], df_merged["price"], c="k", label="price")
        ax1.set_xlabel("datetime")
        ax1.set_ylabel("price [USDT]")
        ax1.tick_params(axis="x", labelrotation=45)
        ax1.set_xlim(
            [
                datetime.datetime.strptime(self._start_datetime, "%Y-%m-%d %H:%M:%S"),
                datetime.datetime.strptime(self._end_datetime, "%Y-%m-%d %H:%M:%S"),
            ]
        )
        title = f"{self._symbol}\n{self._start_datetime} -> {self._end_datetime}"
        if mode == "gross_value":
            title += f"\nthreshold: >= {threshold_gross_value} [USDT]"
        ax1.set_title(title)
        ax1.legend()

        # Buy liquidation map on ax2
        df_losscut_10x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_buy.index):
            df_losscut_10x.loc[len(df_losscut_10x)] = {
                "price": df_buy.iloc[i, 8],
                "amount": df_buy.iloc[i, 4],
            }

        df_losscut_25x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_buy.index):
            df_losscut_25x.loc[len(df_losscut_25x)] = {
                "price": df_buy.iloc[i, 7],
                "amount": df_buy.iloc[i, 4],
            }

        df_losscut_50x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_buy.index):
            df_losscut_50x.loc[len(df_losscut_50x)] = {
                "price": df_buy.iloc[i, 6],
                "amount": df_buy.iloc[i, 4],
            }

        df_losscut_100x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_buy.index):
            df_losscut_100x.loc[len(df_losscut_100x)] = {
                "price": df_buy.iloc[i, 5],
                "amount": df_buy.iloc[i, 4],
            }

        current_price = df_merged.iloc[-1, 1]

        df_losscut_list = [
            df_losscut_10x,
            df_losscut_25x,
            df_losscut_50x,
            df_losscut_100x,
        ]
        labels = ["10x Leveraged", "25x Leveraged", "50x Leveraged", "100x Leveraged"]
        colors = ["r", "g", "b", "y"]
        tick_degits = 2 - math.ceil(
            math.log10(df_merged["price"].max() - df_merged["price"].min())
        )
        max_amount = 0
        for i, df_losscut in enumerate(df_losscut_list):
            df_losscut = df_losscut[df_losscut["price"] <= current_price]
            g_ids = int(
                (
                    round(df_losscut["price"].max(), tick_degits)
                    - round(df_losscut["price"].min(), tick_degits)
                )
                * 10**tick_degits
            )
            bins = [
                round(
                    round(df_losscut["price"].min(), tick_degits)
                    + i * 10**-tick_degits,
                    tick_degits,
                )
                for i in range(g_ids)
            ]
            df_losscut["group_id"] = pd.cut(df_losscut["price"], bins=bins)
            agg_df = df_losscut.groupby("group_id").sum()
            ax2.barh(
                [f.left for f in agg_df.index],
                agg_df["amount"],
                height=10**-tick_degits,
                color=colors[i],
                label=labels[i],
                alpha=0.5,
            )
            if agg_df["amount"].max() > max_amount:
                max_amount = agg_df["amount"].max()

        # Save liquidation map data as csv
        save_title = f"{self._symbol}_{self._start_datetime.replace(' ', '_').replace(':', '-')}-{self._end_datetime.replace(' ', '_').replace(':', '-')}"
        if mode == "gross_value":
            save_title += f"_gross_value_{threshold_gross_value}.png"
        elif mode == "top_n":
            save_title += f"_top_n_{threshold_top_n}.png"
        elif mode == "portion":
            save_title += f"_portion_{threshold_portion}.png"
        else:
            raise InvalidParamError(f"mode {mode} is not supported.")

        for df_l, label in zip(df_losscut_list, labels):
            df_l.to_csv(
                f"{save_title.replace('.png', '')}_{label.replace(' ','_')}_buy.csv"
            )

        # Sell liquidation map on ax2
        df_losscut_10x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_sell.index):
            df_losscut_10x.loc[len(df_losscut_10x)] = {
                "price": df_sell.iloc[i, 8],
                "amount": df_sell.iloc[i, 4],
            }

        df_losscut_25x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_sell.index):
            df_losscut_25x.loc[len(df_losscut_25x)] = {
                "price": df_sell.iloc[i, 7],
                "amount": df_sell.iloc[i, 4],
            }

        df_losscut_50x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_sell.index):
            df_losscut_50x.loc[len(df_losscut_50x)] = {
                "price": df_sell.iloc[i, 6],
                "amount": df_sell.iloc[i, 4],
            }

        df_losscut_100x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_sell.index):
            df_losscut_100x.loc[len(df_losscut_100x)] = {
                "price": df_sell.iloc[i, 5],
                "amount": df_sell.iloc[i, 4],
            }

        current_price = df_merged.iloc[-1, 1]

        df_losscut_list = [
            df_losscut_10x,
            df_losscut_25x,
            df_losscut_50x,
            df_losscut_100x,
        ]
        labels = ["10x Leveraged", "25x Leveraged", "50x Leveraged", "100x Leveraged"]
        colors = ["r", "g", "b", "y"]
        tick_degits = 2 - math.ceil(
            math.log10(df_merged["price"].max() - df_merged["price"].min())
        )
        max_amount = 0
        for i, df_losscut in enumerate(df_losscut_list):
            df_losscut = df_losscut[df_losscut["price"] >= current_price]
            g_ids = int(
                (
                    round(df_losscut["price"].max(), tick_degits)
                    - round(df_losscut["price"].min(), tick_degits)
                )
                * 10**tick_degits
            )
            bins = [
                round(
                    round(df_losscut["price"].min(), tick_degits)
                    + i * 10**-tick_degits,
                    tick_degits,
                )
                for i in range(g_ids)
            ]
            df_losscut["group_id"] = pd.cut(df_losscut["price"], bins=bins)
            agg_df = df_losscut.groupby("group_id").sum()
            ax2.barh(
                [f.left for f in agg_df.index],
                agg_df["amount"],
                height=10**-tick_degits,
                color=colors[i],
                alpha=0.5,
            )
            if agg_df["amount"].max() > max_amount:
                max_amount = agg_df["amount"].max()

        ax2.annotate(
            "",
            xytext=(max_amount, current_price),
            xy=(0, current_price),
            arrowprops=dict(
                arrowstyle="->,head_length=1,head_width=0.5", lw=2, linestyle="dashed"
            ),
            label="Current Price",
        )
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(self.human_format))
        ax2.set_title("Estimated Liquidation Amount")
        ax2.set_xlabel("Amount [USDT]")
        ax2.tick_params(axis="x", labelrotation=45)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_title)
        plt.close()

        # Save liquidation map data as csv
        for df_l, label in zip(df_losscut_list, labels):
            df_l.to_csv(
                f"{save_title.replace('.png', '')}_{label.replace(' ','_')}_sell.csv"
            )

    def liquidation_map_depth_from_historical(
        self, mode="gross_value", threshold_gross_value=10000, threshold_top_n=100, threshold_portion=0.010
    ) -> None:
        """
        Draw liquidation map depth from historical data
        :param mode: draw mode "gross_value", "top_n", "portion" is available.
        "gross_value": draw liquidation map from above threshold gross value trades
        "top_n": draw liquidation map from top n trades
        "portion": draw liquidation map from top n% trades
        example:
        threshold_gross_value=100000 means draw liquidation map from trades whose gross value is above 100000 USDT.
        threshold_top_n=100 means draw liquidation map from top 100 large trades.
        threshold_portion=0.01 means draw liquidation map from top 1% large trades.
        :param threshold_gross_value: threshold for gross value
        :param threshold_top_n: threshold for top n
        :param threshold_portion: threshold for top n%
        :return:
        """
        # Downloading historical data
        self._download()

        # Formatting historical data
        df_merged = self._make_merged_dataframe()
        df_buy = self._make_buy_dataframe()
        df_sell = self._make_sell_dataframe()

        # mode: gross_value
        if mode == "gross_value":
            df_buy = df_buy[df_buy["amount"] >= threshold_gross_value]
            df_sell = df_sell[df_sell["amount"] >= threshold_gross_value]
        elif mode == "top_n":
            df_buy = df_buy.sort_values(by="amount", ascending=False)
            df_buy = df_buy.iloc[:threshold_top_n]
            df_sell = df_sell.sort_values(by="amount", ascending=False)
            df_sell = df_sell.iloc[:threshold_top_n]
        elif mode == "portion":
            df_buy = df_buy.sort_values(by="amount", ascending=False)
            df_buy = df_buy.iloc[:int(len(df_buy) * threshold_portion)]
            df_sell = df_sell.sort_values(by="amount", ascending=False)
            df_sell = df_sell.iloc[:int(len(df_sell) * threshold_portion)]
        else:
            raise InvalidParamError(f"mode {mode} is not supported.")

        # Visualize liquidation map depth
        fig = plt.figure(figsize=(9, 9))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        plt.xlabel("datetime")
        plt.ylabel("volume [USDT]")
        plt.xticks(rotation=45)
        title = f"{self._symbol}\n{self._start_datetime} -> {self._end_datetime}"
        if mode == "gross_value":
            title += f"\nthreshold: >= {threshold_gross_value} [USDT]"
        plt.title(title)

        df_losscut_10x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_buy.index):
            df_losscut_10x.loc[len(df_losscut_10x)] = {
                "price": df_buy.iloc[i, 8],
                "amount": df_buy.iloc[i, 4],
            }

        df_losscut_25x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_buy.index):
            df_losscut_25x.loc[len(df_losscut_25x)] = {
                "price": df_buy.iloc[i, 7],
                "amount": df_buy.iloc[i, 4],
            }

        df_losscut_50x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_buy.index):
            df_losscut_50x.loc[len(df_losscut_50x)] = {
                "price": df_buy.iloc[i, 6],
                "amount": df_buy.iloc[i, 4],
            }

        df_losscut_100x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_buy.index):
            df_losscut_100x.loc[len(df_losscut_100x)] = {
                "price": df_buy.iloc[i, 5],
                "amount": df_buy.iloc[i, 4],
            }

        current_price = df_merged.iloc[-1, 1]

        df_depth_buy = pd.concat(
            [df_losscut_10x, df_losscut_25x, df_losscut_50x, df_losscut_100x],
            ignore_index=True,
        )
        df_depth_buy = df_depth_buy.sort_values(by="price", ascending=False)
        df_depth_buy = df_depth_buy[df_depth_buy["price"] <= current_price]
        df_depth_buy = df_depth_buy.reset_index(drop=True)
        df_depth_buy["price"] = df_depth_buy["price"].astype(float)
        df_depth_buy["cumsum"] = df_depth_buy["amount"].cumsum().astype(float)
        ax1.plot(df_depth_buy["price"], df_depth_buy["cumsum"], label="buy", c="b")

        df_losscut_list = [
            df_losscut_10x,
            df_losscut_25x,
            df_losscut_50x,
            df_losscut_100x,
        ]
        labels = ["10x Leveraged", "25x Leveraged", "50x Leveraged", "100x Leveraged"]
        colors = ["r", "g", "b", "y"]
        tick_degits = 2 - math.ceil(
            math.log10(df_merged["price"].max() - df_merged["price"].min())
        )
        max_amount = 0
        for i, df_losscut in enumerate(df_losscut_list):
            df_losscut = df_losscut[df_losscut["price"] <= current_price]
            g_ids = int(
                (
                    round(df_losscut["price"].max(), tick_degits)
                    - round(df_losscut["price"].min(), tick_degits)
                )
                * 10**tick_degits
            )
            bins = [
                round(
                    round(df_losscut["price"].min(), tick_degits)
                    + i * 10**-tick_degits,
                    tick_degits,
                )
                for i in range(g_ids)
            ]
            df_losscut["group_id"] = pd.cut(df_losscut["price"], bins=bins)
            agg_df = df_losscut.groupby("group_id").sum()
            ax2.bar(
                x=[f.left for f in agg_df.index],
                height=agg_df["amount"],
                width=10**-tick_degits,
                color=colors[i],
                label=labels[i],
                alpha=0.5,
            )
            if agg_df["amount"].max() > max_amount:
                max_amount = agg_df["amount"].max()

        # Save liquidation map data as csv
        save_title = f"{self._symbol}_{self._start_datetime.replace(' ', '_').replace(':', '-')}-{self._end_datetime.replace(' ', '_').replace(':', '-')}"
        if mode == "gross_value":
            save_title += f"_gross_value_{threshold_gross_value}_depth.png"
        elif mode == "top_n":
            save_title += f"_top_n_{threshold_top_n}_depth.png"
        elif mode == "portion":
            save_title += f"_portion_{threshold_portion}_depth.png"
        else:
            raise InvalidParamError(f"mode {mode} is not supported.")


        for df_l, label in zip(df_losscut_list, labels):
            df_l.to_csv(
                f"{save_title.replace('.png', '')}_{label.replace(' ', '_')}_buy.csv"
            )

        # Sell liquidation map on ax2
        df_losscut_10x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_sell.index):
            df_losscut_10x.loc[len(df_losscut_10x)] = {
                "price": df_sell.iloc[i, 8],
                "amount": df_sell.iloc[i, 4],
            }

        df_losscut_25x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_sell.index):
            df_losscut_25x.loc[len(df_losscut_25x)] = {
                "price": df_sell.iloc[i, 7],
                "amount": df_sell.iloc[i, 4],
            }

        df_losscut_50x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_sell.index):
            df_losscut_50x.loc[len(df_losscut_50x)] = {
                "price": df_sell.iloc[i, 6],
                "amount": df_sell.iloc[i, 4],
            }

        df_losscut_100x = pd.DataFrame(columns=["price", "amount"])
        for i, dt in enumerate(df_sell.index):
            df_losscut_100x.loc[len(df_losscut_100x)] = {
                "price": df_sell.iloc[i, 5],
                "amount": df_sell.iloc[i, 4],
            }

        current_price = df_merged.iloc[-1, 1]

        df_depth_sell = pd.concat(
            [df_losscut_10x, df_losscut_25x, df_losscut_50x, df_losscut_100x],
            ignore_index=True,
        )
        df_depth_sell = df_depth_sell.sort_values(by="price")
        df_depth_sell = df_depth_sell[df_depth_sell["price"] >= current_price]
        df_depth_sell = df_depth_sell.reset_index(drop=True)
        df_depth_sell["cumsum"] = df_depth_sell["amount"].cumsum()
        ax1.plot(df_depth_sell["price"], df_depth_sell["cumsum"], label="sell", c="r")

        df_losscut_list = [
            df_losscut_10x,
            df_losscut_25x,
            df_losscut_50x,
            df_losscut_100x,
        ]
        labels = ["10x Leveraged", "25x Leveraged", "50x Leveraged", "100x Leveraged"]
        colors = ["r", "g", "b", "y"]
        tick_degits = 2 - math.ceil(
            math.log10(df_merged["price"].max() - df_merged["price"].min())
        )
        max_amount = 0
        for i, df_losscut in enumerate(df_losscut_list):
            df_losscut = df_losscut[df_losscut["price"] >= current_price]
            g_ids = int(
                (
                    round(df_losscut["price"].max(), tick_degits)
                    - round(df_losscut["price"].min(), tick_degits)
                )
                * 10**tick_degits
            )
            bins = [
                round(
                    round(df_losscut["price"].min(), tick_degits)
                    + i * 10**-tick_degits,
                    tick_degits,
                )
                for i in range(g_ids)
            ]
            df_losscut["group_id"] = pd.cut(df_losscut["price"], bins=bins)
            agg_df = df_losscut.groupby("group_id").sum()
            ax2.bar(
                x=[f.left for f in agg_df.index],
                height=agg_df["amount"],
                width=10**-tick_degits,
                color=colors[i],
                alpha=0.5,
            )
            if agg_df["amount"].max() > max_amount:
                max_amount = agg_df["amount"].max()

        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(self.human_format))
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(self.human_format))
        plt.annotate(
            "",
            xytext=(current_price, max_amount),
            xy=(current_price, 0),
            arrowprops=dict(
                arrowstyle="->,head_length=1,head_width=0.5", lw=2, linestyle="dashed"
            ),
            label="Current Price",
        )
        plt.legend(loc="upper right")
        plt.xlabel("Price [USDT]")
        plt.tight_layout()
        plt.savefig(save_title)
        plt.close()

        # Save liquidation map data as csv
        for df_l, label in zip(df_losscut_list, labels):
            df_l.to_csv(
                f"{save_title.replace('.png', '')}_{label.replace(' ', '_')}_sell.csv"
            )
