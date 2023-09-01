[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110//)
[![Format code](https://github.com/aoki-h-jp/py-liquidation-map/actions/workflows/Formatter.yml/badge.svg)](https://github.com/aoki-h-jp/py-liquidation-map/actions/workflows/Formatter.yml)

# py-liquidation-map
Visualize Liquidation Map from actual execution data. Supports for all historical data from binance and bybit. Receiving orders in real-time via websocket and drawing liquidation maps is being implemented.

## How to Understand a Liquidation Map
A liquidation map, also known as a "liq map," provides a visual chart of liquidations or liquidation risk in the futures cryptocurrency trading market. It displays liquidations that are predicted based on previous price trends.

When traders engage in trading on unregulated cryptocurrency derivative exchanges, they are constantly exposed to additional risks, namely liquidation risks. When the liquidation price of a trader's position is triggered, their position is forcibly closed by the exchange's risk engine.

The impact on the market is relatively small when a small number of positions are liquidated. However, if thousands of positions with similar liquidation prices are liquidated, the effect on the market price can be significant. Moreover, market buy and sell orders triggered by liquidations can cause rapid price movements, leading to a "cascading effect" where more nearby positions get liquidated. This phenomenon creates substantial price fluctuations (which institutional players often take advantage of as an entry strategy since the rapid injection of liquidity within a short period can meet the demand for institutional large orders).

Different combinations of leverage and time frames depict various clusters of liquidations. The denser and higher the liquidation clusters, the greater their impact on price behavior when reached.

## Installation

```bash
pip install git+https://github.com/aoki-h-jp/py-liquidation-map
```

## Usage
### Visualize liquidation map from historical data
Download binance BTCUSDT data from start_datetime to end_datetime and draw a liquidation map calculated from orders above threshold=100000 [USDT].
```python
from liqmap.mapping import HistoricalMapping

mapping = HistoricalMapping(
    start_datetime='2023-08-01 00:00:00',
    end_datetime='2023-08-01 06:00:00',
    symbol='BTCUSDT',
    exchange='binance',
)

mapping.liquidation_map_from_historical(
    mode="gross_value",
    threshold_gross_value=100000
)
```
### Output
![BTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_gross_value_100000.png](img%2FBTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_gross_value_100000.png)

### Visualize liquidation map depth
```python
from liqmap.mapping import HistoricalMapping

mapping = HistoricalMapping(
    start_datetime='2023-08-01 00:00:00',
    end_datetime='2023-08-01 06:00:00',
    symbol='BTCUSDT',
    exchange='binance',
)

mapping.liquidation_map_depth_from_historical(
    mode="gross_value",
    threshold_gross_value=100000
)
```

### Output
![BTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_gross_value_100000_depth.png](img%2FBTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_gross_value_100000_depth.png)

## Examples
### top_n mode
mode="top_n": draw liquidation map from top n trades.

threshold_top_n=100 means draw liquidation map from top 100 large trades.

```python
from liqmap.mapping import HistoricalMapping

mapping = HistoricalMapping(
    start_datetime='2023-08-01 00:00:00',
    end_datetime='2023-08-01 06:00:00',
    symbol='BTCUSDT',
    exchange='binance',
)

mapping.liquidation_map_from_historical(
    mode="top_n",
    threshold_top_n=100
)

mapping.liquidation_map_depth_from_historical(
    mode="top_n",
    threshold_top_n=100
)
```
![BTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_top_n_100.png](img%2FBTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_top_n_100.png)
![BTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_top_n_100_depth.png](img%2FBTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_top_n_100_depth.png)

### portion mode
mode="portion": draw liquidation map from top n% trades.

threshold_portion=0.01 means draw liquidation map from top 1% large trades.

```python
from liqmap.mapping import HistoricalMapping

mapping = HistoricalMapping(
    start_datetime='2023-08-01 00:00:00',
    end_datetime='2023-08-01 06:00:00',
    symbol='BTCUSDT',
    exchange='binance',
)

mapping.liquidation_map_from_historical(
    mode="portion",
    threshold_portion=0.01
)

mapping.liquidation_map_depth_from_historical(
    mode="portion",
    threshold_portion=0.01
)
```
![BTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_portion_0.01.png](img%2FBTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_portion_0.01.png)
![BTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_portion_0.01_depth.png](img%2FBTCUSDT_2023-08-01_00-00-00-2023-08-01_06-00-00_portion_0.01_depth.png)

## If you want to report a bug or request a feature
Please create an issue on this repository!

## Disclaimer
This project is for educational purposes only. You should not construe any such information or other material as legal, tax, investment, financial, or other advice. Nothing contained here constitutes a solicitation, recommendation, endorsement, or offer by me or any third party service provider to buy or sell any securities or other financial instruments in this or in any other jurisdiction in which such solicitation or offer would be unlawful under the securities laws of such jurisdiction.

Under no circumstances will I be held responsible or liable in any way for any claims, damages, losses, expenses, costs, or liabilities whatsoever, including, without limitation, any direct or indirect damages for loss of profits.
