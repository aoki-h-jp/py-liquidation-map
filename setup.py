from setuptools import setup

setup(
    name="py-liquidation-map",
    version="1.0.0",
    description="Visualize Liquidation Map from actual execution data.",
    install_requires=[
        "binance-bulk-downloader @ git+https://github.com/aoki-h-jp/binance-bulk-downloader",
        "bybit-bulk-downloader @ git+https://github.com/aoki-h-jp/bybit-bulk-downloader",
    ],
    author="aoki-h-jp",
    author_email="aoki.hirotaka.biz@gmail.com",
    license="MIT",
    packages=["liqmap"],
)
