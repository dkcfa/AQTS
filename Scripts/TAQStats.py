# Author: Dennis Kim

import os
import scipy
import heapq
import pandas as pd
import numpy as np
import statistics as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


class TAQStats(object):
    """
    Calculates return statistics using adjusted or cleaned data
    """

    def __init__(self, data_file_path, ticker, adjusted_or_cleaned, max_k_lag, max_freq):
        """
        Initializes class and conducts all prep and calculation work automatically

        params:
        data_file_path: (str) full file path of ticker to conduct statistical analysis
        ticker: (str) ticker handle for the target stock to analyze
        x: (int) frequency in seconds
        adjusted_or_cleaned: (str) data to be analyzed
        """
        self.x = 1
        self.adjusted_or_cleaned = adjusted_or_cleaned.lower()
        self.parent_file_path = data_file_path
        self.ticker = ticker.upper()
        self.loaded_quotes_data = self.load_data('quotes')
        self.loaded_trades_data = self.load_data('trades')
        self.quote_df = self.loaded_quotes_data[1]
        self.trade_df = self.loaded_trades_data[1]
        self.quote_data_at_every_second = self.get_price_data_each_second(self.quote_df, 'quotes')
        self.trade_data_at_every_second = self.get_price_data_each_second(self.trade_df, 'trades')
        self.quote_data_at_x_interval = self.quote_data_at_every_second[::self.x]
        self.trade_data_at_x_interval = self.trade_data_at_every_second[::self.x]
        self.n_trades = len(self.trade_df)
        self.n_quotes = len(self.quote_df)
        self.sample_trade_days = self.loaded_trades_data[0]
        self.sample_quote_days = self.loaded_quotes_data[0]
        self.frac_trades_to_quotes = self.n_trades / self.n_quotes
        self.returns_lsts = self.x_sec_returns()
        self.quotes_stats = self.basic_statistics('quotes')
        self.trades_stats = self.basic_statistics('trades')

        # NOTE: everything below this line is run only on class instantiation
        self.k_lag_freq_list_quotes = self.get_lag_and_freq_combos('quotes', max_freq, max_k_lag)
        self.k_lag_freq_list_trades = self.get_lag_and_freq_combos('trades', max_freq, max_k_lag)

    def basic_statistics(self, quotes_or_trades='quotes'):
        """
        Calculates basis statistics: annualized mean/median/median absolute deviation returns, skew, kurtosis,
        ten largest and smallest returns, and max drawdown.

        params:
        quotes_or_trades: (str) designated tick record type to analyze

        returns:
        (list) list of tuples where each tuple has the statistic type and its value
        """

        # retrieves returns for appropriate record type and calculates annualizing factor
        returns = self.returns_lsts[1] if quotes_or_trades == 'quotes' else self.returns_lsts[0]
        annual_factor = 252 / self.sample_quote_days if quotes_or_trades == 'quotes' else 252 / self.sample_trade_days

        # annualized
        mean_return = (1 + stats.mean(returns)) ** annual_factor - 1
        med_return = (1 + stats.median(returns)) ** annual_factor - 1
        std_return = stats.stdev(returns) * (annual_factor ** .5)
        med_abs_dev_return = (1 + scipy.stats.median_abs_deviation(returns)) ** annual_factor - 1

        # nominal
        skew = scipy.stats.skew(np.array(returns))
        kurtosis = scipy.stats.kurtosis(returns)
        ten_largest_returns = heapq.nlargest(10, returns)
        ten_smallest_returns = heapq.nsmallest(10, returns)
        max_drawdown = self.max_drawdown(returns)

        return [('ann_mean_return', mean_return),
                ('ann_med_return', med_return),
                ('ann_std_return', std_return),
                ('ann_med_abs_dev_return', med_abs_dev_return),
                ('skew', skew),
                ('kurtosis', kurtosis),
                ('ten_largest_returns', ten_largest_returns),
                ('ten_smallest_returns', ten_smallest_returns),
                ('max_drawdown', max_drawdown)
                ]

    def x_sec_returns(self):
        """
        Calculates the returns for the indicates frequency in load_data(), x (in seconds).

        returns:
        (tuple) tuple consisting of trade and quote returns
        """

        # NOTE: this logic will provide returns in decimals, not percentages
        trade_returns = np.diff(self.trade_data_at_x_interval) / np.array(self.trade_data_at_x_interval[:-1])
        quote_returns = np.diff(self.quote_data_at_x_interval) / np.array(self.quote_data_at_x_interval[:-1])

        return trade_returns, quote_returns

    def load_data(self, quotes_or_trades='quotes'):
        """
        Loads in the raw data

        params:
        quotes_or_trades: (str) type of record to load

        returns:
        (tuple) tuple of number of days of available records and a pandas dataframe of available tick records
        """
        quotes_or_trades = quotes_or_trades.lower()
        dates_to_traverse = self.get_folders_to_traverse(os.path.join(self.parent_file_path, quotes_or_trades))
        day_ct = 0
        lst_of_dataframes = []
        quote_cols = ['millis from epoc', 'Ask Size', 'Ask Price', 'Bid Size', 'Bid Price']
        trade_cols = ['millis from epoc', 'Size', 'Price']
        cols_to_extract = quote_cols if quotes_or_trades == 'quotes' else trade_cols

        # traverse each date folder of records
        for date_folder_path in dates_to_traverse:
            file_name = self.ticker + "_" + quotes_or_trades + "_" + self.adjusted_or_cleaned + "_data.feather"
            file_path = os.path.join(date_folder_path, file_name)
            if os.path.exists(file_path):
                lst_of_dataframes.append(pd.read_feather(file_path, columns=cols_to_extract))
                day_ct += 1
            else:
                continue

        # concatenate all dataframes, reset the index, drop unnecessary 'index' column
        return day_ct, pd.concat(lst_of_dataframes).reset_index().drop('index', axis=1)

    def get_price_data_each_second(self, df, quotes_or_trades):
        """
        Retrieves price data, quotes or trades, for each second

        params:
        df: (Pandas df) dataframe to extract second-by-second data
        quotes_or_trades: (str) the record type

        returns:
        (list) a list of prices
        """

        prices = []

        # get appropriate start and end time stamps
        start_stamp = df['millis from epoc'].iloc[0]
        end_stamp = df['millis from epoc'].iloc[-1]
        time_stamp = start_stamp

        # get the start of day records
        df_filt_by_stamp = df.loc[df['millis from epoc'] == time_stamp]
        record = self.get_record(df_filt_by_stamp, quotes_or_trades)
        prices.append(record)
        time_stamp += 1000

        # iterate through rows by x-second frequencies
        while time_stamp <= end_stamp:
            df_filt_by_stamp = df.loc[df['millis from epoc'] == time_stamp]

            # get the correct record
            # forward fill the data if the specific time stamp DNE
            if len(df_filt_by_stamp) != 0:
                record = self.get_record(df_filt_by_stamp, quotes_or_trades)
            time_stamp += 1000
            prices.append(record)

        return prices

    def set_x(self, new_x):
        """
        Setter to change returns frequency. Calls _update() method when invoked

        params:
        new_x: (int) input is in second

        returns:
        nothing
        """

        # validating input and updating the necessary instance variables
        self.x = abs(new_x)
        self._update_class()

    def _update_class(self):
        """
        Updates the class when a new frequency is designated

        returns:
        nothing
        """

        self.quote_data_at_x_interval = self.quote_data_at_every_second[::self.x]
        self.trade_data_at_x_interval = self.trade_data_at_every_second[::self.x]
        self.sample_trade_days = self.loaded_trades_data[0]
        self.sample_quote_days = self.loaded_quotes_data[0]
        self.frac_trades_to_quotes = self.n_trades / self.n_quotes
        self.returns_lsts = self.x_sec_returns()
        self.quotes_stats = self.basic_statistics('quotes')
        self.trades_stats = self.basic_statistics('trades')

    def check_autocorr(self, k_lag, quotes_or_trades, alpha=.05):
        """
        Performs Ljung-Box test and returns results

        params:
        k_lag: (int) the size of lag to test
        quotes_or_trades: (str) record type
        alpha: (float) confidence level in decimal

        returns:
        (int) 1 for evidence of autocorrelation, 0 otherwise
        """

        quotes_or_trades_idx = 1 if quotes_or_trades == 'quotes' else 0
        df = acorr_ljungbox(self.returns_lsts[quotes_or_trades_idx], lags=[k_lag])

        return 1 if df['lb_pvalue'].iloc[0] < alpha else 0

    def get_lag_and_freq_combos(self, quotes_or_trades, max_freq=30, max_k_lag=100, alpha=.05):
        """
        Iterate through different k-lag and frequency to find combinations that do not experience autocorrelation.
        Note: this indirectly runs Ljung-Box, but this method directly calls an augmented Dickey-Fuller test
        to test for stationarity on frequency/k-lag combinations that do not suffer autocorrelation.

        params:
        quotes_or_trades: (str) record type
        max_freq: (int) maximum frequency of returns to test
        max_k_lag: (int) max k_lag to test
        alpha: (float) confidence level

        returns:
        (list) list of tuples of the form (frequency, k_lag) that exhibit no evidence of autocorrelation
        """

        returns = self.returns_lsts[1] if quotes_or_trades == 'quotes' else self.returns_lsts[0]
        combos = []

        # iterate through every frequency
        for freq in range(1, max_freq):
            self.set_x(freq)

            # iterate through every k_lag
            for k_lag in range(1, max_k_lag):
                if self.check_autocorr(k_lag, quotes_or_trades, alpha) == 0:  # if no autocorrelation
                    adfuller_results = adfuller(returns)  # run augmented Dickey Fuller test

                    if adfuller_results[1] < alpha:
                        combos.append((freq, k_lag))  # add a valid freq, k_lag combo that is also stationary

        return combos

    @staticmethod
    def max_drawdown(rets_lst):
        """
        Calculates the max drawdown of a portfolio

        params:
        rets_lst: (list) list of returns

        returns:
        (float) max drawdown in decimal form
        """

        # Initialize variables
        peak = 1.0  # Start with the initial peak value of 1.0
        max_drawdown = 0.0

        # Iterate over the returns
        for ret in rets_lst:
            equity = peak * (1 + ret)  # Calculate the equity based on the return

            if equity > peak:
                peak = equity  # Update the peak value
            else:
                drawdown = (peak - equity) / peak  # Calculate the drawdown
                if drawdown > max_drawdown:
                    max_drawdown = drawdown  # Update the max drawdown

        return max_drawdown

    @staticmethod
    def get_folders_to_traverse(file_path):
        """
        Retrieve list of full path folders to traverse

        params:
        file_path: (str) full directory path
        returns:
        (list) list of full path folders to traverse
        """

        return [f.path for f in os.scandir(file_path) if f.is_dir()]

    @staticmethod
    def get_feather_file_ticker(ticker_path):
        """
        Gets the stock ticker handle

        param:
        ticker_path: (str) full file path of specific ticker data
        returns:
        (str) the ticker handle
        """

        return ticker_path.rsplit("\\", 1)[-1].split("_")[0]

    @staticmethod
    def get_record(df_filt_by_stamp, quotes_or_trades):
        """
        Gets the appropriate records for loading price data

        params:
        df_filt_by_stamp: (Pandas df) dataframe
        quotes_or_trades: (str) type of record to return

        returns:
        (float) the appropriate record (e.g., midquote for quotes data and price for trades data)
        """

        if quotes_or_trades == 'quotes':
            ask_price = df_filt_by_stamp['Ask Price'].iloc[0]
            bid_price = df_filt_by_stamp['Bid Price'].iloc[0]
            record = (ask_price + bid_price) / 2
        else:
            record = df_filt_by_stamp['Price'].iloc[0]

        return record
