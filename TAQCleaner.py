# Author: Dennis Kim

import os
import pandas as pd
import numpy as np
import statistics as stats
import multiprocessing as mp


class TAQCleaner(object):
    """
    Takes in adjusted data in feather file format, cleans the data, and saves cleaned data in feather file format

    Note: the logic structure here depends on a specific folder/subfolder structure. Future users need
    to adjust source and destination paths for their specific setup. This is also a long process.
    """

    def __init__(self, parent_path, k_period=6, gamma_multiplier=.0005, source_file_type='feather',
                 dest_file_type='feather'):
        """
        Initializes the class and performs all cleaning automatically

        params:
        parent_path: (str) full file path to feather file types housing adjusted data
        k_period: (int) size of window surrounding each tick data
        gamma_multiplier: (float) granularity parameter for cleaning criteria
        source_file_type: (str) format to read from
        dest_file_type: (str) format to save to local directory
        """
        self.parent_path = parent_path
        self.k_period = k_period
        self.gamma_multiplier = gamma_multiplier
        self.quotes_trades_folders_lst = self.get_folders_to_traverse(self.parent_path)
        self.multiprocessing_args_lst = [(self.k_period, self.gamma_multiplier, source_file_type, dest_file_type, f)
                                         for f in self.quotes_trades_folders_lst]

    def clean(self, k_period, gamma_multiplier, source_file_type, dest_file_type, folder):
        """
        Cleaning the data using Bollinger bands and saves the clean data to local directory

        params
        k_period: (int) size of window surrounding each tick data
        gamma_multiplier: (float) granularity parameter for cleaning criteria
        source_file_type: (str) format to read from
        dest_file_type: (str) format to save to local directory
        folder: (str) full file path of folder destination with data to be cleaned

        returns:
        nothing
        """

        # get date directory files to travers and identify which type of record (quotes or trades)
        dates_to_traverse = self.get_folders_to_traverse(folder)
        quotes_or_trades = folder.rsplit("\\", 1)[-1]

        # traversing the date folders
        for date in dates_to_traverse:
            tickers_to_traverse = [f.path for f in os.scandir(date)
                                   if f.path.rsplit(".", 1)[-1] == source_file_type]

            # traversing the individual files
            for ticker in tickers_to_traverse:

                # load data
                df = pd.read_feather(ticker) if source_file_type == 'feather' else pd.read_csv(ticker)
                ticker_handle = ticker.rsplit("\\", 1)[-1].split("_")[0]

                # get list of outliers to drop
                indices_to_drop = self.get_outliers(df, k_period, quotes_or_trades, gamma_multiplier)

                # clean data
                df = df.drop(index=indices_to_drop)
                df = df.reset_index()

                # save the cleaned data
                self.save_dataframe(dest_file_type, df, date, ticker_handle, quotes_or_trades)

    def get_outliers(self, df, k_period, quotes_or_trades, gamma_multiplier):
        """
        Traverses tick data, applies Bollinger bands, and filters out identified outliers

        params:
        df: (Pandas df) dataframe to be cleaned
        quotes_or_trades: (str) type of tick data
        gamma_multiplier: (float) granularity parameter (multiplier for minimum variation price)

        returns:
        (list) list of outliers to drop
        """

        # validating k_period and setting up necessary data structures to affect Bollinger filter and store outliers
        indices_to_drop = np.array([], dtype=int)
        len_df = len(df)
        k_period = min(k_period, len_df - 1)
        bookend_length = k_period // 2

        for idx, row in df.iterrows():

            # get window edge indices
            start_idx, end_idx = self.get_start_end_indices(k_period, bookend_length, idx, len_df)

            # aggregating outliers by index based on tick record type
            if quotes_or_trades == 'quotes':
                window = ((df['Ask Price'][start_idx:end_idx + 1]
                           + df['Bid Price'][start_idx:end_idx + 1]) / 2)
                mask = ~window.index.isin([idx])
                price_list = window[mask]
                price_mean = stats.mean(price_list)
                price_std = stats.stdev(price_list)
                curr_price = (row['Ask Price'] + row['Bid Price']) / 2
            else:
                window = df['Price'][start_idx:end_idx + 1]
                mask = ~window.index.isin([idx])
                price_list = window[mask]
                price_mean = stats.mean(price_list)
                price_std = stats.stdev(price_list)
                curr_price = row['Price']

            # applying Bollinger band filter
            neigh_dist = abs(curr_price - price_mean)
            if neigh_dist >= 2 * price_std + gamma_multiplier * .01:
                indices_to_drop = np.append(indices_to_drop, idx)

        return indices_to_drop

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
    def get_start_end_indices(k_period, bookend_length, idx, len_df):
        """
        Dynamically calculates each side of window for Bollinger band filtering

        params
        k_period: (int) size of window surrounding each tick data
        bookend_length: (int) size of each side of window surrounding tick data
        idx: (int) index of tick being tested for filtering
        len_df: length of the dataframe to adjust window side lengths

        returns:
        (int) start index for Bollinger band window
        (int) end index for Bollinger band window
        """

        # determine position of idx
        outside_first_k_rows = (idx - bookend_length) > 0
        outside_last_k_rows = (idx + bookend_length) < len_df - 1

        # setting start and end indices for window for Bollinger bands
        if not outside_first_k_rows and outside_last_k_rows:
            start_idx = 0
            end_idx = bookend_length * 2
        elif outside_first_k_rows and outside_last_k_rows:
            start_idx = idx - bookend_length
            end_idx = idx + bookend_length
        elif outside_first_k_rows and not outside_last_k_rows:
            start_idx = len_df - k_period - 1
            end_idx = len_df - 1
        else:
            start_idx = 0
            end_idx = len_df - 1

        return start_idx, end_idx

    @staticmethod
    def save_dataframe(dest_file_type, df, date, ticker_handle, quotes_or_trades):
        """
        Saves dataframe to local file based on csv of feather designation

        params:
        dest_file_type: (str) intended file format for saving
        df: (Pandas df) cleaned data to be saved to local directory
        date: (str) date of cleaned file
        ticker_handle: (str) stock ticker handle
        quotes_or_trades: (str) type of tick record

        returns:
        nothing
        """
        if dest_file_type == 'csv':
            df.to_csv(os.path.join(date, ticker_handle + '_' + quotes_or_trades
                                   + '_cleaned_data.csv'), index=False)
        else:
            df.reset_index(drop=True, inplace=True)
            df.drop('index', axis=1, inplace=True)
            df.to_feather(os.path.join(date, ticker_handle + '_' + quotes_or_trades
                                       + '_cleaned_data.feather'))


if __name__ == '__main__':
    # NOTE: PyCharm has a bug that required the multiprocessing functionality to be placed in the main block
    cleaner = TAQCleaner(r'C:\Users\Dennis Kim\Desktop\AQTS Redux\AQTS\Testing', gamma_multiplier=20)
    p = mp.Pool(mp.cpu_count())
    p.starmap(cleaner.clean, cleaner.multiprocessing_args_lst)
    p.close()
