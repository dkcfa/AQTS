# Author: Dennis Kim

import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from AQTS.Utils.TAQQuotesReader import TAQQuotesReader
from AQTS.Utils.TAQTradesReader import TAQTradesReader


class TAQAdjust(object):
    """
    This class will traverse every binary trade and quote file, make corporate action adjustments, and save
    to the appropriate date folder as a feather file.

    Note: the logic structure here depends on a specific folder/subfolder structure. Future users need
    to adjust source and destination paths for their specific setup. This is also a long process.
    """

    def __init__(self, parent_path, file_save_type='feather'):
        """
        Initializer will do all the calling. User simply calls the class.

        params:
        parent_path: (str) directory with adjustment excel and quotes/trades files
        file_save_type: (str) the format of the save file
        """

        self.parent_path = parent_path
        self.records_to_adjust = self.get_records_to_adjust()
        self.adjustment_data = self.get_adjustment_data()

        # create list of lists for multiprocessing
        self.multiprocessing_args_lst = [(file_save_type, f) for f in self.records_to_adjust]

    def get_records_to_adjust(self):
        """
        Retrieves paths for quotes and trades folders to traverse

        returns:
        list of quotes and trades path
        """
        return [f.path for f in os.scandir(self.parent_path) if f.is_dir()]

    def get_adjustment_data(self, reference_file_name=r's&p500.xlsx'):
        """
        Retrieves that corporate adjustment data from the file that should be in the parent directory

        params:
        reference_file_name: (str) file with corporate adjustments

        returns:
        dataframe of adjustment data for all tickers and dates
        """

        # retrieving the adjustment data
        file_path = os.path.join(self.parent_path, reference_file_name)
        cols_for_adjustment = ['Names Date', 'Ticker Symbol', 'Cumulative Factor to Adjust Prices',
                               'Cumulative Factor to Adjust Shares/Vol']
        df = pd.read_excel(file_path, usecols=cols_for_adjustment)

        # casting columns to appropriate dtype
        df.loc[:, 'Names Date'] = df.loc[:, 'Names Date'].astype(str)
        df.loc[:, 'Ticker Symbol'] = df.loc[:, 'Ticker Symbol'].astype(str)
        df.loc[:, 'Cumulative Factor to Adjust Prices'] = df.loc[:, 'Cumulative Factor to Adjust Prices'].astype(float)
        df.loc[:, 'Cumulative Factor to Adjust Shares/Vol'] = \
            df.loc[:, 'Cumulative Factor to Adjust Shares/Vol'].astype(float)

        return df

    def adjust_and_save(self, file_type, folder):
        """
        Traverses through every binary file for quotes and trades, makes adjustments, and saves data in format
            designated

        params:
        file_type: (str) desired format for saved file

        returns:
        nothing
        """

        dates_folders = [f.path for f in os.scandir(folder) if f.is_dir()]
        reader_type = 'TAQQuotesReader' if folder.split('\\')[-1] == 'quotes' else 'TAQTradesReader'
        source_file_type = 'binRQ' if folder.split('\\')[-1] == 'quotes' else 'binRT'

        # traverse every date folder
        for date_path in dates_folders:
            tickers = [f.path for f in os.scandir(date_path) if f.path.rsplit(".", 1)[-1] == source_file_type]
            date_string = date_path.split("\\")[-1]

            # traverse every ticker for each date folder
            for ticker in tickers:

                # initialize appropriate reader and ticker
                reader = self.get_reader(ticker, reader_type)
                ticker_handle = ticker.rsplit("\\", 1)[-1].split("_")[0]

                # retrieve price and vol adjustment figures
                prices_lst, vol_lst, len_prices_list, len_vol_list = \
                    self.get_ticker_date_adjustment_data(ticker_handle, date_string)

                # validate adjustment data exists
                if len_prices_list == 0 or len_vol_list == 0:
                    continue
                else:
                    price_adj = prices_lst[0]
                    vol_adj = vol_lst[0]

                # adjust data
                curr_df, quotes_or_trades = self.adjusted_data(reader, reader_type, price_adj, vol_adj)

                # save data
                self.save_dataframe(file_type, curr_df, date_path, ticker_handle, quotes_or_trades)

    def get_ticker_date_adjustment_data(self, ticker_handle, date_string):
        """
        Retrieves the correct adjustment data for the ticker and date in question

        params:
        ticker_handle: ticker
        date_string: date of corporate action adjustment
        return:
        dataframes for both prices and volume adjustments specific to the specified ticker passed
        """

        # filter the full adjustment data to the ticker and date needed
        ticker_table = self.adjustment_data.loc[(self.adjustment_data['Ticker Symbol'] == ticker_handle)]
        ticker_date_table = ticker_table.loc[(ticker_table['Names Date'] == date_string)]

        # extract the appropriate column
        prices_table = ticker_date_table['Cumulative Factor to Adjust Prices']
        vol_table = ticker_date_table['Cumulative Factor to Adjust Shares/Vol']

        # transform pd.series to list type
        prices_lst = list(prices_table)
        vol_lst = list(vol_table)

        # get total element count for future validation
        len_prices_list = len(prices_lst)
        len_vol_list = len(vol_lst)

        return prices_lst, vol_lst, len_prices_list, len_vol_list

    @staticmethod
    def adjusted_data(reader, reader_type, price_adj, vol_adj):
        """
        Retrieve data from TAQQuotesReader/TAQTradesReader, apply corporate action adjustments, return the adjusted
            data and quote/trade type

        params:
        reader: TAQQuotesReader TAQTradesReader
        reader_type: (str) quotes or trades
        price_adj: (float) the price adjustment factor
        vol_adj: (float) vol adjustment factor

        returns:
        dataframe of adjusted data and if quote/trade data
        """

        # adjust quote side data
        if reader_type == "TAQQuotesReader":
            ct = reader.getN() - 1

            orig_ask_size_list = []
            orig_ask_price_list = []
            orig_bid_size_list = []
            orig_bid_price_list = []
            millis_from_epoc = []

            # read in every available record
            i = 0
            while i <= ct:
                ask_size = reader.getAskSize(i)
                ask_price = reader.getAskPrice(i)
                bid_size = reader.getBidSize(i)
                bid_price = reader.getBidPrice(i)
                millis_from_epoc.append(reader.getSecsFromEpocToMidn() + reader.getMillisFromMidn(i))
                orig_ask_size_list.append(ask_size)
                orig_ask_price_list.append(ask_price)
                orig_bid_size_list.append(bid_size)
                orig_bid_price_list.append(bid_price)

                i += 1

            # make corporate action adjustments and save to a dataframe
            ask_size_list = np.array(orig_ask_size_list) * vol_adj
            ask_price_list = np.array(orig_ask_price_list) * price_adj
            bid_size_list = np.array(orig_bid_size_list) * vol_adj
            bid_price_list = np.array(orig_bid_price_list) * price_adj
            curr_df = pd.DataFrame({"millis from epoc": millis_from_epoc,
                                    "Orig Ask Size": orig_ask_size_list,
                                    "Orig Ask Price": orig_ask_price_list,
                                    "Orig Bid Size": orig_bid_size_list,
                                    "Orig Bid Price": orig_bid_price_list,
                                    "Ask Size": ask_size_list, "Ask Price": ask_price_list,
                                    "Bid Size": bid_size_list, "Bid Price": bid_price_list})

        else:
            ct = reader.getN() - 1

            orig_size_list = []
            orig_price_list = []
            millis_from_epoc = []

            i = 0
            while i <= ct:
                millis_from_epoc.append(reader.getSecsFromEpocToMidn() + reader.getMillisFromMidn(i))
                orig_size_list.append(reader.getSize(i))
                orig_price_list.append(reader.getPrice(i))
                i += 1

            size_list = np.array(orig_size_list) * vol_adj
            price_list = np.array(orig_price_list) * price_adj
            curr_df = pd.DataFrame({"millis from epoc": millis_from_epoc,
                                    "Orig Size": orig_size_list, "Orig Price": orig_price_list,
                                    "Size": size_list, "Price": price_list})

        # indicate which type of data for the returned adjusted dataframe
        if reader_type == "TAQQuotesReader":
            quotes_or_trades = "quotes"
        else:
            quotes_or_trades = "trades"

        return curr_df, quotes_or_trades

    @staticmethod
    def get_reader(ticker, reader_type):
        """
        Initializes TAQQuotesReader or TAQTradesReader depending on the reader_type passed

        params
        ticker: (str) stock ticker
        reader_type: (str) quotes or trades reader

        returns:
        TAQQuotesReader or TAQTradesReader
        """

        # returns appropriate reader type based on ticker type
        if reader_type == 'TAQQuotesReader':
            return TAQQuotesReader(ticker)
        else:
            return TAQTradesReader(ticker)

    @staticmethod
    def save_dataframe(dest_file_type, df, date, ticker_handle, quotes_or_trades):
        """
        Saves a dataframe to local directory

        params:
        dest_file_type: (str) file type for saving
        df: (Pandas df) adjusted data to be saved to local directory
        date: (str) date of records
        ticker_handle: (str) stock ticker
        quotes_or_trades: (str) indicator for type of record

        returns:
        nothing
        """

        # saves as the indicated file type
        if dest_file_type == 'csv':
            df.to_csv(os.path.join(date, ticker_handle + '_' + quotes_or_trades
                                   + '_adjusted_data.csv'), index=False)
        else:
            df.reset_index(drop=True, inplace=True)
            df.to_feather(os.path.join(date, ticker_handle + '_' + quotes_or_trades
                                       + '_adjusted_data.feather'))


if __name__ == '__main__':
    # NOTE: PyCharm has a bug that required the multiprocessing functionality to be placed in the main block
    adjust = TAQAdjust(r'C:\Users\Dennis Kim\Desktop\AQTS Redux\AQTS\Testing', 'csv')
    p = mp.Pool(mp.cpu_count())
    p.starmap(adjust.adjust_and_save, adjust.multiprocessing_args_lst)
    p.close()
