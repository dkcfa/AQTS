import os
import tarfile
import pandas as pd


class TAQExtraction(object):
    """
    This script will go through both quotes and trades folders, extract the single file in each tar.gz file,
    extract daily trade/quote records, do some other necessary logistics, and save files locally.

    Note: the logic structure here depends on a specific folder/subfolder structure. Future users need
    to adjust source and destination paths for their specific setup. This is also a long process.
    """

    def __init__(self, parent_directory=r'C:\Users\Dennis Kim\Desktop\AQTS Redux\AQTS\Data',
                 filter_file_name='s&p500.xlsx'):
        """
        Decompresses quotes and trades data from tar.gz files into binRT and binQT formats

        params:
        parent_directory: (str) source folder housing tar.gz files
        filter_file_name: (str) name of xlsx file containing
        """
        self.parent_directory = parent_directory
        self.dirs_with_trade_quote_sources = self.fetch_trade_quote_folder_sources()
        self.to_filter_list = self.get_filter_list(filter_file_name)
        self.extract_daily_folder_data()
        self.filter_tickers()

    def fetch_trade_quote_folder_sources(self):
        """
        Creates a list of folders to traverse for tar.gz files

        returns:
        (list) list of folders housing tar.gz files
        """

        dirs_with_trade_quote_sources = []

        for file in os.listdir(self.parent_directory):
            item_path = os.path.join(self.parent_directory, file)

            if os.path.isdir(item_path):
                dirs_with_trade_quote_sources.append(item_path)

        return dirs_with_trade_quote_sources

    def extract_daily_folder_data(self):
        """
        This does not change the class. It only extracts the contents of a tar.gz file to the directory
        of the original tar.gz file

        returns:
        nothing
        """

        # traversing quotes and trades folders
        for directory in self.dirs_with_trade_quote_sources:

            # traverses all tar.gz files
            for file in os.listdir(directory):

                # extract the file to the created destination folder
                file_path = os.path.join(directory, file)
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(directory)

    def get_filter_list(self, filter_file='s&p500.xlsx'):
        """
        Retrieves a list of unique tickers belonging to the s&p500

        params:
        filter_file: (str) file name housing the stocks used to filter the available data
        returns:
        (list) list of unique tickers
        """

        file_path = os.path.join(self.parent_directory, filter_file)
        df = pd.read_excel(file_path)
        unique_tickers = list(set(df['Ticker Symbol'].tolist()))

        return unique_tickers

    def filter_tickers(self):
        """
        This doesn't change the class. It simply filters out non-sp500 stocks.

        returns:
        nothing
        """

        # iterating through quotes and trade folders
        for directory in self.dirs_with_trade_quote_sources:

            # list of daily sub-folders in each directory
            subdirectory = [f.path for f in os.scandir(directory) if f.is_dir()]

            # iterating through each file_path in subdirectory list
            for sub in subdirectory:
                file_paths = [f.path for f in os.scandir(sub)]

                # iterating through each record, filtering out files that are not of interest
                for file_path in file_paths:

                    file_name_split = file_path.split("\\")[-1]

                    co_tick = file_name_split.rsplit('_', 1)[0]

                    if co_tick not in self.to_filter_list:
                        os.remove(file_path)







