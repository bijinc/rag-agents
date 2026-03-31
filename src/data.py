import kagglehub
import pandas as pd
import os
import edgar

edgar.set_identity("user@example.com")

##############################################################################
#                  Earning Call Transcripts Data Retrieval                   #
##############################################################################

def download_ect_dataset():
    # Download latest version
    path = kagglehub.dataset_download("ramssvimala/earning-call-transcripts")
    print("Path to dataset files:", path)
    return path


def load_ect_data(path, name):
    # get folder path for the given company name
    folder_path = os.path.join(path, "cleaned_ECTs_dataset", name)

    # load all files in the folder into a pandas DataFrame
    df = pd.DataFrame([
        {"content": open(os.path.join(folder_path, file), "r").read()}
        for file in os.listdir(folder_path)
    ])

    return df

##############################################################################
#                        SEC Data Retrieval                                  #
##############################################################################

def get_sec_balance_sheet(ticker):
    company = edgar.Company(ticker)
    return company.get_financials().balance_sheet()


def get_sec_filings(ticker):
    company = edgar.Company(ticker)
    return company.get_filings()


if __name__ == "__main__":
    # path = download_ect_dataset()
    # df = load_ect_data(path, "Apple")
    # print(df.head())

    ticker = "AAPL"
    balance_sheet = get_sec_balance_sheet(ticker)
    print(balance_sheet)
    filings = get_sec_filings(ticker)
    print(filings)
