import kagglehub
import pandas as pd
import os
import json
import edgar
from datetime import datetime
from pathlib import Path

edgar.set_identity("user@example.com")

##############################################################################
#                        EARNINGS CALL TRANSCRIPTS DATA                      #
##############################################################################

def download_ect_dataset():
    """Download latest version of earnings call transcripts from Kaggle"""
    path = kagglehub.dataset_download("ramssvimala/earning-call-transcripts")
    print("Path to dataset files:", path)
    return path


def load_ect_data(path, name):
    """Load earnings call transcripts for a given company name"""
    folder_path = os.path.join(path, "cleaned_ECTs_dataset", name)
    df = pd.DataFrame([
        {"content": open(os.path.join(folder_path, file), "r").read()}
        for file in os.listdir(folder_path)
    ])
    return df


##############################################################################
#                    SEC FILINGS DATA COLLECTION                             #
##############################################################################

COMPANY_INFO = {
    "AAPL": "Apple Inc.",
    "AMD": "Advanced Micro Devices"
}

FILING_TYPES = {
    "10-K": 2,   # Annual reports
    "10-Q": 8,   # Quarterly reports (2 years = 8 quarters)
    "8-K": 5     # Current reports (material events)
}


def setup_directories(tickers):
    """Create directory structure for storing SEC filings"""
    for ticker in tickers:
        for filing_type in FILING_TYPES.keys():
            dir_path = Path(f"data/raw/sec_filings/{ticker}/{filing_type}")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")


def fetch_filings_by_type(company, ticker, filing_type, n):
    """
    Fetch N filings of a specific type (10-K, 10-Q, or 8-K).

    Returns:
        List of dicts with keys: ticker, company_name, filing_type, filing_date, content
    """
    filings_list = []

    try:
        print(f"  Fetching {filing_type} filings for {ticker}...")

        # Get all filings and filter by type
        all_filings = company.get_filings()
        matching_filings = [f for f in all_filings if f.form == filing_type][:n]

        # Extract text and metadata from each filing
        for filing in matching_filings:
            try:
                # Get filing text
                text = filing.text()

                # Create filing data dict
                filing_data = {
                    "ticker": ticker,
                    "company_name": COMPANY_INFO.get(ticker, ticker),
                    "filing_type": filing.form,
                    "filing_date": str(filing.filing_date),
                    "accession_number": filing.accession_no,
                    "content": text
                }
                filings_list.append(filing_data)

            except Exception as e:
                print(f"      ✗ Error extracting text from {filing_type} filing: {e}")
                continue

        if len(filings_list) > 0:
            print(f"    ✓ Fetched {len(filings_list)} {filing_type} filings")
        else:
            print(f"    ✗ No {filing_type} filings found")

    except Exception as e:
        print(f"    ✗ Error fetching {filing_type} filings: {e}")

    return filings_list


def save_filing(filing_data, ticker, filing_type):
    """Save a single filing as JSON file"""
    try:
        # Create filename from filing type and date
        date_str = filing_data["filing_date"].replace("-", "")
        filename = f"{filing_type}_{date_str}.json"
        filepath = Path(f"data/raw/sec_filings/{ticker}/{filing_type}/{filename}")

        # Write JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(filing_data, f, indent=2, ensure_ascii=False)

        print(f"    Saved: {filepath}")
        return True

    except Exception as e:
        print(f"    ✗ Error saving filing: {e}")
        return False


def download_sec_data(tickers):
    """
    Main orchestrator: Download and save SEC filings for given tickers.
    """
    print("\n" + "="*70)
    print("SEC FILINGS DATA COLLECTION")
    print("="*70)

    # Track statistics for manifest
    stats = {}

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")

        try:
            # Create Company object
            company = edgar.Company(ticker)
            stats[ticker] = {
                "company_name": COMPANY_INFO.get(ticker, ticker),
                "filings": {}
            }

            # Fetch each filing type
            for filing_type, count in FILING_TYPES.items():
                print(f"  {filing_type}:")

                # Fetch filings
                filings = fetch_filings_by_type(company, ticker, filing_type, count)
                stats[ticker]["filings"][filing_type] = len(filings)

                # Save each filing
                for filing in filings:
                    save_filing(filing, ticker, filing_type)

        except Exception as e:
            print(f"  ✗ Error processing {ticker}: {e}")
            stats[ticker]["filings"] = {"error": str(e)}

    # Save manifest
    save_manifest(stats)
    print("\n" + "="*70)
    print("SEC DATA COLLECTION COMPLETE")
    print("="*70 + "\n")


def save_manifest(stats):
    """
    Save manifest.json with metadata about downloaded filings.
    """
    try:
        # Calculate totals
        total_filings = 0
        for ticker_data in stats.values():
            if "filings" in ticker_data and isinstance(ticker_data["filings"], dict):
                total_filings += sum(
                    v for k, v in ticker_data["filings"].items() if k != "error"
                )

        manifest = {
            "last_updated": datetime.now().isoformat(),
            "companies": stats,
            "total_documents": total_filings
        }

        manifest_path = Path("data/manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"✓ Manifest saved to {manifest_path}")

    except Exception as e:
        print(f"✗ Error saving manifest: {e}")


##############################################################################
#                        EXISTING SEC FUNCTIONS                              #
##############################################################################

def get_sec_balance_sheet(ticker):
    """Get balance sheet for a company"""
    company = edgar.Company(ticker)
    return company.get_financials().balance_sheet()


def get_sec_filings(ticker):
    """Get all filings for a company"""
    company = edgar.Company(ticker)
    return company.get_filings()


if __name__ == "__main__":
    # Set up directories
    TICKERS = ["AAPL", "AMD"]
    setup_directories(TICKERS)

    # Download SEC filings
    download_sec_data(TICKERS)
