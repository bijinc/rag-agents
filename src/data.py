from datetime import datetime
from pathlib import Path
import pandas as pd
import kagglehub
import edgar
import json
import time
import os
import re

edgar.set_identity("user@example.com")

# Single source of truth for company configuration
COMPANIES = {
    "AAPL": {
        "display_name": "Apple Inc.",
        "ect_dataset_name": "Apple"  # Name as it appears in Kaggle dataset
    },
    "AMD": {
        "display_name": "Advanced Micro Devices",
        "ect_dataset_name": "AMD"
    },
    "COST": {
        "display_name": "Costco Wholesale Corporation",
        "ect_dataset_name": "Costco"
    },
    "JPM": {
        "display_name": "JPMorgan Chase & Co.",
        "ect_dataset_name": "JPM"
    },
    "F": {
        "display_name": "Ford Motor Company",
        "ect_dataset_name": "Ford"
    },
    "ELV": {
        "display_name": "Elevance Health, Inc.",
        "ect_dataset_name": "Elevance Health"
    }
}

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


def extract_ect_metadata(filename):
    """
    Extract year and quarter from ECT filename.
    Format: YYYY_QX_ticker_processed.txt
    Returns: (year, quarter) or (None, None) if parsing fails
    """
    try:
        match = re.match(r'(\d{4})_Q(\d)_', filename)
        if match:
            year, quarter = match.groups()
            return int(year), int(quarter)
    except:
        pass
    return None, None


def fetch_ect_for_companies():
    """
    Fetch earnings call transcripts for specified companies.

    Args:
        dataset_path: Path to the downloaded Kaggle dataset
        tickers: List of tickers (e.g., ["AAPL", "AMD"])

    Returns:
        Dict mapping ticker to list of ECT dicts
    """

    print("\n" + "="*70)
    print("EARNINGS CALL TRANSCRIPTS DATA COLLECTION")
    print("="*70 + "\n")

    ect_data = {}
    dataset_path = download_ect_dataset()
    dataset_root = os.path.join(dataset_path, "cleaned_ECTs_dataset")

    for ticker in COMPANIES.keys():
        display_name = COMPANIES[ticker]["display_name"]
        dataset_name = COMPANIES[ticker]["ect_dataset_name"]
        company_path = os.path.join(dataset_root, dataset_name)

        if not os.path.exists(company_path):
            print(f"  Company folder not found: {dataset_name}")
            continue

        print(f"  Loading ECT for {ticker}...")
        ect_files = sorted(os.listdir(company_path))
        transcripts = []

        for filename in ect_files:
            if not filename.endswith('.txt'):
                continue

            filepath = os.path.join(company_path, filename)
            try:
                # Extract metadata from filename
                year, quarter = extract_ect_metadata(filename)

                # Read file content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Create ECT data dict
                ect_entry = {
                    "ticker": ticker,
                    "company_name": display_name,
                    "year": year,
                    "quarter": quarter,
                    "period": f"{year}_Q{quarter}" if year and quarter else filename.replace('_processed.txt', ''),
                    "content": content
                }
                transcripts.append(ect_entry)

            except Exception as e:
                print(f"    Error reading {filename}: {e}")
                continue

        if transcripts:
            ect_data[ticker] = transcripts
            print(f"    Fetched {len(transcripts)} ECT records for {ticker}")
        else:
            print(f"    No ECT records found for {ticker}")

    return ect_data


def save_ect_filings(ect_data):
    """
    Save ECT data as JSON files.
    Directory structure: data/raw/ect/{ticker}/{period}.json
    """
    for ticker, transcripts in ect_data.items():
        if not transcripts:
            continue

        # Create directory using ticker for consistency
        ect_dir = Path(f"data/raw/ect/{ticker}")
        ect_dir.mkdir(parents=True, exist_ok=True)

        # Save each transcript
        for transcript in transcripts:
            filename = f"{transcript['period']}.json"
            filepath = ect_dir / filename

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(transcript, f, indent=2, ensure_ascii=False)
                print(f"    Saved: {filepath}")
            except Exception as e:
                print(f"    Error saving {filepath}: {e}")

##############################################################################
#                    SEC FILINGS DATA COLLECTION                             #
##############################################################################

FILING_TYPES = {
    "10-K": 2,   # Annual reports
    "10-Q": 8,   # Quarterly reports (2 years = 8 quarters)
    "8-K": 5     # Current reports (material events)
}


def setup_directories():
    """Create directory structure for storing SEC filings"""
    for ticker in COMPANIES.keys():
        for filing_type in FILING_TYPES.keys():
            dir_path = Path(f"data/raw/sec_filings/{ticker}/{filing_type}")
            dir_path.mkdir(parents=True, exist_ok=True)
            # print(f"Created directory: {dir_path}")


def fetch_filings_by_type(company, ticker, filing_type, n):
    """
    Fetch N filings of a specific type (10-K, 10-Q, or 8-K).
    Args:
        company: edgar.Company object
        ticker: Stock ticker (for logging)
        filing_type: Type of filing to fetch (e.g., "10-K")
        n: Number of filings to fetch

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
                    "company_name": COMPANIES[ticker]["display_name"],
                    "filing_type": filing.form,
                    "filing_date": str(filing.filing_date),
                    "accession_number": filing.accession_no,
                    "content": text
                }
                filings_list.append(filing_data)

            except Exception as e:
                print(f"      Error extracting text from {filing_type} filing: {e}")
                continue

        if len(filings_list) > 0:
            print(f"    Fetched {len(filings_list)} {filing_type} filings")
        else:
            print(f"    No {filing_type} filings found")

    except Exception as e:
        print(f"    Error fetching {filing_type} filings: {e}")

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
        print(f"    Error saving filing: {e}")
        return False


def download_sec_data():
    """
    Main orchestrator: Download and save SEC filings for given tickers.
    """
    print("\n" + "="*70)
    print("SEC FILINGS DATA COLLECTION")
    print("="*70)

    # Track statistics for manifest
    stats = {}

    for ticker in COMPANIES.keys():
        print(f"\nProcessing {ticker}...")

        try:
            # Create Company object
            company = edgar.Company(ticker)
            stats[ticker] = {
                "company_name": COMPANIES[ticker]["display_name"],
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
            print(f"  Error downloading SEC data for {ticker}: {e}")
            stats[ticker]["filings"] = {"error": str(e)}

    # Save manifest
    save_manifest(stats)
    print("\nSEC DATA COLLECTION COMPLETE")


def save_manifest(stats, ect_counts=None):
    """Save manifest.json with metadata about downloaded filings and ECT data."""
    try:
        # Calculate totals for SEC filings
        total_sec_filings = 0
        for ticker_data in stats.values():
            if "filings" in ticker_data and isinstance(ticker_data["filings"], dict):
                total_sec_filings += sum(
                    v for k, v in ticker_data["filings"].items() if k != "error"
                )

        # Calculate totals for ECT
        total_ect = sum(ect_counts.values()) if ect_counts else 0

        manifest = {
            "last_updated": datetime.now().isoformat(),
            "sec_filings": stats,
            "ect_data": ect_counts or {},
            "totals": {
                "sec_filings": total_sec_filings,
                "ect_transcripts": total_ect,
                "total_documents": total_sec_filings + total_ect
            }
        }

        manifest_path = Path("data/manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved to {manifest_path}")

    except Exception as e:
        print(f"Error saving manifest: {e}")

if __name__ == "__main__":

    start = time.perf_counter()

    print(f"Fetching data for tickers: {', '.join(COMPANIES.keys())}")
    
    setup_directories()
    download_sec_data()

    # Download ECT data
    ect_data = fetch_ect_for_companies()
    save_ect_filings(ect_data)

    # Count ECT records per ticker
    ect_counts = {ticker: len(ect_data.get(ticker, [])) for ticker in COMPANIES.keys()}

    # Update manifest with ECT data
    # Note: save_manifest is called in download_sec_data, need to update it there
    # For now, update manifest with ECT info
    try:
        manifest_path = Path("data/manifest.json")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        manifest["ect_data"] = ect_counts
        manifest["totals"] = {
            "sec_filings": manifest.get("totals", {}).get("sec_filings", 30),
            "ect_transcripts": sum(ect_counts.values()),
            "total_documents": manifest.get("totals", {}).get("sec_filings", 30) + sum(ect_counts.values())
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest updated with ECT data")
    except Exception as e:
        print(f"Error updating manifest: {e}")

    end = time.perf_counter()
    print(f"\nECT DATA COLLECTION COMPLETE (Time taken: {end - start:.2f} seconds)")
