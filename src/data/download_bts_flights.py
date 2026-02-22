"""
BTS On-Time Performance Data -> JFK Daily Scheduled Flights
=============================================================

STEP 1: MANUAL DOWNLOAD (must be done by you - BTS requires browser)
STEP 2: RUN THIS SCRIPT to process downloaded files

=== DOWNLOAD INSTRUCTIONS ===

The BTS website requires JavaScript and does not support direct API downloads.
You must download month-by-month through the web interface.

1. Go to: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr

2. Under "Filter Geography": select ORIGIN = JFK or leave blank (we filter later)

3. Under "Filter Year/Period": select one year + one month at a time
   (e.g., Year=2019, Period=January)

4. Check ONLY these fields (uncheck everything else to keep file small):
   [x] FlightDate           (FL_DATE)
   [x] Marketing_Airline_Network  (or UNIQUE_CARRIER)
   [x] Origin               (ORIGIN)
   [x] Dest                 (DEST)
   [x] CRSDepTime           (scheduled departure time)
   [x] CRSArrTime           (scheduled arrival time)
   [x] Cancelled            (cancellation flag)
   [x] Diverted             (diversion flag)

5. Click "Download" -> saves as a .zip containing .csv

6. Repeat for each month from January 2019 to the latest available
   (BTS data lags ~2-3 months, so probably through Oct 2025)

   TIP: You can speed this up by downloading ALL airports and filtering
   to JFK in this script. The files are bigger but you save many clicks.

7. Unzip all files and put ALL the CSVs into one folder:
   data/raw/bts_flights/

=== ALTERNATIVE: FASTER BULK DOWNLOAD ===

Option A - Kaggle (pre-packaged, but may not have latest months):
  Search "airline on-time performance" on kaggle.com

Option B - Annual pre-zipped files:
  https://transtats.bts.gov/PREZIP/
  Files named: On_Time_Reporting_Carrier_On_Time_Performance_1987_present_YYYY_M.zip
  
  Example direct URL pattern:
  https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip
  
  You can try downloading these directly with a browser or download manager.
  Months: 1-12, Years: 2019-2025

=== AFTER DOWNLOADING, RUN THIS SCRIPT ===

Usage:
    python download_bts_flights.py

Input:  data/raw/bts_flights/*.csv
Output: data/processed/jfk_daily_scheduled_flights.csv
"""

import pandas as pd
import numpy as np
import glob
import os
import zipfile


def unzip_all(folder='data/raw/bts_flights'):
    """Unzip any .zip files in the folder"""
    zip_files = glob.glob(f'{folder}/*.zip')
    if zip_files:
        print(f"📦 Found {len(zip_files)} zip files, extracting...")
        for zf in zip_files:
            try:
                with zipfile.ZipFile(zf, 'r') as z:
                    z.extractall(folder)
                print(f"  ✅ {os.path.basename(zf)}")
            except Exception as e:
                print(f"  ❌ {os.path.basename(zf)}: {e}")


def load_and_filter_jfk(folder='data/raw/bts_flights'):
    """
    Load all BTS CSV files and filter to JFK flights.
    
    We keep flights where JFK is either the origin OR destination,
    because both departing and arriving passengers go through TSA
    (departing at JFK, arriving at origin airport).
    
    For our purposes, we focus on:
    - scheduled_departures: flights DEPARTING from JFK
    - scheduled_arrivals: flights ARRIVING at JFK
    """
    csv_files = sorted(glob.glob(f'{folder}/*.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in {folder}/")
        print(f"   Please download BTS data first (see instructions at top of script)")
        return None
    
    print(f"📂 Found {len(csv_files)} CSV files")
    
    all_data = []
    for i, f in enumerate(csv_files):
        try:
            # BTS CSVs sometimes have trailing comma -> extra unnamed column
            df = pd.read_csv(f, low_memory=False)
            
            # Drop unnamed columns (artifact of trailing comma)
            df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
            
            # Standardize column names (BTS changes case sometimes)
            df.columns = [c.strip().upper() for c in df.columns]
            
            # Find the date column (different BTS versions use different names)
            date_col = None
            for candidate in ['FL_DATE', 'FLIGHTDATE', 'FLIGHT_DATE']:
                if candidate in df.columns:
                    date_col = candidate
                    break
            
            if date_col is None:
                print(f"  ⚠️ Skipping {os.path.basename(f)}: no date column found")
                continue
            
            # Find origin/dest columns
            origin_col = 'ORIGIN' if 'ORIGIN' in df.columns else None
            dest_col = 'DEST' if 'DEST' in df.columns else None
            
            if origin_col is None or dest_col is None:
                print(f"  ⚠️ Skipping {os.path.basename(f)}: no ORIGIN/DEST columns")
                continue
            
            # Filter to JFK (origin OR destination)
            jfk_mask = (df[origin_col].str.strip() == 'JFK') | (df[dest_col].str.strip() == 'JFK')
            jfk_df = df[jfk_mask].copy()
            
            if len(jfk_df) > 0:
                all_data.append(jfk_df)
            
            if (i + 1) % 10 == 0 or i == len(csv_files) - 1:
                print(f"  Processed {i+1}/{len(csv_files)} files, "
                      f"JFK rows so far: {sum(len(d) for d in all_data):,}")
                
        except Exception as e:
            print(f"  ❌ Error reading {os.path.basename(f)}: {e}")
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Total JFK flight records: {len(combined):,}")
    return combined


def aggregate_to_daily(df):
    """
    Aggregate individual flight records to daily counts.
    
    Output columns:
    - date
    - scheduled_departures: flights scheduled to DEPART from JFK
    - scheduled_arrivals: flights scheduled to ARRIVE at JFK
    - total_scheduled_flights: departures + arrivals
    - cancelled_flights: flights that were cancelled
    - cancellation_rate: cancelled / total
    
    NOTE: For forecasting, use scheduled_departures and scheduled_arrivals
    as features (known in advance from published schedules).
    Do NOT use cancelled_flights as a feature — that's only known after the fact.
    """
    # Find date column
    date_col = None
    for candidate in ['FL_DATE', 'FLIGHTDATE', 'FLIGHT_DATE']:
        if candidate in df.columns:
            date_col = candidate
            break
    
    df['date'] = pd.to_datetime(df[date_col])
    
    # Find cancelled column
    cancelled_col = None
    for candidate in ['CANCELLED', 'CANCELED']:
        if candidate in df.columns:
            cancelled_col = candidate
            break
    
    # Classify departures vs arrivals
    df['is_departure'] = (df['ORIGIN'].str.strip() == 'JFK').astype(int)
    df['is_arrival'] = (df['DEST'].str.strip() == 'JFK').astype(int)
    
    if cancelled_col:
        df['is_cancelled'] = pd.to_numeric(df[cancelled_col], errors='coerce').fillna(0)
    else:
        df['is_cancelled'] = 0
    
    # Daily aggregation
    daily = df.groupby('date').agg(
        scheduled_departures=('is_departure', 'sum'),
        scheduled_arrivals=('is_arrival', 'sum'),
        cancelled_flights=('is_cancelled', 'sum'),
    ).reset_index()
    
    daily['total_scheduled_flights'] = daily['scheduled_departures'] + daily['scheduled_arrivals']
    daily['cancellation_rate'] = (daily['cancelled_flights'] / 
                                  daily['total_scheduled_flights']).round(4)
    
    # Also get unique carrier count per day
    if 'UNIQUE_CARRIER' in df.columns or 'MARKETING_AIRLINE_NETWORK' in df.columns:
        carrier_col = 'UNIQUE_CARRIER' if 'UNIQUE_CARRIER' in df.columns else 'MARKETING_AIRLINE_NETWORK'
        carriers = df.groupby('date')[carrier_col].nunique().reset_index()
        carriers.columns = ['date', 'num_carriers']
        daily = daily.merge(carriers, on='date', how='left')
    
    daily = daily.sort_values('date').reset_index(drop=True)
    
    return daily


def main():
    print("=" * 70)
    print("✈️  BTS On-Time Performance -> JFK Daily Scheduled Flights")
    print("=" * 70)
    
    folder = 'data/raw/bts_flights'
    
    # Check if folder exists
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"\n📁 Created folder: {folder}/")
        print(f"   Please download BTS data into this folder first.")
        print(f"   See download instructions at the top of this script.")
        print(f"\n   Quick start:")
        print(f"   1. Go to https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr")
        print(f"   2. Select fields: FlightDate, Origin, Dest, CRSDepTime, CRSArrTime, Cancelled")
        print(f"   3. Download month by month (2019-01 to latest)")
        print(f"   4. Put all CSV/ZIP files in {folder}/")
        print(f"   5. Re-run this script")
        return
    
    # Unzip any zip files
    unzip_all(folder)
    
    # Load and filter
    jfk_flights = load_and_filter_jfk(folder)
    
    if jfk_flights is None:
        return
    
    # Aggregate
    print("\n📊 Aggregating to daily counts...")
    daily = aggregate_to_daily(jfk_flights)
    
    print(f"\n{'=' * 70}")
    print(f"📊 JFK Daily Scheduled Flights Summary")
    print(f"{'=' * 70}")
    print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"Total days: {len(daily)}")
    print(f"\nScheduled departures per day:")
    print(f"  Mean: {daily['scheduled_departures'].mean():.0f}")
    print(f"  Min:  {daily['scheduled_departures'].min():.0f} ({daily.loc[daily['scheduled_departures'].idxmin(), 'date'].date()})")
    print(f"  Max:  {daily['scheduled_departures'].max():.0f} ({daily.loc[daily['scheduled_departures'].idxmax(), 'date'].date()})")
    print(f"\nScheduled arrivals per day:")
    print(f"  Mean: {daily['scheduled_arrivals'].mean():.0f}")
    print(f"\nTotal scheduled flights per day:")
    print(f"  Mean: {daily['total_scheduled_flights'].mean():.0f}")
    print(f"\nAvg cancellation rate: {daily['cancellation_rate'].mean()*100:.1f}%")
    
    if 'num_carriers' in daily.columns:
        print(f"Avg carriers per day: {daily['num_carriers'].mean():.0f}")
    
    # Save
    os.makedirs('data/processed', exist_ok=True)
    daily.to_csv('data/processed/jfk_daily_scheduled_flights.csv', index=False)
    print(f"\n✅ Saved: data/processed/jfk_daily_scheduled_flights.csv")
    
    # Also save a preview
    print(f"\n--- First 10 rows ---")
    print(daily.head(10).to_string(index=False))
    
    # ================================================================
    # MERGE WITH EXISTING PROCESSED DATA (if available)
    # ================================================================
    merged_path = 'data/processed/jfk_daily_merged.csv'
    if os.path.exists(merged_path):
        print(f"\n{'=' * 70}")
        print(f"🔗 Merging with existing dataset: {merged_path}")
        print(f"{'=' * 70}")
        
        existing = pd.read_csv(merged_path)
        existing['Date'] = pd.to_datetime(existing['Date'])
        
        # Merge on date
        flights_for_merge = daily[['date', 'scheduled_departures', 'scheduled_arrivals',
                                    'total_scheduled_flights']].copy()
        flights_for_merge = flights_for_merge.rename(columns={'date': 'Date'})
        
        merged = existing.merge(flights_for_merge, on='Date', how='left')
        
        n_matched = merged['scheduled_departures'].notna().sum()
        n_missing = merged['scheduled_departures'].isna().sum()
        print(f"  Matched: {n_matched} days")
        print(f"  Missing flight data: {n_missing} days")
        
        # Save updated
        merged.to_csv('data/processed/jfk_daily_merged_with_flights.csv', index=False)
        print(f"  ✅ Saved: data/processed/jfk_daily_merged_with_flights.csv")
    
    return daily


if __name__ == '__main__':
    main()
