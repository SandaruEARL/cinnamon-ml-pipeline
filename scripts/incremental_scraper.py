#!/usr/bin/env python3
"""
INCREMENTAL CINNAMON SCRAPER (WITH NATIONAL VALUES)
Scrapes ONLY last 4 weeks of data (for weekly automation)
Now includes national benchmark prices

"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
import re
import os

BASE_URL = "https://exagri.info/mkt/"
INDEX_URL = "https://exagri.info/mkt/"

def fetch_page(url, max_retries=3):
    """Fetch a page with retry logic"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 404:
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
    return None

def parse_date_from_text(text):
    """Parse date from link text like '05-January-2016' or '01.11.2016'"""
    try:
        for fmt in ['%d-%B-%Y', '%d-%b-%Y', '%d.%m.%Y']:
            try:
                return datetime.strptime(text, fmt)
            except:
                continue
        
        match = re.search(r'(\d{1,2})[-.](\w+)[-.](\d{4})', text)
        if match:
            day, month, year = match.groups()
            date_str = f"{day}-{month}-{year}"
            for fmt in ['%d-%B-%Y', '%d-%b-%Y']:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
    except:
        pass
    
    return None

def fetch_recent_urls_from_index(cutoff_date):
    """Fetch URLs from index page for dates >= cutoff_date"""
    print(f"🔍 Fetching index page for dates >= {cutoff_date.strftime('%Y-%m-%d')}...")
    
    html = fetch_page(INDEX_URL)
    if not html:
        print(f"  ❌ Failed to fetch index page")
        return {}
    
    soup = BeautifulSoup(html, 'html.parser')
    
    url_map = {}
    
    current_year = datetime.now().year  # ← DEFINE ONCE HERE
    years_to_check = [current_year, current_year - 1]
    
    for year in years_to_check:
        year_header = None
        for center in soup.find_all('center'):
            h1 = center.find('h1')
            if h1 and str(year) in h1.get_text():
                year_header = center
                break
        
        if not year_header:
            continue
        
        current = year_header
        
        while True:
            current = current.find_next()
            
            if not current:
                break
            
            # Add dynamic year boundary detection
            if current.name == 'center':
               h1 = current.find('h1')
               if h1:
                 # REMOVE: current_year = datetime.now().year  
                 next_year_str = str(year + 1) if year <= current_year else str(year - 1)
                 if next_year_str in h1.get_text():
                    break
            
            if current.name == 'a':
                href = current.get('href', '')
                
                if href.startswith(f"{year}/"):
                    link_text = current.get_text(strip=True)
                    date_obj = parse_date_from_text(link_text)
                    
                    if date_obj and date_obj >= cutoff_date:
                        url_map[date_obj] = href
    
    print(f"  ✓ Found {len(url_map)} URLs for dates >= {cutoff_date.strftime('%Y-%m-%d')}")
    return url_map

def clean_price(price_str):
    """Clean price string and convert to float"""
    if not price_str or price_str.strip() in ['-', '', 'N/A', 'n/a']:
        return None
    cleaned = price_str.strip().replace(',', '').replace('Rs.', '').replace('/', '').replace('\t', '')
    try:
        return float(cleaned)
    except:
        return None

def extract_cinnamon_grades(soup, date_str):
    """Extract cinnamon grade prices from CINNAMON table - INCLUDING NATIONAL VALUES"""
    data = []
    
    cinnamon_anchor = soup.find('a', {'name': 'cinnamon'})
    
    if not cinnamon_anchor:
        return data
    
    table = cinnamon_anchor.find_next('table', {'id': 'rt2'})
    
    if not table:
        for tbl in soup.find_all('table'):
            headers = tbl.find_all('th')
            header_text = ' '.join([h.get_text().upper() for h in headers])
            if 'ALBA' in header_text or 'C-5' in header_text:
                table = tbl
                break
    
    if not table:
        return data
    
    thead = table.find('thead')
    if not thead:
        return data
    
    headers = []
    header_row = thead.find('tr')
    if header_row:
        ths = header_row.find_all('th')
        headers = [th.get_text(strip=True) for th in ths]
    
    # Parse grade names
    grades = []
    i = 1
    while i < len(headers):
        header = headers[i]
        match = re.match(r'(.+?)\s*\((?:Highest|Average)\s*Price\)', header, re.IGNORECASE)
        if match:
            grade_name = match.group(1).strip()
            if grade_name not in grades:
                grades.append(grade_name)
            i += 2
        else:
            i += 1
    
    tbody = table.find('tbody')
    if not tbody:
        return data
    
    rows = tbody.find_all('tr')
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 2:
            continue
        
        district = cells[0].get_text(strip=True)
        
        if not district or district.lower() in ['district', 'total']:
            continue
        
        is_national = district.lower() == 'national'
        
        col_idx = 1
        for grade in grades:
            if col_idx + 1 < len(cells):
                highest = clean_price(cells[col_idx].get_text(strip=True))
                average = clean_price(cells[col_idx + 1].get_text(strip=True))
                
                if highest is not None or average is not None:
                    record = {
                        'date': date_str,
                        'district': district,
                        'grade': grade,
                        'highest_price_rs_kg': highest,
                        'average_price_rs_kg': average
                    }
                    
                    if not is_national:
                        record['national_highest_price_rs_kg'] = None
                        record['national_average_price_rs_kg'] = None
                    
                    data.append(record)
                
                col_idx += 2
    
    # Create national values lookup and update district rows
    national_values = {}
    for record in data:
        if record['district'].lower() == 'national':
            national_values[record['grade']] = {
                'highest': record['highest_price_rs_kg'],
                'average': record['average_price_rs_kg']
            }
    
    updated_data = []
    for record in data:
        if record['district'].lower() != 'national':
            grade = record['grade']
            if grade in national_values:
                record['national_highest_price_rs_kg'] = national_values[grade]['highest']
                record['national_average_price_rs_kg'] = national_values[grade]['average']
            updated_data.append(record)
    
    return updated_data

def scrape_last_4_weeks(existing_csv_path=None):
    """Scrape ONLY last 4 weeks of data with national values"""
    print("\n" + "="*80)
    print("🔄 INCREMENTAL SCRAPER - Last 4 Weeks (with National Values)")
    print("="*80)
    
    cutoff_date = datetime.now() - timedelta(days=28)
    print(f"📅 Scraping dates from: {cutoff_date.strftime('%Y-%m-%d')} to today")
    
    url_map = fetch_recent_urls_from_index(cutoff_date)
    
    if not url_map:
        print("⚠️  No recent URLs found in index")
        return None
    
    all_grades = []
    success_count = 0
    
    sorted_dates = sorted(url_map.keys())
    
    print(f"\n📥 Scraping {len(sorted_dates)} recent reports...")
    
    for idx, date_obj in enumerate(sorted_dates, 1):
        date_str = date_obj.strftime('%d.%m.%Y')
        relative_url = url_map[date_obj]
        full_url = f"{BASE_URL}{relative_url}"
        
        print(f"  [{idx}/{len(sorted_dates)}] {date_str}...", end=" ")
        
        html = fetch_page(full_url)
        
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            grades = extract_cinnamon_grades(soup, date_str)
            
            if grades:
                success_count += 1
                all_grades.extend(grades)
                print(f"✓ {len(grades)} records")
            else:
                print("⚠ (no data)")
        else:
            print("✗")
        
        time.sleep(0.4)
    
    print(f"\n{'='*80}")
    print(f"✓ Scraped: {success_count}/{len(sorted_dates)} reports")
    print(f"📊 New records: {len(all_grades)}")
    
    if not all_grades:
        print("❌ No new data collected")
        return None
    
    new_df = pd.DataFrame(all_grades)
    
    # Merge with existing data if provided
    if existing_csv_path and os.path.exists(existing_csv_path):
        print(f"\n🔗 Merging with existing data: {existing_csv_path}")
        
        existing_df = pd.read_csv(existing_csv_path)
        print(f"  Existing records: {len(existing_df)}")
        
        # Check if existing data has national columns
        has_national_existing = 'national_highest_price_rs_kg' in existing_df.columns
        has_national_new = 'national_highest_price_rs_kg' in new_df.columns
        
        if not has_national_existing and has_national_new:
            print("  ⚠️  Existing data lacks national columns - adding them as NaN")
            existing_df['national_highest_price_rs_kg'] = None
            existing_df['national_average_price_rs_kg'] = None
        
        existing_df['date_parsed'] = pd.to_datetime(existing_df['date'], format='%d.%m.%Y')
        new_df['date_parsed'] = pd.to_datetime(new_df['date'], format='%d.%m.%Y')
        
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        
        combined = combined.drop_duplicates(
            subset=['date', 'district', 'grade'], 
            keep='last'
        )
        
        combined = combined.sort_values('date_parsed')
        combined = combined.drop('date_parsed', axis=1)
        
        print(f"  Combined records: {len(combined)}")
        print(f"  Net new: {len(combined) - len(existing_df)}")
        
        return combined
    else:
        new_df['date_parsed'] = pd.to_datetime(new_df['date'], format='%d.%m.%Y')
        new_df = new_df.sort_values('date_parsed')
        new_df = new_df.drop('date_parsed', axis=1)
        
        return new_df

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔄 INCREMENTAL CINNAMON SCRAPER (WITH NATIONAL VALUES)")
    print("   Scrapes ONLY last 4 weeks for weekly updates")
    print("="*80)
    
    existing_csv = "../data/cinnamon_grades.csv"
    
    if os.path.exists(existing_csv):
        print(f"\n✓ Found existing data: {existing_csv}")
        if os.environ.get('CI'):
            print("🤖 CI mode: Auto-merging with existing data")
        else:
            choice = input("Merge with existing data? (y/n): ").strip().lower()
            if choice != 'y':
                existing_csv = None
    else:
        print(f"\n⚠️  No existing CSV found at: {existing_csv}")
        existing_csv = None
    
    print("\n🚀 Starting incremental scrape...")
    
    df = scrape_last_4_weeks(existing_csv)
    
    if df is not None:
        output_file = "cinnamon_grades_updated.csv"
        df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("✅ INCREMENTAL SCRAPE COMPLETE!")
        print("="*80)
        print(f"📁 File: {output_file}")
        print(f"📊 Total records: {len(df):,}")
        print(f"📅 Date range: {df['date'].min()} → {df['date'].max()}")
        print(f"🏷️  Grades: {', '.join(sorted(df['grade'].unique()))}")
        print(f"📍 Districts: {df['district'].nunique()}")
        
        # Check for national columns
        if 'national_highest_price_rs_kg' in df.columns:
            print(f"✨ National benchmark columns: INCLUDED")
        
        print(f"\n📊 Latest 3 records:")
        print(df.tail(3).to_string(index=False))
        
        print(f"\n🎉 SUCCESS! Updated data saved to {output_file}")
    else:
        print("\n❌ Failed to collect data")