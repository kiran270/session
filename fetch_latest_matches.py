"""
Fetch the latest ODI matches from ESPNcricinfo and append them to MODI.csv
in the exact format the score-prediction model expects.

How it works
------------
1. Read the target CSV and find the most recent match date already stored.
2. Scrape the legacy stats records page (plain HTTP works there) to list every
   ODI of the given year with its match id, teams, ground and date.
3. For each match newer than what we already have, open the match page in a
   real Chrome browser (undetected-chromedriver, non-headless -- Akamai blocks
   headless and plain HTTP). The legacy match URL redirects to the canonical
   slug URL; we switch that to the "match-overs-comparison" view whose embedded
   __NEXT_DATA__ JSON holds per-over cumulative scores.
4. Build two rows per match (one per innings) and append them to the CSV.

CSV columns: Match, Batting_Team, Bowling_Team, Venue, Innings, Date,
             Over 1 .. Over 50   (cells look like ="123/4", or N/A)

Data is scraped from an external site; treat it as untrusted and review
ESPNcricinfo's terms before running this regularly.

Usage
-----
    python fetch_latest_matches.py                 # 2026 ODIs -> MODI.csv
    python fetch_latest_matches.py --year 2026 --csv MODI.csv --overs 50
    python fetch_latest_matches.py --dry-run       # scrape but don't write
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

# undetected_chromedriver needs distutils; on py3.12 setuptools provides a shim.
import undetected_chromedriver as uc

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
STATS_RECORDS_URL = (
    "https://stats.espncricinfo.com/ci/engine/records/team/match_results.html"
    "?class=2;id={year};type=year"
)
LEGACY_MATCH_URL = "https://www.cricinfo.com/ci/engine/match/{match_id}.html"
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)
# Fallback: the overs-comparison JSON blob is a plain {"props":...} script.
PROPS_BLOB_RE = re.compile(r'(\{"props":.*?\})</script>', re.S)


# --------------------------------------------------------------------------- #
# Step 1 & 2: discover matches from the reachable legacy records page
# --------------------------------------------------------------------------- #
def discover_matches(year):
    """Return a list of match dicts for the given year's ODIs."""
    url = STATS_RECORDS_URL.format(year=year)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results_table = None
    for tab in soup.find_all("table", class_="engineTable"):
        headers = [th.get_text(strip=True) for th in tab.find_all("th")]
        if "Winner" in headers and "Match Date" in headers:
            results_table = tab
            break
    if results_table is None:
        raise RuntimeError("Could not find the match-results table on the page.")

    matches = []
    for row in results_table.find_all("tr", class_="data1"):
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 7:
            continue
        team1, team2, _winner, _margin, ground, date_str, _card = cells[:7]

        match_id = None
        for a in row.find_all("a", href=True):
            m = re.search(r"/match/(\d+)\.html", a["href"])
            if m:
                match_id = m.group(1)
                break
        if not match_id:
            continue

        try:
            date = datetime.strptime(date_str, "%b %d, %Y").date()
        except ValueError:
            continue

        matches.append(
            {
                "match_id": match_id,
                "team1": team1,
                "team2": team2,
                "ground": ground,
                "date": date,
            }
        )
    matches.sort(key=lambda m: m["date"])
    return matches


# --------------------------------------------------------------------------- #
# Step 3: browser scrape per-over data
# --------------------------------------------------------------------------- #
def build_driver():
    opts = uc.ChromeOptions()
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument(f"user-agent={USER_AGENT}")
    # Non-headless on purpose: Akamai blocks headless sessions.
    return uc.Chrome(options=opts, version_main=153)


def _extract_next_data(html):
    m = NEXT_DATA_RE.search(html)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: find the largest {"props":...} script that has inningOvers
    for blob in PROPS_BLOB_RE.findall(html):
        if "inningOvers" in blob:
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                continue
    return None


def scrape_match_overs(driver, match_id, wait=9):
    """Return (match_meta, innings_list) for a single match id, or None.

    Raises on a dead browser session so the caller can rebuild the driver.
    """
    driver.get(LEGACY_MATCH_URL.format(match_id=match_id))
    time.sleep(wait)

    canonical = driver.current_url
    if "/full-scorecard" not in canonical:
        return None  # redirect failed or match page not available

    overs_url = canonical.replace("/full-scorecard", "/match-overs-comparison")
    driver.get(overs_url)
    time.sleep(wait)
    html = driver.page_source
    if "Access Denied" in html and len(html) < 3000:
        return None

    data = _extract_next_data(html)
    if not data:
        return None
    try:
        app = data["props"]["appPageProps"]["data"]
    except (KeyError, TypeError):
        return None

    match = app.get("match") or {}
    content = app.get("content") or {}
    innings = content.get("innings") or []
    if not innings:
        return None
    return match, innings


# --------------------------------------------------------------------------- #
# Step 4: format rows for the CSV
# --------------------------------------------------------------------------- #
def over_cell(total_runs, total_wickets):
    """Match the Excel-text format used in the CSV: ="123/4"."""
    return f'="{total_runs}/{total_wickets}"'


def build_rows(match, innings, num_overs, match_label):
    """Turn one scraped match into CSV row dicts (one per innings)."""
    ground = (match.get("ground") or {}).get("name") or ""
    start_date = (match.get("startDate") or "")[:10]

    team_names = [(inn.get("team") or {}).get("longName") for inn in innings]

    rows = []
    over_cols = [f"Over {i}" for i in range(1, num_overs + 1)]
    for idx, inn in enumerate(innings):
        batting = (inn.get("team") or {}).get("longName") or ""
        bowling = team_names[1 - idx] if len(team_names) == 2 else ""

        row = {
            "Match": match_label,
            "Batting_Team": batting,
            "Bowling_Team": bowling,
            "Venue": ground,
            "Innings": inn.get("inningNumber", idx + 1),
            "Date": start_date,
        }
        for col in over_cols:
            row[col] = "N/A"

        for ov in inn.get("inningOvers", []):
            n = ov.get("overNumber")
            if n is None or n < 1 or n > num_overs:
                continue
            row[f"Over {n}"] = over_cell(ov.get("totalRuns", 0), ov.get("totalWickets", 0))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Append latest ODIs to the CSV")
    parser.add_argument("--csv", default="MODI.csv", help="Target CSV (default MODI.csv)")
    parser.add_argument("--year", type=int, default=datetime.now().year,
                        help="Records year to scrape (default: current year)")
    parser.add_argument("--overs", type=int, default=50, help="Overs per innings (default 50)")
    parser.add_argument("--wait", type=int, default=9, help="Seconds to wait per page load")
    parser.add_argument("--limit", type=int, default=0, help="Max new matches to fetch (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Scrape but do not write the CSV")
    args = parser.parse_args()

    # Step 1: current state of the CSV
    df = pd.read_csv(args.csv)
    existing_dates = pd.to_datetime(df["Date"], errors="coerce")
    last_date = existing_dates.max()
    last_match_num = 0
    for label in df["Match"].dropna():
        m = re.search(r"(\d+)", str(label))
        if m:
            last_match_num = max(last_match_num, int(m.group(1)))

    # Signatures of matches already stored, to avoid duplicates on re-run.
    existing_sig = {
        (str(r["Batting_Team"]).strip(), str(r["Bowling_Team"]).strip(),
         str(r["Date"])[:10])
        for _, r in df.iterrows()
    }
    print(f"CSV '{args.csv}': {len(df)} rows, latest date loaded = "
          f"{last_date.date() if pd.notna(last_date) else 'none'}")

    # Step 2: discover candidate matches
    print(f"Discovering {args.year} ODIs from ESPNcricinfo records ...")
    matches = discover_matches(args.year)
    print(f"  found {len(matches)} matches for {args.year}")

    # Include same-day matches (>=) but rely on signature dedup below, so a
    # crashed run can resume without duplicating already-saved matches.
    cutoff = last_date.date() if pd.notna(last_date) else None
    new_matches = [m for m in matches if cutoff is None or m["date"] >= cutoff]

    def already_have(meta):
        d = meta["date"].isoformat()
        return (
            (meta["team1"], meta["team2"], d) in existing_sig
            or (meta["team2"], meta["team1"], d) in existing_sig
        )

    new_matches = [m for m in new_matches if not already_have(m)]
    if args.limit:
        new_matches = new_matches[: args.limit]

    if not new_matches:
        print("Nothing new to fetch. CSV is already up to date.")
        return

    print(f"  {len(new_matches)} new match(es) to fetch:")
    for m in new_matches:
        print(f"    {m['date']}  {m['team1']} vs {m['team2']}  (id={m['match_id']})")

    # Step 3 + 4: scrape and build rows.
    # The browser occasionally dies mid-run, so we rebuild it on failure and
    # append to the CSV after every successful match (crash-safe / resumable).
    driver = build_driver()
    match_counter = last_match_num
    total_added = 0

    def restart_driver(old):
        try:
            old.quit()
        except Exception:
            pass
        print("    restarting browser ...")
        time.sleep(2)
        return build_driver()

    try:
        for i, meta in enumerate(new_matches, 1):
            print(f"[{i}/{len(new_matches)}] scraping id={meta['match_id']} "
                  f"({meta['team1']} vs {meta['team2']}) ...")

            scraped = None
            for attempt in range(2):
                try:
                    scraped = scrape_match_overs(driver, meta["match_id"], wait=args.wait)
                    break
                except Exception as exc:
                    print(f"    attempt {attempt + 1} failed: {str(exc)[:120]}")
                    driver = restart_driver(driver)

            if not scraped:
                print("    no per-over data found, skipping.")
                continue

            match, innings = scraped
            match_counter += 1
            rows = build_rows(match, innings, args.overs,
                              match_label=f"Match {match_counter}")

            if args.dry_run:
                print(f"    ok (dry-run): {len(rows)} innings row(s).")
                continue

            # Append immediately so progress survives a later crash.
            new_df = pd.DataFrame(rows, columns=df.columns)
            new_df.to_csv(args.csv, mode="a", header=False, index=False)
            total_added += len(rows)
            print(f"    ok: appended {len(rows)} row(s) (running total {total_added}).")
    finally:
        try:
            driver.quit()
        except OSError:
            pass

    if args.dry_run:
        print("\nDry run complete - nothing written.")
    else:
        print(f"\nDone. Appended {total_added} row(s) to '{args.csv}'.")


if __name__ == "__main__":
    main()
