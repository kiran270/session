"""
Fetch historical cricket odds from The Odds API (v4).

Flow:
1. GET /v4/historical/sports/{sport}/events -> find the event id for the match
   (matched by team names and/or commence time close to the given date).
2. GET /v4/historical/sports/{sport}/events/{eventId}/odds -> get bookmaker
   odds for that event at the given historical timestamp.

Docs: https://the-odds-api.com/liveapi/guides/v4/

Note: historical endpoints are only available on paid usage plans.
"""

import argparse
import sys
from datetime import datetime

import requests

BASE_URL = "https://api.the-odds-api.com"

# Common cricket sport keys used by The Odds API. Adjust as needed.
CRICKET_SPORT_KEYS = [
    "cricket_test_match",
    "cricket_odi",
    "cricket_t20",
    "cricket_ipl",
    "cricket_big_bash",
    "cricket_psl",
    "cricket_international_t20",
]


def get_historical_events(api_key, sport, date_iso, team_a=None, team_b=None):
    """Fetch the list of historical events available at the given timestamp."""
    url = f"{BASE_URL}/v4/historical/sports/{sport}/events"
    params = {"apiKey": api_key, "date": date_iso}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    events = payload.get("data", [])
    if team_a and team_b:
        team_a_lower, team_b_lower = team_a.lower(), team_b.lower()
        events = [
            e for e in events
            if team_a_lower in (e.get("home_team", "").lower(), e.get("away_team", "").lower())
            and team_b_lower in (e.get("home_team", "").lower(), e.get("away_team", "").lower())
        ]

    return payload, events


def get_historical_event_odds(api_key, sport, event_id, date_iso, regions="uk,eu,us,au", markets="h2h"):
    """Fetch historical odds for a single event at the given timestamp."""
    url = f"{BASE_URL}/v4/historical/sports/{sport}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "dateFormat": "iso",
        "oddsFormat": "decimal",
        "date": date_iso,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json(), resp.headers


def print_odds(odds_payload):
    data = odds_payload.get("data", {})
    print(f"\nSnapshot timestamp : {odds_payload.get('timestamp')}")
    print(f"Previous timestamp : {odds_payload.get('previous_timestamp')}")
    print(f"Next timestamp     : {odds_payload.get('next_timestamp')}")
    print(f"\nMatch: {data.get('home_team')} vs {data.get('away_team')}")
    print(f"Commence time: {data.get('commence_time')}\n")

    for bookmaker in data.get("bookmakers", []):
        print(f"Bookmaker: {bookmaker.get('title')} (updated {bookmaker.get('last_update')})")
        for market in bookmaker.get("markets", []):
            print(f"  Market: {market.get('key')}")
            for outcome in market.get("outcomes", []):
                print(f"    {outcome.get('name'):<25} {outcome.get('price')}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Fetch historical cricket odds from The Odds API")
    parser.add_argument("--api-key", required=True, help="The Odds API key")
    parser.add_argument("--sport", required=True, help=f"Sport key, e.g. one of: {', '.join(CRICKET_SPORT_KEYS)}")
    parser.add_argument("--date", required=True, help="Snapshot timestamp, ISO8601 e.g. 2023-11-19T14:00:00Z")
    parser.add_argument("--team-a", help="First team name to filter the match (optional)")
    parser.add_argument("--team-b", help="Second team name to filter the match (optional)")
    parser.add_argument("--event-id", help="Skip event lookup and use this event id directly")
    parser.add_argument("--regions", default="uk,eu,us,au", help="Comma separated regions (default: uk,eu,us,au)")
    parser.add_argument("--markets", default="h2h", help="Comma separated markets (default: h2h)")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        print("Error: --date must be ISO8601 format, e.g. 2023-11-19T14:00:00Z", file=sys.stderr)
        sys.exit(1)

    event_id = args.event_id

    if not event_id:
        print(f"Looking up historical events for sport '{args.sport}' at {args.date} ...")
        try:
            events_payload, matches = get_historical_events(
                args.api_key, args.sport, args.date, args.team_a, args.team_b
            )
        except requests.HTTPError as exc:
            print(f"Error fetching events: {exc} - {exc.response.text}", file=sys.stderr)
            sys.exit(1)

        candidates = matches if (args.team_a and args.team_b) else events_payload.get("data", [])

        if not candidates:
            print("No matching events found for the given date/teams.", file=sys.stderr)
            sys.exit(1)

        if len(candidates) > 1 and not (args.team_a and args.team_b):
            print(f"Found {len(candidates)} events at this snapshot. Showing available matches:\n")
            for e in candidates:
                print(f"  id={e['id']}  {e.get('home_team')} vs {e.get('away_team')}  ({e.get('commence_time')})")
            print("\nRe-run with --event-id <id>, or with --team-a/--team-b to narrow it down.")
            sys.exit(0)

        event = candidates[0]
        event_id = event["id"]
        print(f"Using event: {event.get('home_team')} vs {event.get('away_team')} (id={event_id})")

    print(f"\nFetching historical odds for event {event_id} at {args.date} ...")
    try:
        odds_payload, headers = get_historical_event_odds(
            args.api_key, args.sport, event_id, args.date, args.regions, args.markets
        )
    except requests.HTTPError as exc:
        print(f"Error fetching odds: {exc} - {exc.response.text}", file=sys.stderr)
        sys.exit(1)

    print_odds(odds_payload)

    print("Quota usage:")
    print(f"  Requests used     : {headers.get('x-requests-used')}")
    print(f"  Requests remaining: {headers.get('x-requests-remaining')}")
    print(f"  Last request cost : {headers.get('x-requests-last')}")


if __name__ == "__main__":
    main()
