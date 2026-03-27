#!/usr/bin/env python3
"""Scrape basketball player biodata, anthropometric data, performance stats,
and advanced analytics into a single CSV.

Sources:
- basketball-reference.com (season per-game + advanced tables, player directory biodata)
- staging.nbadraft.net (draft combine anthropometric measurements)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cloudscraper
import requests
from bs4 import BeautifulSoup


BREF_BASE = "https://www.basketball-reference.com"
NBA_DRAFT_BASE = "https://staging.nbadraft.net"

DEFAULT_OUTPUT = Path("data") / "player_dataset.csv"
DEFAULT_BIO_CACHE = Path("data") / "player_bio_cache.csv"

OUTPUT_COLUMNS: List[str] = [
    "entry_id",
    "record_type",
    "source",
    "source_url",
    "season_end_year",
    "season_label",
    "draft_year",
    "player_id",
    "player_name",
    "team_abbr",
    "position",
    "age",
    "birth_date",
    "college",
    "height_ft_in",
    "height_in",
    "weight_lb",
    "wingspan_ft_in",
    "wingspan_in",
    "standing_reach_ft_in",
    "standing_reach_in",
    "body_fat_pct",
    "hand_length_in",
    "hand_width_in",
    "games",
    "games_started",
    "minutes_per_game",
    "points_per_game",
    "assists_per_game",
    "rebounds_per_game",
    "steals_per_game",
    "blocks_per_game",
    "turnovers_per_game",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "per",
    "ts_pct",
    "usg_pct",
    "ows",
    "dws",
    "ws",
    "ws_per_48",
    "bpm",
    "vorp",
    "ast_pct",
    "trb_pct",
    "stl_pct",
    "blk_pct",
    "tov_pct",
    "raw_height_wo_shoes",
    "raw_height_w_shoes",
]


@dataclass
class RunStats:
    season_rows_written: int = 0
    combine_rows_written: int = 0
    season_rows_seen: int = 0
    combine_rows_seen: int = 0


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_space(text: str) -> str:
    return " ".join((text or "").replace("\xa0", " ").split())


def season_label(season_end_year: int) -> str:
    return f"{season_end_year - 1}-{str(season_end_year)[-2:]}"


def normalize_prime_chars(text: str) -> str:
    return (
        text.replace("′", "'")
        .replace("’", "'")
        .replace("‵", "'")
        .replace("″", '"')
        .replace("“", '"')
        .replace("”", '"')
    )


def parse_height_to_inches(height_text: str) -> Optional[float]:
    if not height_text:
        return None
    s = normalize_prime_chars(height_text)
    s = normalize_space(s)

    # basketball-reference style: 6-10
    m = re.search(r"(\d+)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        feet = float(m.group(1))
        inches = float(m.group(2))
        return feet * 12 + inches

    # combine style: 6' 8.75"
    m = re.search(r"(\d+)\s*'\s*(\d+(?:\.\d+)?)", s)
    if m:
        feet = float(m.group(1))
        inches = float(m.group(2))
        return feet * 12 + inches

    return None


def parse_float(text: str) -> Optional[float]:
    s = normalize_space(text)
    if not s or s == "-":
        return None
    s = s.replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_int(text: str) -> Optional[int]:
    s = normalize_space(text)
    if not s or s == "-":
        return None
    s = s.replace(",", "")
    try:
        return int(float(s))
    except ValueError:
        return None


def make_scraper() -> cloudscraper.CloudScraper:
    return cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )


def fetch_html(scraper: cloudscraper.CloudScraper, url: str, retries: int = 4) -> str:
    err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = scraper.get(url, timeout=45)
            if resp.status_code == 200 and "Just a moment" not in resp.text:
                return resp.text
            if resp.status_code == 429:
                # Basketball-Reference aggressively rate-limits bursts.
                cooldown = min(30 * attempt, 180)
                err = RuntimeError(f"HTTP 429 for {url}")
                log(f"Rate-limited on {url}; cooling down for {cooldown}s")
                time.sleep(cooldown)
                continue
            err = RuntimeError(f"HTTP {resp.status_code} for {url}")
        except Exception as exc:  # noqa: BLE001
            err = exc
        sleep_for = min(2.5 * attempt, 12)
        log(f"Retry {attempt}/{retries} for {url} after error: {err}")
        time.sleep(sleep_for)
    raise RuntimeError(f"Failed to fetch {url}: {err}")


def fetch_soup(scraper: cloudscraper.CloudScraper, url: str) -> BeautifulSoup:
    return BeautifulSoup(fetch_html(scraper, url), "html.parser")


def load_existing_entry_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry_id = (row.get("entry_id") or "").strip()
            if entry_id:
                ids.add(entry_id)
    return ids


def append_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            out = {col: "" for col in OUTPUT_COLUMNS}
            for col in OUTPUT_COLUMNS:
                val = row.get(col, "")
                out[col] = "" if val is None else val
            writer.writerow(out)


def load_bio_cache(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    data: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_id = normalize_space(row.get("player_id", ""))
            if not player_id:
                continue
            data[player_id] = {
                "player_id": player_id,
                "player_name": normalize_space(row.get("player_name", "")),
                "position": normalize_space(row.get("position", "")),
                "height_ft_in": normalize_space(row.get("height_ft_in", "")),
                "height_in": parse_float(row.get("height_in", "")),
                "weight_lb": parse_int(row.get("weight_lb", "")),
                "birth_date": normalize_space(row.get("birth_date", "")),
                "college": normalize_space(row.get("college", "")),
            }
    return data


def save_bio_cache(path: Path, bio_by_id: Dict[str, Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "player_id",
        "player_name",
        "position",
        "height_ft_in",
        "height_in",
        "weight_lb",
        "birth_date",
        "college",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for player_id in sorted(bio_by_id):
            row = bio_by_id[player_id]
            out = {c: row.get(c, "") for c in cols}
            writer.writerow(out)


def scrape_player_directory_bio(scraper: cloudscraper.CloudScraper) -> Dict[str, Dict[str, object]]:
    bio_by_id: Dict[str, Dict[str, object]] = {}
    log("Scraping player directory biodata (A-Z)...")
    for letter in string.ascii_lowercase:
        url = f"{BREF_BASE}/players/{letter}/"
        soup = fetch_soup(scraper, url)
        table = soup.find("table", id="players")
        if table is None or table.tbody is None:
            log(f"Skipping {url}: no players table")
            continue

        count = 0
        for tr in table.tbody.find_all("tr"):
            if "thead" in (tr.get("class") or []):
                continue
            player_cell = tr.find(attrs={"data-stat": "player"})
            if not player_cell:
                continue

            player_id = player_cell.get("data-append-csv")
            if not player_id:
                continue

            row: Dict[str, object] = {
                "player_id": player_id,
                "player_name": normalize_space(player_cell.get_text(" ", strip=True)),
                "position": normalize_space(
                    (tr.find(attrs={"data-stat": "pos"}) or {}).get_text(" ", strip=True)
                    if tr.find(attrs={"data-stat": "pos"})
                    else ""
                ),
                "height_ft_in": normalize_space(
                    (tr.find(attrs={"data-stat": "height"}) or {}).get_text(" ", strip=True)
                    if tr.find(attrs={"data-stat": "height"})
                    else ""
                ),
                "weight_lb": parse_int(
                    (tr.find(attrs={"data-stat": "weight"}) or {}).get_text(" ", strip=True)
                    if tr.find(attrs={"data-stat": "weight"})
                    else ""
                ),
                "birth_date": normalize_space(
                    (tr.find(attrs={"data-stat": "birth_date"}) or {}).get_text(" ", strip=True)
                    if tr.find(attrs={"data-stat": "birth_date"})
                    else ""
                ),
                "college": normalize_space(
                    (tr.find(attrs={"data-stat": "colleges"}) or {}).get_text(" ", strip=True)
                    if tr.find(attrs={"data-stat": "colleges"})
                    else ""
                ),
            }
            row["height_in"] = parse_height_to_inches(str(row["height_ft_in"]))
            bio_by_id[player_id] = row
            count += 1

        log(f"  {letter.upper()}: {count} player bios")
        time.sleep(0.3)

    log(f"Collected {len(bio_by_id)} player bios from directory pages.")
    return bio_by_id


def get_available_seasons(scraper: cloudscraper.CloudScraper) -> List[int]:
    url = f"{BREF_BASE}/leagues/"
    soup = fetch_soup(scraper, url)
    seasons: set[int] = set()
    for a in soup.select("table#stats a[href*='/leagues/NBA_']"):
        href = a.get("href", "")
        m = re.search(r"/leagues/NBA_(\d+)\.html", href)
        if m:
            seasons.add(int(m.group(1)))
    return sorted(seasons)


def parse_bref_table_rows(
    scraper: cloudscraper.CloudScraper,
    url: str,
    table_id: str,
) -> List[Dict[str, str]]:
    soup = fetch_soup(scraper, url)
    table = soup.find("table", id=table_id)
    if table is None or table.tbody is None:
        raise RuntimeError(f"Could not find table '{table_id}' at {url}")

    rows: List[Dict[str, str]] = []
    for tr in table.tbody.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        row: Dict[str, str] = {}
        player_cell = None

        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if not stat:
                continue
            row[stat] = normalize_space(cell.get_text(" ", strip=True))
            if stat in {"name_display", "player"}:
                player_cell = cell

        if not row:
            continue

        if player_cell is not None:
            row["player_id"] = player_cell.get("data-append-csv", "")
            link = player_cell.find("a")
            row["player_href"] = link.get("href", "") if link else ""

        rows.append(row)

    return rows


def season_row_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    player_id = row.get("player_id", "")
    player_name = row.get("name_display", "")
    team = row.get("team_name_abbr", "")
    age = row.get("age", "")
    return (player_id, player_name, team, age)


def build_season_output_rows(
    season_end_year: int,
    per_rows: List[Dict[str, str]],
    adv_rows: List[Dict[str, str]],
    bio_by_id: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    per_map = {season_row_key(r): r for r in per_rows}
    adv_map = {season_row_key(r): r for r in adv_rows}
    keys = set(per_map.keys()) | set(adv_map.keys())

    out_rows: List[Dict[str, object]] = []
    for key in keys:
        p = per_map.get(key, {})
        a = adv_map.get(key, {})

        player_id = p.get("player_id") or a.get("player_id") or ""
        player_name = p.get("name_display") or a.get("name_display") or ""
        team_abbr = p.get("team_name_abbr") or a.get("team_name_abbr") or ""
        age = p.get("age") or a.get("age") or ""

        bio = bio_by_id.get(player_id, {}) if player_id else {}
        height_ft_in = str(bio.get("height_ft_in") or "")
        height_in = bio.get("height_in")
        weight_lb = bio.get("weight_lb")

        entry_key = f"season_stats|{season_end_year}|{player_id or player_name}|{team_abbr}"
        entry_id = hashlib.sha1(entry_key.encode("utf-8")).hexdigest()

        row: Dict[str, object] = {
            "entry_id": entry_id,
            "record_type": "season_stats",
            "source": "basketball-reference.com",
            "source_url": f"{BREF_BASE}/leagues/NBA_{season_end_year}_per_game.html",
            "season_end_year": season_end_year,
            "season_label": season_label(season_end_year),
            "player_id": player_id,
            "player_name": player_name,
            "team_abbr": team_abbr,
            "position": p.get("pos") or a.get("pos") or bio.get("position") or "",
            "age": age,
            "birth_date": bio.get("birth_date") or "",
            "college": bio.get("college") or "",
            "height_ft_in": height_ft_in,
            "height_in": height_in,
            "weight_lb": weight_lb,
            "games": parse_int(p.get("games", "") or a.get("games", "")),
            "games_started": parse_int(p.get("games_started", "") or a.get("games_started", "")),
            "minutes_per_game": parse_float(p.get("mp_per_g", "")),
            "points_per_game": parse_float(p.get("pts_per_g", "")),
            "assists_per_game": parse_float(p.get("ast_per_g", "")),
            "rebounds_per_game": parse_float(p.get("trb_per_g", "")),
            "steals_per_game": parse_float(p.get("stl_per_g", "")),
            "blocks_per_game": parse_float(p.get("blk_per_g", "")),
            "turnovers_per_game": parse_float(p.get("tov_per_g", "")),
            "fg_pct": parse_float(p.get("fg_pct", "")),
            "fg3_pct": parse_float(p.get("fg3_pct", "")),
            "ft_pct": parse_float(p.get("ft_pct", "")),
            "per": parse_float(a.get("per", "")),
            "ts_pct": parse_float(a.get("ts_pct", "")),
            "usg_pct": parse_float(a.get("usg_pct", "")),
            "ows": parse_float(a.get("ows", "")),
            "dws": parse_float(a.get("dws", "")),
            "ws": parse_float(a.get("ws", "")),
            "ws_per_48": parse_float(a.get("ws_per_48", "")),
            "bpm": parse_float(a.get("bpm", "")),
            "vorp": parse_float(a.get("vorp", "")),
            "ast_pct": parse_float(a.get("ast_pct", "")),
            "trb_pct": parse_float(a.get("trb_pct", "")),
            "stl_pct": parse_float(a.get("stl_pct", "")),
            "blk_pct": parse_float(a.get("blk_pct", "")),
            "tov_pct": parse_float(a.get("tov_pct", "")),
        }
        out_rows.append(row)

    return out_rows


def discover_nbadraft_combine_years(start_year: int = 2010, end_year: int = 2030) -> List[int]:
    years: List[int] = []
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    for year in range(start_year, end_year + 1):
        url = f"{NBA_DRAFT_BASE}/{year}-nba-draft-combine-measurements/"
        try:
            resp = session.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                continue
            if "<table" in resp.text.lower() and "Wingspan" in resp.text:
                years.append(year)
        except Exception:  # noqa: BLE001
            continue
        time.sleep(0.15)

    return years


def parse_nbadraft_combine_table(html: str) -> Tuple[List[str], List[List[str]]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return [], []

    tbodies = table.find_all("tbody")
    if not tbodies:
        return [], []

    header_cells: List[str] = []
    first_body_rows = tbodies[0].find_all("tr")
    if first_body_rows:
        header_cells = [
            normalize_space(td.get_text(" ", strip=True))
            for td in first_body_rows[0].find_all(["td", "th"])
        ]

    data_rows: List[List[str]] = []
    for body in tbodies[1:]:
        for tr in body.find_all("tr"):
            cells = [normalize_space(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            if cells:
                data_rows.append(cells)

    return header_cells, data_rows


def extract_player_name_and_pos(raw_player_cell: str) -> Tuple[str, str]:
    s = normalize_space(raw_player_cell)
    # Typical format: "Bam Adebayo PF-C"
    m = re.match(r"^(.*?)(?:\s+([A-Z]{1,3}(?:-[A-Z]{1,3})*))?$", s)
    if not m:
        return s, ""
    name = (m.group(1) or "").strip()
    pos = (m.group(2) or "").strip()
    return name, pos


def build_combine_rows(
    year: int,
    html: str,
    player_name_to_id: Dict[str, str],
) -> List[Dict[str, object]]:
    headers, raw_rows = parse_nbadraft_combine_table(html)
    if not headers or not raw_rows:
        return []

    idx = {normalize_space(h).lower(): i for i, h in enumerate(headers)}

    def cell(row: List[str], key: str) -> str:
        i = idx.get(key.lower())
        if i is None or i >= len(row):
            return ""
        return row[i]

    out: List[Dict[str, object]] = []
    for rr in raw_rows:
        raw_player = cell(rr, "Player")
        player_name, pos = extract_player_name_and_pos(raw_player)
        if not player_name:
            continue

        h_wo = cell(rr, "Height w/o Shoes")
        h_w = cell(rr, "Height w/ Shoes")
        wingspan = cell(rr, "Wingspan")
        standing_reach = cell(rr, "Standing Reach")
        weight = cell(rr, "Weight")
        body_fat = cell(rr, "Body Fat %")
        hand_length = cell(rr, "Hand Length")
        hand_width = cell(rr, "Hand Width")

        entry_key = f"combine_measurement|{year}|{player_name}"
        entry_id = hashlib.sha1(entry_key.encode("utf-8")).hexdigest()

        normalized_name = normalize_space(player_name).lower()
        player_id = player_name_to_id.get(normalized_name, "")

        out.append(
            {
                "entry_id": entry_id,
                "record_type": "combine_measurement",
                "source": "staging.nbadraft.net",
                "source_url": f"{NBA_DRAFT_BASE}/{year}-nba-draft-combine-measurements/",
                "season_end_year": year,
                "season_label": season_label(year),
                "draft_year": year,
                "player_id": player_id,
                "player_name": player_name,
                "position": pos,
                "height_ft_in": h_wo,
                "height_in": parse_height_to_inches(h_wo),
                "weight_lb": parse_float(weight),
                "wingspan_ft_in": wingspan,
                "wingspan_in": parse_height_to_inches(wingspan),
                "standing_reach_ft_in": standing_reach,
                "standing_reach_in": parse_height_to_inches(standing_reach),
                "body_fat_pct": parse_float(body_fat),
                "hand_length_in": parse_float(hand_length),
                "hand_width_in": parse_float(hand_width),
                "raw_height_wo_shoes": h_wo,
                "raw_height_w_shoes": h_w,
            }
        )

    return out


def scrape_combine_rows(
    player_name_to_id: Dict[str, str],
    start_year: int,
    end_year: int,
) -> Tuple[List[Tuple[int, List[Dict[str, object]]]], List[int]]:
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    years = discover_nbadraft_combine_years(start_year, end_year)
    if not years:
        return [], []

    batches: List[Tuple[int, List[Dict[str, object]]]] = []
    for year in years:
        url = f"{NBA_DRAFT_BASE}/{year}-nba-draft-combine-measurements/"
        try:
            resp = session.get(url, headers=headers, timeout=45)
            if resp.status_code != 200:
                continue
            rows = build_combine_rows(year, resp.text, player_name_to_id)
            batches.append((year, rows))
        except Exception as exc:  # noqa: BLE001
            log(f"Combine year {year} failed: {exc}")
        time.sleep(0.2)

    return batches, years


def count_csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        # minus header
        return max(sum(1 for _ in f) - 1, 0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape basketball data into CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bio-cache", type=Path, default=DEFAULT_BIO_CACHE)
    parser.add_argument("--start-season", type=int, default=1950, help="NBA season end year")
    parser.add_argument("--end-season", type=int, default=None, help="NBA season end year")
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip NBADraft combine anthropometric scraping",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.25,
        help="Pause between season requests",
    )
    args = parser.parse_args(argv)

    scraper = make_scraper()
    output = args.output

    existing_ids = load_existing_entry_ids(output)
    log(f"Existing rows detected: {len(existing_ids)}")

    bio_cache_path = args.bio_cache
    bio_by_id = load_bio_cache(bio_cache_path)
    if bio_by_id:
        log(f"Loaded {len(bio_by_id)} player bios from cache: {bio_cache_path}")
    else:
        bio_by_id = scrape_player_directory_bio(scraper)
        save_bio_cache(bio_cache_path, bio_by_id)
        log(f"Saved player bio cache: {bio_cache_path}")
    name_to_id = {
        normalize_space(str(v.get("player_name", ""))).lower(): pid
        for pid, v in bio_by_id.items()
        if v.get("player_name")
    }

    seasons = get_available_seasons(scraper)
    if not seasons:
        raise RuntimeError("No available NBA seasons discovered.")

    end_season = args.end_season or seasons[-1]
    selected_seasons = [s for s in seasons if args.start_season <= s <= end_season]
    if not selected_seasons:
        raise RuntimeError("No seasons selected after filtering.")

    log(
        f"Scraping season stats for {len(selected_seasons)} seasons "
        f"({selected_seasons[0]} to {selected_seasons[-1]})"
    )

    stats = RunStats()

    for season in selected_seasons:
        per_url = f"{BREF_BASE}/leagues/NBA_{season}_per_game.html"
        adv_url = f"{BREF_BASE}/leagues/NBA_{season}_advanced.html"

        per_rows: List[Dict[str, str]] = []
        adv_rows: List[Dict[str, str]] = []

        try:
            per_rows = parse_bref_table_rows(scraper, per_url, "per_game_stats")
        except Exception as exc:  # noqa: BLE001
            log(f"Season {season} per-game fetch failed: {exc}")

        try:
            adv_rows = parse_bref_table_rows(scraper, adv_url, "advanced")
        except Exception as exc:  # noqa: BLE001
            log(f"Season {season} advanced fetch failed: {exc}")

        if not per_rows and not adv_rows:
            log(f"Season {season} skipped: no table data available.")
            continue

        season_rows = build_season_output_rows(season, per_rows, adv_rows, bio_by_id)
        stats.season_rows_seen += len(season_rows)

        new_rows = [r for r in season_rows if str(r.get("entry_id", "")) not in existing_ids]
        for r in new_rows:
            existing_ids.add(str(r.get("entry_id", "")))

        append_rows(output, new_rows)
        stats.season_rows_written += len(new_rows)

        total_rows = count_csv_rows(output)
        log(
            f"Season {season}: seen={len(season_rows)} new={len(new_rows)} "
            f"running_csv_rows={total_rows}"
        )
        time.sleep(max(args.pause_seconds, 0.0))

    if not args.skip_combine:
        combine_batches, years = scrape_combine_rows(
            player_name_to_id=name_to_id,
            start_year=max(2010, args.start_season),
            end_year=end_season,
        )
        log(f"Combine years discovered: {years}")

        for year, rows in combine_batches:
            stats.combine_rows_seen += len(rows)
            new_rows = [r for r in rows if str(r.get("entry_id", "")) not in existing_ids]
            for r in new_rows:
                existing_ids.add(str(r.get("entry_id", "")))
            append_rows(output, new_rows)
            stats.combine_rows_written += len(new_rows)

            total_rows = count_csv_rows(output)
            log(
                f"Combine {year}: seen={len(rows)} new={len(new_rows)} "
                f"running_csv_rows={total_rows}"
            )

    final_rows = count_csv_rows(output)
    log("\nRun complete.")
    log(f"Season rows seen: {stats.season_rows_seen}")
    log(f"Season rows written: {stats.season_rows_written}")
    log(f"Combine rows seen: {stats.combine_rows_seen}")
    log(f"Combine rows written: {stats.combine_rows_written}")
    log(f"Total CSV rows: {final_rows}")

    if final_rows < 10_000:
        log("WARNING: total rows are below 10,000. Expand season range or rerun.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
