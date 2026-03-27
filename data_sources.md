# Data Sources Researched

## 1) Basketball-Reference
- Site: https://www.basketball-reference.com
- Why used:
  - Season-level performance data (`per_game`)
  - Season-level advanced analytics (`advanced`)
  - Player biodata + anthropometric basics from player directory pages (`height`, `weight`, `birth_date`, `college`, `position`)
- URLs used by scraper:
  - `https://www.basketball-reference.com/leagues/`
  - `https://www.basketball-reference.com/leagues/NBA_<SEASON>_per_game.html`
  - `https://www.basketball-reference.com/leagues/NBA_<SEASON>_advanced.html`
  - `https://www.basketball-reference.com/players/<LETTER>/`

## 2) NBADraft (Combine Measurements)
- Site: https://staging.nbadraft.net
- Why used:
  - Direct combine anthropometric fields not consistently available elsewhere from this environment
  - Includes `wingspan`, `standing reach`, `body fat %`, `hand length`, `hand width`, `height w/o shoes`, `height w/ shoes`, `weight`
- URL pattern used:
  - `https://staging.nbadraft.net/<YEAR>-nba-draft-combine-measurements/`

## Notes
- `stats.nba.com` draft combine endpoint schema was verified from NBA frontend bundles, but direct endpoint requests from this environment timed out consistently, so this run uses the two sources above for reliable extraction.
