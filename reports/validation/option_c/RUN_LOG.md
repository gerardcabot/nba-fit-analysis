# Option C validation run

- Season: `2024-25`
- Started: `2026-05-26T20:48:01.311903+00:00`
- Impact max games: `30`

## Command log

```text
Option C validation — season 2024-25
Skipping mvp/role ingest — interim tables cached for 2024-25

$ C:\Users\geric\AppData\Local\Programs\Python\Python310\python.exe -m nba_fit train-roles --season 2024-25
Training Option B role model season=2024-25...
  players: 569
  archetypes: 6 cluster labels
  embedding dim: 12
  saved: C:\Users\geric\Desktop\REPOS\propis\nba-fit-analysis\models\role_embedding\season=2024-25

$ C:\Users\geric\AppData\Local\Programs\Python\Python310\python.exe -m nba_fit ingest --tier impact --season 2024-25 --max-games 30
Ingest tier=impact season=2024-25 max_games=30 (cache=on)...
  possessions: 5991 rows (30 games, 30 files)
    first: C:\Users\geric\Desktop\REPOS\propis\nba-fit-analysis\data\interim\possessions\season=2024-25\game_id=0042400165\data.parquet
    leaguegamefinder: 33 rows (cache)
    playbyplayv3: 14634 rows (live)
    gamerotation: 2066 rows (live)

$ C:\Users\geric\AppData\Local\Programs\Python\Python310\python.exe -m nba_fit train-impact --season 2024-25
Training Option C impact models season=2024-25...
  rapm players: 549 (549 low-sample flags)
  net rapm range: 0.00 .. 0.00
  lineup model penalty: ridge
  saved rapm: C:\Users\geric\Desktop\REPOS\propis\nba-fit-analysis\models\rapm\season=2024-25
  saved lineup_model: C:\Users\geric\Desktop\REPOS\propis\nba-fit-analysis\models\lineup_model\season=2024-25
lineup-sim projected_net_rating_delta: +0.00

$ C:\Users\geric\AppData\Local\Programs\Python\Python310\python.exe -m pytest tests/test_rapm.py tests/test_pbp_smoke.py -q
..........                                                               [100%]
10 passed in 2.90s

$ python visual_tests/09_possession_rate.py

$ python visual_tests/10_rapm_distribution.py

$ python visual_tests/11_lineup_delta_bar.py
```
