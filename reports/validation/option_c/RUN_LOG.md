# Option C validation run

- Season: `2024-25`
- Started: `2026-05-28T17:17:39.870847+00:00`
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
  saved: C:\Users\geric\Desktop\REPOS\propis\nba-fit-option-c-wt\models\role_embedding\season=2024-25
Skipping impact ingest — possessions cached for 2024-25

$ C:\Users\geric\AppData\Local\Programs\Python\Python310\python.exe -m nba_fit train-impact --season 2024-25
Training Option C impact models season=2024-25...
  rapm players: 505 (505 low-sample flags)
  net rapm range: -0.24 .. 0.30
  lineup model penalty: ridge
  saved rapm: C:\Users\geric\Desktop\REPOS\propis\nba-fit-option-c-wt\models\rapm\season=2024-25
  saved lineup_model: C:\Users\geric\Desktop\REPOS\propis\nba-fit-option-c-wt\models\lineup_model\season=2024-25
lineup-sim projected_net_rating_delta: +0.30

$ C:\Users\geric\AppData\Local\Programs\Python\Python310\python.exe -m pytest tests/test_rapm.py tests/test_pbp_smoke.py -q
..........                                                               [100%]
10 passed in 9.79s

$ python visual_tests/09_possession_rate.py

$ python visual_tests/10_rapm_distribution.py

$ python visual_tests/11_lineup_delta_bar.py
```
