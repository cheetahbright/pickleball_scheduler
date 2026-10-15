# Pickleball Scheduler v2 - Complete Application

## 🎯 **One Application, All Features You Need**

### 🎾 **Pickleball Scheduler - The Complete Solution**

- **Core Scheduling**: Instant optimal schedules where a fixed design fits, a genetic algorithm otherwise, with mid-session replan for late arrivals/early leavers
- **Advanced Analytics**: Plotly visualizations, fairness metrics, and skill-balance breakdown
- **Configuration Management**: Constraints, player presets, scheduling defaults, export/import
- **History Tracking**: Database storage, weekly partner history, soft-delete with undo
- **Leaderboard & Ratings** (off by default): Score entry (with CSV bulk import), all-time win/loss standings, and ELO ratings. Hidden until you turn on score tracking: see the Leaderboard section below
- **Player Management**: Custom presets, skill ratings
- **Mobile-Friendly & Installable**: Responsive layout and PWA support. On Streamlit Community Cloud the host page carries its own manifest and wraps the app inside it, so the browser's automatic "Install app" prompt (if any) is the host's, not the app's: install from the browser menu ("Add to Home Screen" / "Install app") instead. Running it yourself (`python -m streamlit run src/main_app.py`) gives the real prompt.

> ⚠️ **The hosted app loses all its saved data on every reboot.** On Streamlit Community Cloud the schedule
> history, scores, ratings and settings live in local files that are not in the repo, so each reboot (which every
> deploy needs) and each restart by the platform itself deletes them. Download a full backup
> (**⚙️ Configuration → Backup / Restore → ⬇️ Download full backup**) before rebooting and after each session, and
> restore it after a reboot. See [Data Management](#data-management) and `docs/DEPLOY_SYNC_GUIDE.md`. An optional,
> encrypted cloud copy can restore it for you: see `docs/REMOTE_BACKUP.md`.

## 🚀 **Quick Start**

### **Run the Application**

```bash
# Create a clean local environment
python -m venv .venv

# Activate it
# PowerShell
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Optional: protect the app with a password. Set it before starting the app
# (or put app_password = "..." in .streamlit/secrets.toml instead).
# PowerShell
$env:PICKLEBALL_APP_PASSWORD = "choose-a-password"

# Run the complete application (keeps running until you stop it)
python -m streamlit run src/main_app.py
```

### **For Developers**

```bash
# Clone repository
git clone https://github.com/cheetahbright/pickleball_scheduler_v2.git
cd pickleball_scheduler_v2

# Create and activate a clean project-local environment
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1

# Install development dependencies
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

# Install pre-commit hooks: CI's black, isort, flake8, mypy and Bandit checks
# on the same files (the ggshield secret scan is opt-in; see SECURITY.md)
pre-commit install

# Run the default confidence suite
pytest -q

# Run security scan
bandit -r src/ -ll
python -m pip_audit -r requirements.txt

# Run the maintained browser smoke tests (as CI's ui-smoke job does). They need Playwright and Chromium,
# and no app password configured (see the Pre-Sync Checklist in docs/DEPLOY_SYNC_GUIDE.md).
python -m pytest tests/organized/ui/test_main_ui_smoke.py tests/organized/ui/test_pwa_mobile_ui.py tests/organized/ui/test_attendance_ui.py -q
```

## 📋 **Features**

### **🎯 Main Scheduler**

- **Player Input**: Supports custom players and predefined presets; duplicate names and case/spacing look-alikes are flagged before generation
- **Court Count**: Defaults to the most courts the roster can fill (4 players per court)
- **Hybrid Scheduling**: A provably fair fixed design, returned instantly, when every player plays every round on 2-5 courts and no do-not-pair/oppose rule applies to the roster; a genetic algorithm otherwise
- **Real-time Generation**: Fast schedule creation with progress feedback
- **Balance Summary**: The fairness score, the games, partners, opponents and courts ranges, and "Partner Repeats" - the most times any one pair partners, which the ranges (they count different partners) can't show. Whenever a pair partners more often than an even spread of the games needs, the balance message says so, whatever its outcome
- **Export Options**: CSV, JSON, plain text, and Excel (.xlsx) download formats
- **Arrivals and Departures**: List who arrives after round 1 or leaves before the last round, and each round is scheduled with only the people there (fewer courts when too few are present). The fairness estimates for one fixed roster give way to a who-was-here-when table; such a schedule can't be replayed from a seed
- **Mid-Session Replan**: Regenerate only the not-yet-played rounds if the roster changes mid-session; the new rounds even out who has sat out so far (counting the rounds already played), but don't know who partnered or opposed whom in them

### **📊 Analytics**

- **Fairness Metrics**: Games, partners, opponents, courts balance analysis
- **Interactive Charts**: Plotly bar chart of per-player games, partners and opponents, plus partner/opponent heatmaps
- **Player Statistics**: Detailed breakdown of individual player metrics
- **Quality Assessment**: Overall fairness scoring (0-10 scale)

### **📅 History**

- **Schedule Database**: SQLite storage for all generated schedules
- **Weekly Partner History**: Who partnered whom in each of the four most recent weeks
- **Schedule Replay**: Load and view previous schedules
- **Soft-Delete with Undo**: Deleted schedules can be restored, not lost

### **🏆 Leaderboard (off by default)**

Win/loss score tracking is switched off, so the app shows no Leaderboard tab, score boxes, standings or ELO ratings,
and the Excel download has no Scores sheet. The code and any scores already saved stay in place. To turn it on, set
the environment variable `PICKLEBALL_SCORE_TRACKING=1` (`true` and `yes` also work) before starting the app, for
example in PowerShell `$env:PICKLEBALL_SCORE_TRACKING = "1"`. On Streamlit Cloud, add it under the app's Secrets as
`PICKLEBALL_SCORE_TRACKING = "1"` (in quotes) and reboot the app (download a full backup first: see Data Management
below). Press **Recompute ELO ratings** on the Leaderboard tab afterwards: ratings aren't kept up to date while it is
off. See "Feature flags" in `docs/ARCHITECTURE.md`.

- **Score Entry**: Enter scores per game, or bulk-import via CSV
- **All-Time Standings**: Win/loss/win-rate aggregated across every recorded game
- **ELO Ratings**: Skill ratings recomputed from the recorded score history

### **⚙️ Configuration**

- **Player Constraints**: "Do not pair" and "Do not oppose" rules
- **Custom Presets**: Save and manage player group configurations
- **Scheduling Preferences**: Default rounds, start and end time
- **Export/Import**: Download or upload the full configuration as JSON
- **Full Backup / Restore**: Download everything the app saves (history, scores, configuration, skill and ELO
  ratings) as one file, and restore it, after a confirmation, with a copy of the replaced data kept
- **Cloud Backup** (optional): The same full backup, encrypted in the app and kept in a free hosted Postgres
  database (Neon), sent shortly after the saved data changes and restored automatically when the app starts with no
  data. Off unless configured; see `docs/REMOTE_BACKUP.md`
- **Objective Weights**: Hidden by default. The weights stay saved with the configuration, but schedule generation doesn't use them; it always balances games first, then partners, courts and opponents. Set `PICKLEBALL_OBJECTIVE_WEIGHTS_UI=1` to show the sliders again (see "Feature flags" in `docs/ARCHITECTURE.md`)

### **👥 Player Management**

- **Preset Management**: Create, edit, delete custom player groups
- **Skill Ratings**: Optional 1-5 per-player ratings feeding the Analytics skill-balance view
- **Late Arrivals and Early Leavers**: List them under "Arrivals and departures" on the Main Scheduler tab before generating, and each round is scheduled with only the people who are there. Use "Mid-session replan" there for changes made during play: it gives every remaining round the one list of players it starts with (the players the plan had in the first of those rounds), so arrivals and departures planned for later rounds are not kept - replan again when someone arrives or leaves

## 🔒 **Security & Quality**

### **Security Features**

- **Names Shown Literally**: Player and preset names are escaped before Streamlit renders them as Markdown, so a name can't add a link, an image or LaTeX
- **Spreadsheet-Safe Exports**: CSV, Excel and score-sheet exports neutralize cells that would start a formula
- **Input Limits**: Rosters are capped at 72 players; uploaded config files at 1 MB and 20 levels of nesting
- **SQL Injection Prevention**: Parameterized database queries
- **Optional Password**: One shared password gates the whole app, with failed-login rate limiting
- **Dependency Scanning**: CI fails when `pip-audit` finds a known vulnerability in `requirements.txt`, and Dependabot opens weekly update PRs

See `SECURITY.md` for what is and isn't enforced.

### **Development Standards**

- **Automated Testing**: Pytest in CI with a coverage floor, plus a Playwright browser smoke test
- **Code Quality**: CI checks black, isort, flake8 and mypy
- **Security Scan**: CI runs Bandit on `src/`, and a Medium or High finding fails the build
- **Pre-commit Hooks**: Optional and local; nothing enforces them
- **Secret Detection**: Nothing scans for secrets automatically. The pre-commit config includes GitGuardian's `ggshield`, which needs a GitGuardian API key
- **CI Pipeline**: GitHub Actions on pushes and pull requests (no automated deploy; see `docs/DEPLOY_SYNC_GUIDE.md`)

### **Monitoring**

- **Error Handling**: Graceful failure with user-friendly messages
- **Performance**: Optimized for fast schedule generation
- **Logging**: Console logging (level, logger name, message) for debugging
- **Backup**: Manual full backup and restore in the Configuration tab; nothing is backed up automatically unless
  the optional encrypted cloud copy is set up (`docs/REMOTE_BACKUP.md`)

## 🏗️ **Architecture**

### **Application Structure**

See `docs/ARCHITECTURE.md` for the layout of `src/` (entry points, UI layer, algorithms, managers, utilities)
and how the pieces fit together.

### **Data Management**

```
data/
├── schedule_history.db      # SQLite database for schedules and scores
├── app_config.json          # Application configuration
├── default_player_names.json # Default player presets
├── player_skills.json       # Manually-set 1-5 skill ratings
└── player_elo.json          # ELO ratings recomputed from recorded scores (recomputed only with score tracking on; a restore writes it either way)
```

All of these except `default_player_names.json` are local, untracked state. **A reboot of the hosted app deletes
them**: Streamlit Community Cloud starts a reboot from a fresh clone of the repo, and can also restart the app on its
own (for example when it wakes from sleep) with the same result. To carry the data across:

1. Before the reboot (and after each session, in case the platform restarts the app), open
   **⚙️ Configuration → Backup / Restore** and click **⬇️ Download full backup**. The ZIP file holds a consistent
   snapshot of `schedule_history.db` plus `app_config.json`, `player_skills.json` and `player_elo.json`. The download
   checks the file as a restore would, so a backup that restore would refuse fails here instead ("No full backup
   was made"), while the data still exists: don't reboot until a download works.
2. After the reboot, choose that file under **⬆️ Restore full backup**, check the summary, tick the confirmation and
   click **♻️ Restore full backup**. The restore checks the whole file first, keeps a copy of the data it replaces
   (`full_backup_before_restore_*.zip`, offered for download right there), and puts everything back if any step fails.

The restore accepts a file up to the server's upload limit (`server.maxUploadSize` in `.streamlit/config.toml`,
1 MB). Each saved schedule adds about 1 to 3 KB to a full backup (most when it has scores). Every **Generate
Schedule** click saves one, and deleted schedules stay in the database too, so a weekly 16-player group that
generates three times per session adds about 250 KB a year, and the limit lasts about three years. If a backup ever
grows past it, raise `server.maxUploadSize` before the reboot that needs the restore.

**Optional cloud copy.** With a `[remote_backup]` table in the app's secrets (a Neon database URL and an encryption
key; `docs/REMOTE_BACKUP.md` has the setup), the app also keeps this same verified backup, encrypted with AES-256-GCM
before it leaves the app, in that database. It is sent in the background shortly after the saved data changes, and
when the app starts with no data (no schedule, no `app_config.json`, no ratings) the newest copy is restored by itself,
through the same checked restore. The local files stay the source of truth, a cloud outage never breaks the app, and the manual
download and restore above remain the fallback: keep downloading a full backup now and then, since the free plan has
no uptime promise and a lost key makes the cloud copies unreadable.

### **Key Classes**

- **ScheduleAnalytics**: Fairness calculations and visualizations
- **HistoryManager**: Database operations and persistence (schedules, scores, leaderboard)
- **ConfigurationManager**: Settings, constraints, and config export/import
- **SkillRatingManager / EloRatingManager**: Player skill and ELO ratings

## 📞 **Support**

- **Issues**: Report bugs via GitHub Issues
- **Security**: See `SECURITY.md` for vulnerability reporting
- **Contributing**: See `CONTRIBUTING.md` for development guidelines

---

**Built with ❤️ for the pickleball community**
