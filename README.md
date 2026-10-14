# Pickleball Scheduler v2 - Complete Application

## 🎯 **One Application, All Features You Need**

### 🎾 **Pickleball Scheduler - The Complete Solution**

- **Core Scheduling**: Instant optimal schedules where a fixed design fits, a genetic algorithm otherwise, with mid-session replan for late arrivals/early leavers
- **Advanced Analytics**: Plotly visualizations, fairness metrics, and skill-balance breakdown
- **Configuration Management**: Constraints, player presets, scheduling defaults, export/import
- **History Tracking**: Database storage, weekly partner history, soft-delete with undo
- **Leaderboard & Ratings**: Score entry (with CSV bulk import), all-time win/loss standings, and ELO ratings
- **Player Management**: Custom presets, skill ratings
- **Mobile-Friendly & Installable**: Responsive layout and PWA support
- **Professional Quality**: Security, testing, and development standards

**Perfect for**: Individuals, clubs, leagues, tournaments - any pickleball group!

> ⚠️ **The hosted app loses all its saved data on every reboot.** On Streamlit Community Cloud the schedule
> history, scores, ratings and settings live in local files that are not in the repo, so each reboot (which every
> deploy needs) and each restart by the platform itself deletes them. Download a full backup
> (**⚙️ Configuration → Backup / Restore → ⬇️ Download full backup**) before rebooting and after each session, and
> restore it after a reboot. See [Data Management](#data-management) and `docs/DEPLOY_SYNC_GUIDE.md`.

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
python -m pytest tests/organized/ui/test_main_ui_smoke.py tests/organized/ui/test_pwa_mobile_ui.py -q
```

## 📋 **Features**

### **🎯 Main Scheduler**

- **Player Input**: Supports custom players and predefined presets; duplicate names and case/spacing look-alikes are flagged before generation
- **Court Count**: Defaults to the most courts the roster can fill (4 players per court)
- **Hybrid Scheduling**: A provably fair fixed design, returned instantly, when every player plays every round on 2-5 courts and no do-not-pair/oppose rule applies to the roster; a genetic algorithm otherwise
- **Real-time Generation**: Fast schedule creation with progress feedback
- **Export Options**: CSV, JSON, plain text, and Excel (.xlsx) download formats
- **Mid-Session Replan**: Regenerate only the not-yet-played rounds if the roster changes mid-session

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

### **🏆 Leaderboard**

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
- **Objective Weights**: The sliders are saved with the configuration, but schedule generation doesn't use them yet; it always balances games first, then partners, courts and opponents

### **👥 Player Management**

- **Preset Management**: Create, edit, delete custom player groups
- **Skill Ratings**: Optional 1-5 per-player ratings feeding the Analytics skill-balance view
- **Late Arrivals and Early Leavers**: Use "Mid-session replan" on the Main Scheduler tab. The "Enable player substitutions" panel on this tab records first- and second-half player lists, but schedule generation doesn't use them yet

## 🔒 **Security & Quality**

### **Security Features**

- **Names Shown Literally**: Player and preset names are escaped before Streamlit renders them as Markdown, so a name can't add a link, an image or LaTeX
- **Spreadsheet-Safe Exports**: CSV, Excel and score-sheet exports neutralize cells that would start a formula
- **Input Limits**: Rosters are capped at 200 players; uploaded config files at 1 MB and 20 levels of nesting
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
- **Logging**: Structured logging for debugging
- **Backup**: Manual full backup and restore in the Configuration tab; nothing is backed up automatically

## 🏗️ **Architecture**

### **Application Structure**

```
src/
├── main_app.py              # Main Streamlit application facade
├── app_scheduler_flow.py    # Main Scheduler tab: roster, generation, mid-session replan
├── app_tabs.py              # Analytics, History, Configuration, Player Management,
│                             # Leaderboard and Stress Test tabs
├── app_schedule_helpers.py  # Schedule display and exports
├── app_analytics.py         # Plotly charts
├── app_stress_test.py       # The Stress Test tab's sweep over player, round and constraint combinations
├── app_managers.py          # Compatibility facade that re-exports managers/
├── simple_auth.py           # Optional shared-password gate
├── mobile_styles.py         # Mobile-responsive CSS
├── pwa.py                   # PWA manifest/service-worker injection
├── rating_elo.py            # Pure ELO math
├── algorithms/
│   ├── constructive_scheduler.py # Instant fixed designs (constructive_designs.py)
│   ├── genetic_scheduler.py # Genetic algorithm, with do-not-pair/oppose constraints
│   ├── scheduler_*.py       # GA evolution loop, repair passes, metrics and reporting
│   ├── arrangement_generator.py # Round arrangements the GA searches over
│   ├── cpsat_scheduler.py   # Exact OR-Tools solver, benchmark only
│   └── constraint_model.py  # Scoring model for scripts/algorithm_benchmark.py; not used by the app
├── managers/                # HistoryManager, ConfigurationManager,
│                             # SkillRatingManager, EloRatingManager, PlayerManager
├── utils/                   # Analytics, feasibility, and helper utilities
├── gui/                     # Compatibility entry points
└── main.py                  # CLI and launcher
```

### **Data Management**

```
data/
├── schedule_history.db      # SQLite database for schedules and scores
├── app_config.json          # Application configuration
├── default_player_names.json # Default player presets
├── player_skills.json       # Manually-set 1-5 skill ratings
└── player_elo.json          # ELO ratings recomputed from recorded scores
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

### **Key Classes**

- **ScheduleAnalytics**: Fairness calculations and visualizations
- **HistoryManager**: Database operations and persistence (schedules, scores, leaderboard)
- **ConfigurationManager**: Settings, constraints, and config export/import
- **SkillRatingManager / EloRatingManager**: Player skill and ELO ratings
- **PlayerManager**: The Player Management tab's substitution panel (not yet used by generation)

## 📊 **What Makes This Different**

### **Balanced Approach**

- **Not too simple**: More than basic scheduling - includes analytics and history
- **Not too complex**: No enterprise bloat - just useful features
- **Just right**: Perfect balance for real-world pickleball groups

### **User-Focused**

- **Intuitive Interface**: Clean, tabbed layout with logical flow
- **Helpful Feedback**: Clear error messages and success indicators
- **Export Ready**: Professional CSV output for printing or sharing

### **Developer-Friendly**

- **Clean Code**: Well-documented, modular, testable
- **Security First**: Built-in protection against common vulnerabilities
- **Maintainable**: Professional development practices throughout

## 🎾 **Perfect for Pickleball**

Whether you're organizing games for a few friends or managing a club with dozens of players, this scheduler provides:

- **Fair Game Distribution**: Advanced algorithms ensure everyone plays equally
- **Partner Variety**: Smart pairing prevents the same people always playing together
- **Historical Insight**: See who partnered whom in recent weeks
- **Flexibility**: Handle late arrivals and early leavers, and do-not-pair/oppose constraints
- **Professional Output**: Clean, printable schedules ready for game day

## 📞 **Support**

- **Documentation**: Comprehensive guides in `/docs` folder
- **Issues**: Report bugs via GitHub Issues
- **Security**: See `SECURITY.md` for vulnerability reporting
- **Contributing**: See `CONTRIBUTING.md` for development guidelines
- **Backlog**: See `docs/REPO_REVAMP_TODO.md` for the living revamp plan

---

**Built with ❤️ for the pickleball community**
