# Pickleball Scheduler v2 - Complete Application

## 🎯 **One Application, All Features You Need**

### 🎾 **Pickleball Scheduler - The Complete Solution**

- **Core Scheduling**: Reliable genetic algorithm for game generation, with mid-session replan for late arrivals/early leavers
- **Advanced Analytics**: Plotly visualizations, fairness metrics, and skill-balance breakdown
- **Configuration Management**: Weight tuning, constraints, player presets, export/import
- **History Tracking**: Database storage, weekly partner analytics, soft-delete with undo
- **Leaderboard & Ratings**: Score entry (with CSV bulk import), all-time win/loss standings, and ELO ratings
- **Player Management**: Substitutions, availability, custom presets, skill ratings
- **Mobile-Friendly & Installable**: Responsive layout, dark theme, and PWA support
- **Professional Quality**: Security, testing, and development standards

**Perfect for**: Individuals, clubs, leagues, tournaments - any pickleball group!

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

# Run the complete application
python -m streamlit run src/main_app.py

# Optional: protect the app with a password
# PowerShell
$env:PICKLEBALL_APP_PASSWORD = "choose-a-password"
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

# Install pre-commit hooks for security and quality
pre-commit install

# Run the default confidence suite
pytest -q

# Run security scan
bandit -r src/ -ll
python -m pip_audit -r requirements.txt

# Run the maintained browser smoke test
python -m pytest tests/organized/ui/test_main_ui_smoke.py -q
```

## 📋 **Features**

### **🎯 Main Scheduler**

- **Player Input**: Supports custom players and predefined presets
- **Smart Court Calculation**: Automatically determines optimal courts
- **Genetic Algorithm**: Proven scheduling algorithm for fair games
- **Real-time Generation**: Fast schedule creation with progress feedback
- **Export Options**: CSV, JSON, plain text, and Excel (.xlsx) download formats
- **Mid-Session Replan**: Regenerate only the not-yet-played rounds if the roster changes mid-session

### **📊 Analytics**

- **Fairness Metrics**: Games, partners, opponents, courts balance analysis
- **Interactive Charts**: Plotly radar charts and bar graphs
- **Player Statistics**: Detailed breakdown of individual player metrics
- **Quality Assessment**: Overall fairness scoring (0-10 scale)

### **📅 History**

- **Schedule Database**: SQLite storage for all generated schedules
- **Weekly Tracking**: Partner and opponent history analysis
- **Schedule Replay**: Load and view previous schedules
- **Soft-Delete with Undo**: Deleted schedules can be restored, not lost
- **Trend Analysis**: Identify patterns in player pairings

### **🏆 Leaderboard**

- **Score Entry**: Enter scores per game, or bulk-import via CSV
- **All-Time Standings**: Win/loss/win-rate aggregated across every recorded game
- **ELO Ratings**: Persistent skill ratings recomputed from recorded score history

### **⚙️ Configuration**

- **Objective Weights**: Tune importance of fairness vs variety vs constraints
- **Player Constraints**: "Do not pair" and "Do not oppose" rules
- **Custom Presets**: Save and manage player group configurations
- **Scheduling Preferences**: Default rounds, times, game duration
- **Export/Import**: Download or upload the full configuration as JSON

### **👥 Player Management**

- **Player Substitutions**: Handle early/late player scenarios
- **Availability Constraints**: Time-based player management
- **Preset Management**: Create, edit, delete custom player groups
- **Skill Ratings**: 1-5 per-player ratings feeding the Analytics skill-balance view
- **Input Validation**: Security-hardened player name processing

## 🔒 **Security & Quality**

### **Security Features**

- **Input Sanitization**: All user inputs validated and cleaned
- **SQL Injection Prevention**: Parameterized database queries
- **Secret Detection**: GitGuardian pre-commit hooks
- **Dependency Scanning**: Automated vulnerability checking

### **Development Standards**

- **Automated Testing**: Pytest with coverage reporting
- **Code Quality**: Black, flake8, mypy, isort
- **Pre-commit Hooks**: Security and quality enforcement
- **CI/CD Pipeline**: GitHub Actions for testing and security

### **Monitoring**

- **Error Handling**: Graceful failure with user-friendly messages
- **Performance**: Optimized for fast schedule generation
- **Logging**: Structured logging for debugging
- **Backup**: Automatic schedule history preservation

## 🏗️ **Architecture**

### **Application Structure**

```
src/
├── main_app.py              # Main Streamlit application facade
├── simple_auth.py           # Authentication system
├── theme_styles.py          # Dark theme CSS
├── mobile_styles.py         # Mobile-responsive CSS
├── pwa.py                   # PWA manifest/service-worker injection
├── rating_elo.py            # Pure ELO math
├── algorithms/
│   ├── genetic_scheduler.py # Core scheduling algorithm
│   └── constraint_model.py  # Constraint handling
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
└── player_elo.json          # Persistent ELO ratings
```

### **Key Classes**

- **ScheduleAnalytics**: Fairness calculations and visualizations
- **HistoryManager**: Database operations and persistence (schedules, scores, leaderboard)
- **ConfigurationManager**: Settings, constraints, and config export/import
- **SkillRatingManager / EloRatingManager**: Player skill and ELO ratings
- **PlayerManager**: Advanced player and substitution handling

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
- **Historical Insight**: Track patterns and improve future scheduling
- **Flexibility**: Handle substitutions, constraints, and special requirements
- **Professional Output**: Clean, printable schedules ready for game day

## 📞 **Support**

- **Documentation**: Comprehensive guides in `/docs` folder
- **Issues**: Report bugs via GitHub Issues
- **Security**: See `SECURITY.md` for vulnerability reporting
- **Contributing**: See `CONTRIBUTING.md` for development guidelines
- **Backlog**: See `docs/REPO_REVAMP_TODO.md` for the living revamp plan

---

**Built with ❤️ for the pickleball community**
