#!/usr/bin/env python3
"""PICKLEBALL SCHEDULER V2 - MAIN LAUNCHER.

Main entry point for the Pickleball Scheduler application.
Simple, effective tournament scheduling with genetic optimization.

Usage:
    python main.py                    # Launch GUI
    python main.py --cli             # Command line interface
    python main.py --test            # Run validation tests
    python main.py --help            # Show help and exit

Features:
    • Genetic algorithm optimization
    • Player constraint support
    • Multiple export formats
    • Clean, intuitive interface
"""

import os
import subprocess
import sys
from pathlib import Path

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

project_root = Path(__file__).parent.parent

_logging_config = import_module_with_fallback("utils.logging_config")
log_error_with_context = _logging_config.log_error_with_context
logger = _logging_config.setup_logging(level="INFO")
logger.info("Pickleball Scheduler V2 starting", extra={"version": "2.0.0"})


def _console_print(message: object = "") -> None:
    """Print safely on Windows terminals that cannot encode emoji."""
    text = "" if message is None else str(message)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_text)


def _handle_error(
    error: BaseException, user_msg: str, context: dict | None = None, exit_code: int | None = None
) -> None:
    """Print error to console and log it with context."""
    _console_print(user_msg)
    log_error_with_context(logger, error, user_msg, context or {})
    if exit_code is not None:
        sys.exit(exit_code)


def launch_gui() -> None:
    """Launch the Pickleball Scheduler GUI."""
    gui_path = project_root / "src" / "main_app.py"

    # Get port from environment with validation and fallback
    env_port = os.environ.get("PORT", "8501")
    port = "8501"  # Default port

    # Validate environment port, fall back to default if invalid
    try:
        if env_port.isdigit() and 1024 <= int(env_port) <= 65535:
            port = env_port
        else:
            _console_print(f"⚠️ Invalid PORT '{env_port}', using default 8501")
    except (ValueError, AttributeError):
        _console_print(f"⚠️ Invalid PORT '{env_port}', using default 8501")

    logger.info("Launching GUI", extra={"port": port, "gui_path": str(gui_path)})
    _console_print("🚀 Launching Pickleball Scheduler Application...")
    _console_print(f"🌐 Opening at: http://localhost:{port}")

    try:
        # The server runs until it is stopped, so no timeout: it ends with Ctrl+C.
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "streamlit", "run", str(gui_path), "--server.port", port],
            check=True,
        )
    except KeyboardInterrupt:
        # Ctrl+C reaches Streamlit as well, which stops the server.
        _console_print("\n👋 Pickleball Scheduler stopped.")
    except subprocess.CalledProcessError as e:
        _handle_error(e, f"❌ Failed to launch streamlit: {e}", {"port": port, "return_code": e.returncode}, 1)
    except OSError as e:
        _handle_error(e, f"❌ Failed to launch GUI: {e}", {"port": port}, 1)
    except (RuntimeError, ValueError) as e:
        _handle_error(e, f"❌ Unexpected error: {e}", {"port": port}, 1)


def launch_cli() -> None:
    """Launch command line interface with genetic algorithm scheduler."""

    try:
        from src.algorithms.genetic_scheduler import GeneticPickleballScheduler
    except ImportError:
        from algorithms.genetic_scheduler import GeneticPickleballScheduler

    try:
        from src.utils.schedule_shapes import games_in_round
    except ImportError:
        from utils.schedule_shapes import games_in_round

    try:
        from src.app_scheduler_flow import find_player_name_issues
    except ImportError:
        from app_scheduler_flow import find_player_name_issues

    try:
        _console_print("🏓 Pickleball Scheduler - Enhanced Command Line Interface")
        _console_print("=" * 60)

        # Get player names
        _console_print("\nEnter player names (4+ required, Enter=done, 'q'=quit):")
        player_names: list[str] = []
        while True:
            try:
                player_name = input(f"Player {len(player_names)+1}: ").strip()
            except (EOFError, OSError):
                # Handle cases where input is not available (e.g., during testing)
                _console_print("❌ Input not available - using test mode")
                break

            if not player_name:
                break
            if player_name.lower() in ["q", "quit", "exit"]:
                _console_print("👋 Goodbye!")
                return
            if player_name.lower() in ["h", "help"]:
                _console_print("\n📖 Available commands:")
                _console_print("  • Enter player names one by one")
                _console_print("  • Press Enter when done adding players")
                _console_print("  • Type 'h' or 'help' for this help message")
                _console_print("  • Type 'g' or 'gui' to launch GUI")
                _console_print("  • Type 't' or 'test' to run tests")
                _console_print("  • Type 'q' or 'quit' to exit")
                _console_print("  • Minimum 4 players required\n")
                continue
            if player_name.lower() in ["g", "gui"]:
                _console_print("🚀 Launching GUI...")
                launch_gui()
                return
            if player_name.lower() in ["t", "test"]:
                _console_print("🧪 Running tests...")
                run_tests()
                return
            # Check for single letter commands that might be invalid
            valid_commands = ["g", "t", "h", "q"]
            is_single_char = len(player_name) == 1
            if is_single_char and player_name.lower() not in valid_commands:
                _console_print(f"❌ Unknown command '{player_name}'. Type 'h' for help.")
                continue
            # The same checks as the app: one person can't take two places in a round.
            name_issues = find_player_name_issues([*player_names, player_name])
            if name_issues:
                for issue in name_issues:
                    _console_print(f"❌ {issue} - enter a different name.")
                continue
            player_names.append(player_name)

        if len(player_names) < 4:
            _console_print("❌ Need at least 4 players!")
            return

        # Get courts - each needs 4 players, so more than players // 4 leaves no valid round
        max_courts = max(1, len(player_names) // 4)
        try:
            courts = int(input(f"\nNumber of courts (default {max_courts}): ") or str(max_courts))
        except ValueError:
            courts = max_courts
        if not 1 <= courts <= max_courts:
            _console_print(f"⚠️ {len(player_names)} players can fill 1 to {max_courts} court(s) - using {max_courts}.")
            courts = max_courts

        # Get rounds
        try:
            rounds = int(input("Number of rounds (default 4): ") or "4")
        except ValueError:
            rounds = 4
        if rounds < 1:
            _console_print("⚠️ At least 1 round is needed - using 4.")
            rounds = 4

        # Generate schedule using genetic algorithm
        _console_print("\n🚀 GENETIC ALGORITHM SCHEDULING")
        _console_print("=" * 50)
        _console_print(f"Players: {len(player_names)} | Courts: {courts} | Rounds: {rounds}")

        scheduler = GeneticPickleballScheduler(players=player_names, num_courts=courts, num_rounds=rounds)

        _console_print("\nGenerating schedule (30 second timeout)...")
        result = scheduler.generate_schedule(max_time=30.0)

        if isinstance(result, dict) and result.get("schedule"):
            schedule = result["schedule"]
            _console_print("\n📋 SCHEDULE GENERATED")
            _console_print("=" * 60)

            for round_num, round_data in enumerate(schedule, 1):
                _console_print(f"\n🎾 Round {round_num}:")
                round_games = games_in_round(round_data)

                for court, game in enumerate(round_games, 1):
                    team1 = " & ".join(game.team1)
                    team2 = " & ".join(game.team2)
                    _console_print(f"   Court {court}: {team1} vs {team2}")

            _console_print("\n✅ Schedule generation completed!")
        else:
            _console_print("\n❌ Could not generate a valid schedule")
            _console_print("Consider:")
            _console_print("• Reducing the number of rounds")
            _console_print("• Adding more players")
            _console_print("• Adjusting the number of courts")

    except KeyboardInterrupt:
        _console_print("\n\n👋 Goodbye!")
        logger.info("CLI operation cancelled by user")
    except (ImportError, AttributeError, RuntimeError, Exception) as e:
        _handle_error(e, f"❌ Scheduler failed with error: {e}", {"component": "cli_launcher"}, 1)


def run_tests() -> bool:
    """Run validation tests using pytest."""
    logger.info("Starting validation tests")

    _console_print("🧪 Running validation tests...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            check=False,
            timeout=300,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        if result.returncode == 0:
            _console_print("✅ All tests passed!")
            logger.info("All validation tests passed")
            return True

        _console_print(f"❌ Tests failed with return code {result.returncode}")
        logger.error(f"Tests failed with return code {result.returncode}")
        return False

    except subprocess.TimeoutExpired as e:
        _handle_error(e, "❌ Tests timed out after 5 minutes")
        return False
    except Exception as e:
        _handle_error(e, f"❌ Error running tests: {e}", {"operation": "run_tests"})
        return False


def main() -> None:
    """Main entry point."""
    logger.info("Main entry point called", extra={"command_args": sys.argv})

    try:
        if len(sys.argv) > 1:
            arg = sys.argv[1]
            if arg == "--cli":
                launch_cli()
            elif arg == "--test":
                sys.exit(0 if run_tests() else 1)
            elif arg == "--gui":
                launch_gui()
            elif arg == "--help":
                _console_print(__doc__)
                sys.exit(0)
            else:
                _console_print(f"Unknown argument: {arg}")
                _console_print("Available options: --gui, --cli, --test, --help")
                sys.exit(1)
        else:
            _console_print("🎾 Pickleball Scheduler V2\n" + "=" * 30 + "\nLaunching GUI...")
            launch_gui()

    except (ImportError, OSError, RuntimeError) as e:
        _handle_error(e, f"❌ Critical error: {e}", {"operation": "main"}, 1)
    except Exception as e:
        _handle_error(e, f"❌ Error: {e}", {"operation": "main"})


if __name__ == "__main__":
    main()
