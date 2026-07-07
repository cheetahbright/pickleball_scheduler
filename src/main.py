#!/usr/bin/env python3
"""PICKLEBALL SCHEDULER V2 - MAIN LAUNCHER.

Main entry point for the Pickleball Scheduler application.
Simple, effective tournament scheduling with genetic optimization.

Usage:
    python main.py                    # Launch GUI
    python main.py --enhanced         # Launch legacy enhanced GUI (if available)
    python main.py --cli             # Command line interface
    python main.py --test            # Run validation tests
    python main.py --help            # Show this help

Features:
    • Genetic algorithm optimization
    • Player constraint support
    • Multiple export formats
    • Clean, intuitive interface
"""

import builtins
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

# Directory references used across launch modes
project_root = Path(__file__).parent.parent
src_path = project_root / "src"


class _FallbackSecurityError(Exception):
    """Fallback security error class."""


class _FallbackInputSanitizer:
    """Fallback input sanitizer class."""

    @staticmethod
    def sanitize_file_path(path: str, _base_dir: str | None = None) -> str:
        """Fallback file path sanitization."""
        return path


def _fallback_setup_logging(level: str = "INFO"):
    """Fallback logger factory."""
    import logging

    logger = logging.getLogger("pickleball_scheduler")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def _fallback_log_error_with_context(
    logger: Any,
    error: BaseException,
    message: str,
    context: dict[str, object] | None = None,
) -> None:
    """Fallback logging hook when the structured logger is unavailable."""
    _ = (logger, error, message, context)
    return None


SecurityError: type[Exception] = _FallbackSecurityError
InputSanitizer: Any = _FallbackInputSanitizer
setup_logging: Callable[..., Any] = _fallback_setup_logging
log_error_with_context: Callable[..., None] = _fallback_log_error_with_context
logger: Any | None = None

# Initialize logging and security modules
try:
    try:
        from src.security.input_sanitizer import InputSanitizer, SecurityError
    except ImportError:
        from security.input_sanitizer import InputSanitizer, SecurityError

    try:
        from src.utils.logging_config import log_error_with_context, setup_logging
    except ImportError:
        from utils.logging_config import log_error_with_context, setup_logging

    configured_logger = setup_logging(level="INFO")
    configured_logger.info("Pickleball Scheduler V2 starting", extra={"version": "2.0.0"})
    logger = configured_logger
except ImportError as e:
    # Fallback to print if logging/security setup fails
    builtins.print(f"Warning: Could not initialize logging/security: {e}")
    logger = None


def _console_print(message: object = "") -> None:
    """Print safely on Windows terminals that cannot encode emoji."""
    text = "" if message is None else str(message)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_text)


def _resolve_gui_path(enhanced: bool = False) -> tuple[Path, bool]:
    """Resolve the GUI entry point, falling back when legacy files drift."""
    standard_gui_path = project_root / "src" / "main_app.py"
    enhanced_gui_path = project_root / "src" / "gui" / "multi_algorithm_gui.py"

    if not enhanced:
        return standard_gui_path, False

    try:
        if enhanced_gui_path.exists() and enhanced_gui_path.stat().st_size > 0:
            return enhanced_gui_path, True
    except OSError:
        pass

    return standard_gui_path, False


def launch_gui(enhanced: bool = False) -> None:
    """Launch the Pickleball Scheduler GUI."""
    gui_path, using_enhanced_gui = _resolve_gui_path(enhanced=enhanced)

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

    if enhanced and not using_enhanced_gui:
        _console_print("ℹ️ Enhanced GUI entry point not available; launching standard GUI instead.")

    message = "🚀 Launching Pickleball Scheduler Application..."
    if using_enhanced_gui:
        message = "🚀 Launching Pickleball Scheduler Enhanced Application..."

    if logger:
        logger.info(
            "Launching GUI",
            extra={
                "port": port,
                "gui_path": str(gui_path),
                "enhanced_requested": enhanced,
                "enhanced_active": using_enhanced_gui,
            },
        )
    _console_print(message)
    _console_print(f"🌐 Opening at: http://localhost:{port}")

    # Launch streamlit with secure subprocess call
    try:
        safe_gui_path = InputSanitizer.sanitize_file_path(str(gui_path), str(project_root))
        if not isinstance(safe_gui_path, str) or not safe_gui_path:
            safe_gui_path = str(gui_path)

        subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                safe_gui_path,
                "--server.port",
                port,
            ],
            check=True,
            timeout=10,
        )
    except SecurityError as e:
        error_msg = f"🚫 Security error: {e}"
        if logger:
            log_error_with_context(
                logger,
                e,
                "Security error during GUI launch",
                {"port": port, "gui_path": str(gui_path)},
            )
        _console_print(error_msg)
        sys.exit(1)
    except subprocess.TimeoutExpired as e:
        error_msg = "⏰ Streamlit launch timed out"
        if logger:
            log_error_with_context(logger, e, "Streamlit launch timeout", {"port": port})
        _console_print(error_msg)
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ Failed to launch streamlit: {e}"
        if logger:
            log_error_with_context(
                logger,
                e,
                "Failed to launch streamlit",
                {"port": port, "return_code": e.returncode},
            )
        _console_print(error_msg)
        sys.exit(1)
    except OSError as e:
        error_msg = f"❌ Failed to launch GUI: {e}"
        if logger:
            log_error_with_context(logger, e, "GUI launch failed", {"port": port})
        _console_print(error_msg)
        sys.exit(1)
    except (RuntimeError, ValueError) as e:
        error_msg = f"❌ Unexpected error: {e}"
        if logger:
            log_error_with_context(logger, e, "Unexpected error during GUI launch", {"port": port})
        _console_print(error_msg)
        sys.exit(1)


def launch_cli() -> None:
    """Launch command line interface with genetic algorithm scheduler."""

    try:
        from src.algorithms.genetic_scheduler import GeneticPickleballScheduler
    except ImportError:
        from algorithms.genetic_scheduler import GeneticPickleballScheduler

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
                _console_print("  • Type 'e' or 'enhanced' for enhanced GUI")
                _console_print("  • Type 't' or 'test' to run tests")
                _console_print("  • Type 'q' or 'quit' to exit")
                _console_print("  • Minimum 4 players required\n")
                continue
            if player_name.lower() in ["g", "gui"]:
                _console_print("🚀 Launching GUI...")
                launch_gui(enhanced=False)
                return
            if player_name.lower() in ["e", "enhanced"]:
                _console_print("🚀 Launching enhanced GUI...")
                launch_gui(enhanced=True)
                return
            if player_name.lower() in ["t", "test"]:
                _console_print("🧪 Running tests...")
                run_tests()
                return
            # Check for single letter commands that might be invalid
            valid_commands = ["g", "e", "t", "h", "q"]
            is_single_char = len(player_name) == 1
            if is_single_char and player_name.lower() not in valid_commands:
                _console_print(f"❌ Unknown command '{player_name}'. Type 'h' for help.")
                continue
            player_names.append(player_name)

        if len(player_names) < 4:
            _console_print("❌ Need at least 4 players!")
            return

        # Get courts
        try:
            courts = int(input("\nNumber of courts (default 4): ") or "4")
        except ValueError:
            courts = 4

        # Get rounds
        try:
            rounds = int(input("Number of rounds (default 4): ") or "4")
        except ValueError:
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
                if hasattr(round_data, "get"):
                    round_games = round_data.get("games", [])
                else:
                    round_games = [round_data]

                for court, game in enumerate(round_games, 1):
                    team1 = " & ".join(game.team1)
                    team2 = " & ".join(game.team2)
                    _console_print(f"   Court {court}: {team1} vs {team2}")

            _console_print("\n✅ Schedule generation completed!")
        else:
            _console_print("\n� Could not generate a valid schedule")
            _console_print("Consider:")
            _console_print("• Reducing the number of rounds")
            _console_print("• Adding more players")
            _console_print("• Adjusting the number of courts")

    except KeyboardInterrupt:
        _console_print("\n\n👋 Goodbye!")
        if logger:
            logger.info("CLI operation cancelled by user")
        # Don't call sys.exit for KeyboardInterrupt - handle gracefully
        return
    except (ImportError, AttributeError, RuntimeError, Exception) as e:
        error_msg = f"❌ Scheduler failed with error: {e}"
        _console_print(error_msg)
        if logger:
            log_error_with_context(logger, e, error_msg, {"component": "cli_launcher"})
        sys.exit(1)


def run_tests() -> bool:
    """Run validation tests using pytest."""
    try:
        if logger:
            logger.info("Starting validation tests")

        _console_print("🧪 Running validation tests...")

        # Run the maintained default pytest lane from the project root.
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
            if logger:
                logger.info("All validation tests passed")
            return True
        else:
            _console_print(f"❌ Tests failed with return code {result.returncode}")
            if logger:
                logger.error(f"Tests failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        error_msg = "❌ Tests timed out after 5 minutes"
        _console_print(error_msg)
        if logger:
            logger.error("Test execution timed out")
        return False
    except Exception as e:
        error_msg = f"❌ Error running tests: {e}"
        _console_print(error_msg)
        if logger:
            log_error_with_context(logger, e, "Error running validation tests", {"operation": "run_tests"})
        return False


def main() -> None:
    """Main entry point with enhanced multi-algorithm support."""
    try:
        if logger:
            logger.info("Main entry point called", extra={"command_args": sys.argv})

        if len(sys.argv) > 1:
            arg = sys.argv[1]
            if arg == "--cli":
                launch_cli()
            elif arg == "--test":
                result = run_tests()
                sys.exit(0 if result else 1)
            elif arg == "--gui":
                launch_gui(enhanced=False)
            elif arg == "--enhanced":
                launch_gui(enhanced=True)
            elif arg == "--help":
                _console_print(__doc__)
                sys.exit(0)
            else:
                _console_print(f"Unknown argument: {arg}")
                _console_print("Available options:")
                _console_print("  --gui         Launch GUI interface")
                _console_print("  --enhanced    Launch legacy enhanced GUI interface")
                _console_print("  --cli         Command line interface")
                _console_print("  --test        Run validation tests")
                _console_print("  --help        Show detailed help")
                sys.exit(1)
        else:
            # Default: launch GUI
            _console_print("🎾 Pickleball Scheduler V2")
            _console_print("=" * 30)
            _console_print("Launching GUI...")
            launch_gui(enhanced=False)

    except (ImportError, OSError, RuntimeError) as e:
        if logger:
            log_error_with_context(logger, e, "Critical error in main function", {"operation": "main"})
        else:
            _console_print(f"❌ Critical error: {e}")
        sys.exit(1)
    except Exception as e:
        # Handle any other exceptions (like from GUI launch)
        if logger:
            log_error_with_context(logger, e, "Error in main function", {"operation": "main"})
        else:
            _console_print(f"❌ Error: {e}")
        # Don't exit for general exceptions, let program continue


def show_enhanced_help():
    """Show enhanced help message with detailed usage information"""
    help_text = """
🎾 PICKLEBALL SCHEDULER V2 - Enhanced Help
=========================================

USAGE:
    python main.py [OPTION]

OPTIONS:
    --enhanced    Launch legacy enhanced GUI when available, otherwise use standard GUI
    --cli         Enhanced command line interface for terminal users
    --test        Run validation tests and system checks
    --help        Show this detailed help message

FEATURES:
    • Genetic Algorithm optimization for fairness
    • Multi-algorithm comparison and selection
    • 10-second optimization budget system
    • Partner/opponent balance tracking
    • Schedule history and analytics
    • Configuration management

EXAMPLES:
    python main.py                    # Launch standard GUI
    python main.py --enhanced         # Launch enhanced GUI if available
    python main.py --cli             # Use command line interface
    python main.py --test            # Run system tests

For more information, see the README.md file.
    """
    _console_print(help_text)


if __name__ == "__main__":
    main()
