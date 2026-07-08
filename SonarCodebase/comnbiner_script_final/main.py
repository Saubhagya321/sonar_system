"""
main.py - CLI Entry Point for Recommendation Combiner System
============================================================

Usage:
    python main.py gridsearch    # Run monthly optimization
    python main.py quickstart    # Show setup checklist
    python main.py health        # Check system health
    python main.py --help        # Show help

NOTE: For production API, use routes.py with gunicorn:
    gunicorn --workers 4 --bind 0.0.0.0:8001 routes:app
"""

import sys
import argparse
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loaders import DataLoader
from grid_search import run_monthly_optimization
from config import setup_logging, DATA_DIR, WEIGHTS_DIR

setup_logging()
logger = logging.getLogger(__name__)


def display_header(title: str) -> None:
    """Display formatted header."""
    logger.info("\n" + "=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70 + "\n")


def cmd_gridsearch() -> int:
    """Run grid search optimization."""
    display_header("RUNNING GRID SEARCH (MONTHLY OPTIMIZATION)")
    try:
        run_monthly_optimization()
        logger.info("Grid search completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Grid search failed: {e}")
        return 1


def cmd_quickstart() -> int:
    """Show quick start checklist."""
    display_header("QUICK START CHECKLIST")
    
    from config import GRID_SEARCH_PATHS
    
    checks = []
    
    # Check data directory
    if DATA_DIR.exists():
        checks.append(("✓", "Data directory exists", str(DATA_DIR)))
    else:
        checks.append(("✗", "Data directory missing", str(DATA_DIR)))
        DATA_DIR.mkdir(exist_ok=True, parents=True)
    
    # Check weights directory
    if WEIGHTS_DIR.exists():
        checks.append(("✓", "Weights directory exists", str(WEIGHTS_DIR)))
    else:
        checks.append(("✗", "Weights directory missing", str(WEIGHTS_DIR)))
        WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Check grid search data
    gs_lightgcn = GRID_SEARCH_PATHS.get('lightgcn')
    if gs_lightgcn and gs_lightgcn.exists():
        checks.append(("✓", "LightGCN data found", str(gs_lightgcn)))
    else:
        checks.append(("✗", "LightGCN data missing", str(gs_lightgcn)))
    
    gs_collab = GRID_SEARCH_PATHS.get('collaborative')
    if gs_collab and gs_collab.exists():
        checks.append(("✓", "Collaborative data found", str(gs_collab)))
    else:
        checks.append(("✗", "Collaborative data missing", str(gs_collab)))
    
    # Print results
    for status, message, path in checks:
        logger.info(f"  {status} {message}")
        logger.info(f"      {path}")
    
    logger.info("\n" + "-" * 70)
    logger.info("For production API, run:")
    logger.info("  gunicorn --workers 4 --bind 0.0.0.0:8001 routes:app")
    logger.info("-" * 70)
    
    return 0


def cmd_health() -> int:
    """Check system health."""
    display_header("SYSTEM HEALTH CHECK")
    
    from sqlalchemy import create_engine, text
    from config import DB_CONNECTION_STRING
    import requests
    import os
    
    results = {}
    
    # Check database
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        results['database'] = True
        logger.info("  ✓ Database: Connected")
        engine.dispose()
    except Exception as e:
        results['database'] = False
        logger.error(f"  ✗ Database: {e}")
    
    # Check external APIs
    apis = {
        'lightgcn': os.environ.get('LIGHTGCN_API', 'http://localhost:8002'),
        'collaborative': os.environ.get('COLLABORATIVE_API', 'http://localhost:8003'),
        'user_content': os.environ.get('USER_CONTENT_API', 'http://localhost:8004'),
        'item_content': os.environ.get('ITEM_CONTENT_API', 'http://localhost:8005'),
    }
    
    for name, url in apis.items():
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                results[name] = True
                logger.info(f"  ✓ {name}: Healthy")
            else:
                results[name] = False
                logger.warning(f"  ✗ {name}: Status {resp.status_code}")
        except requests.exceptions.RequestException as e:
            results[name] = False
            logger.warning(f"  ✗ {name}: Unreachable ({type(e).__name__})")
    
    # Summary
    healthy_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"\n  Summary: {healthy_count}/{total_count} services healthy")
    
    return 0 if all(results.values()) else 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Recommendation Combiner System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py gridsearch    Run monthly weight optimization
  python main.py quickstart    Show setup checklist
  python main.py health        Check all services health

For production API:
  gunicorn --workers 4 --bind 0.0.0.0:8001 routes:app
        """
    )
    
    parser.add_argument(
        'command',
        choices=['gridsearch', 'quickstart', 'health'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    commands = {
        'gridsearch': cmd_gridsearch,
        'quickstart': cmd_quickstart,
        'health': cmd_health,
    }
    
    return commands[args.command]()


if __name__ == "__main__":
    sys.exit(main())