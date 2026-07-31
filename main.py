"""Compatibility entry point; prefer ``python -m gridworld_rl``."""

from gridworld_rl.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
