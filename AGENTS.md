# Repository Workflow & Fork Rules

## Branching & Merging Policy

- **`main`** is our working base and primary target for all feature PRs, bugfixes, and internal worktrees. Point all pull requests to `main`.
- **`upstream`** tracks the fork's source repository (`https://github.com/agentscope-ai/TuFT.git`, `main` branch).
- Keep `upstream` clean and synced directly with upstream releases/commits. Rebase or merge `upstream` into `main` deliberately.

## Agent Guidelines & Standards

- Follow the `/ponytail` ladder on every task: force minimal working changes, stdlib/native/in-repo reuse first, shortest diffs, no unnecessary abstractions.
- Non-trivial features or structural changes must have a design plan reviewed and approved via `/grill-me` before code implementation.
- Execute feature implementations in isolated Git worktrees.
