#!/usr/bin/env python3
"""
Branch Protection Setup Script for HELPDESK.AI
Configures branch protection rules using GitHub CLI (`gh`).
Supports configurable target branch, repository, and admin enforcement.
"""

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, Optional

DEFAULT_REPO = "riteshbonthalakoti/HELPDESK.AI"
DEFAULT_BRANCH = "gssoc"


def build_protection_payload(
    enforce_admins: bool = True,
    required_approving_review_count: int = 1,
    dismiss_stale_reviews: bool = True,
    require_code_owner_reviews: bool = True,
    strict_status_checks: bool = True,
    contexts: Optional[list] = None,
) -> Dict[str, Any]:
    """Constructs the GitHub branch protection REST API JSON payload."""
    return {
        "required_status_checks": {
            "strict": strict_status_checks,
            "contexts": contexts or [],
        },
        "enforce_admins": enforce_admins,
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": dismiss_stale_reviews,
            "require_code_owner_reviews": require_code_owner_reviews,
            "required_approving_review_count": required_approving_review_count,
        },
        "restrictions": None,
    }


def setup_branch_protection(
    repo: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
    enforce_admins: bool = True,
    required_reviews: int = 1,
    dry_run: bool = False,
) -> int:
    """Applies branch protection rules for the specified repository and branch.

    Returns:
        0 on success or dry-run, non-zero exit code on failure.
    """
    payload = build_protection_payload(
        enforce_admins=enforce_admins,
        required_approving_review_count=required_reviews,
    )

    endpoint = f"/repos/{repo}/branches/{branch}/protection"
    print(f"Setting up branch protection for {repo} on branch '{branch}' (enforce_admins={enforce_admins})...")

    if dry_run:
        print("[DRY RUN] Would execute: gh api -X PUT " + endpoint)
        print("[DRY RUN] Payload:")
        print(json.dumps(payload, indent=2))
        return 0

    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                "-X",
                "PUT",
                endpoint,
                "--input",
                "-",
            ],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("[ERROR] GitHub CLI ('gh') is not installed or not found on PATH.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Exception while running gh: {e}", file=sys.stderr)
        return 1

    if result.returncode != 0:
        print(f"[ERROR] Failed to set branch protection (code {result.returncode}):\n{result.stderr}", file=sys.stderr)
        return result.returncode

    print("[SUCCESS] Branch protection rules applied successfully.")
    if result.stdout.strip():
        print(result.stdout.strip())
    return 0


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Configure GitHub branch protection rules via gh CLI."
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"Target branch name to apply protection rules (default: '{DEFAULT_BRANCH}').",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Target GitHub repository in owner/repo format (default: '{DEFAULT_REPO}').",
    )
    parser.add_argument(
        "--enforce-admins",
        dest="enforce_admins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce branch protection rules on administrators (default: True). Use --no-enforce-admins to disable.",
    )
    parser.add_argument(
        "--required-reviews",
        type=int,
        default=1,
        help="Number of approving reviews required before merging (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution and print API payload without contacting GitHub.",
    )
    return parser.parse_args(args)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    return setup_branch_protection(
        repo=args.repo,
        branch=args.branch,
        enforce_admins=args.enforce_admins,
        required_reviews=args.required_reviews,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
