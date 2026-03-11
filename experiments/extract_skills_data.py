"""
Extract skill data from SKILL.md files into a structured JSON format.

Usage:
    python extract_skills_data.py [--skills-dir PATH] [--output PATH] [--domain DOMAIN]

Each SKILL.md has:
  - YAML frontmatter: name, description
  - Markdown body: detailed instructions
  - Optional references/ subdirectory with supplementary docs

Output JSON schema:
  [
    {
      "name": str,
      "description": str,
      "domain": str,          # alfworld | scienceworld | webshop
      "instructions": str,    # Markdown body after frontmatter
      "references": {         # filename -> content, may be empty
        "filename.md": str
      }
    },
    ...
  ]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def parse_skill_md(path: Path) -> dict | None:
    """Parse a SKILL.md file into a dict with name, description, and instructions."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Cannot read {path}: {e}", file=sys.stderr)
        return None

    if not content.strip().startswith("---"):
        print(f"[WARNING] No YAML frontmatter in {path}, skipping.", file=sys.stderr)
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        print(f"[WARNING] Malformed frontmatter in {path}, skipping.", file=sys.stderr)
        return None

    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        print(f"[ERROR] YAML parse error in {path}: {e}", file=sys.stderr)
        return None

    if not isinstance(meta, dict) or not meta.get("name") or not meta.get("description"):
        print(f"[WARNING] Missing name/description in {path}, skipping.", file=sys.stderr)
        return None

    return {
        "name": meta["name"],
        "description": meta["description"],
        "instructions": parts[2].strip(),
    }


def load_references(skill_dir: Path) -> dict[str, str]:
    """Read all files in the references/ subdirectory (text only)."""
    refs: dict[str, str] = {}
    refs_dir = skill_dir / "references"
    if not refs_dir.is_dir():
        return refs
    for file_path in sorted(refs_dir.rglob("*")):
        if not file_path.is_file():
            continue
        try:
            refs[file_path.name] = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, Exception):
            pass  # skip binary or unreadable files
    return refs


def extract_skills(skills_dir: Path, domain_filter: str | None = None) -> list[dict]:
    """Walk the skills directory and extract all skill data."""
    skills: list[dict] = []

    # Top-level subdirectories are domains (alfworld, scienceworld, webshop)
    domain_dirs = sorted(d for d in skills_dir.iterdir() if d.is_dir())

    for domain_dir in domain_dirs:
        domain = domain_dir.name
        if domain_filter and domain != domain_filter:
            continue

        for skill_dir in sorted(d for d in domain_dir.iterdir() if d.is_dir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                print(f"[WARNING] SKILL.md missing in {skill_dir}, skipping.", file=sys.stderr)
                continue

            parsed = parse_skill_md(skill_md)
            if parsed is None:
                continue

            parsed["domain"] = domain
            parsed["references"] = load_references(skill_dir)
            skills.append(parsed)

    return skills


def main():
    parser = argparse.ArgumentParser(
        description="Extract skill data from SKILL.md files to JSON."
    )
    parser.add_argument(
        "--skills-dir",
        default=str(Path(__file__).parent / "src" / "skills"),
        help="Root directory containing domain subdirectories (default: src/skills/)",
    )
    parser.add_argument(
        "--output",
        default="skills_data.json",
        help="Output JSON file path (default: skills_data.json)",
    )
    parser.add_argument(
        "--domain",
        default=None,
        choices=["alfworld", "scienceworld", "webshop"],
        help="Extract only skills from a specific domain (default: all)",
    )
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir)
    if not skills_dir.is_dir():
        print(f"[ERROR] skills-dir not found: {skills_dir}", file=sys.stderr)
        sys.exit(1)

    skills = extract_skills(skills_dir, domain_filter=args.domain)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(skills, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Extracted {len(skills)} skills → {output_path}")

    # Print a brief summary by domain
    from collections import Counter
    counts = Counter(s["domain"] for s in skills)
    for domain, count in sorted(counts.items()):
        print(f"  {domain}: {count} skills")


if __name__ == "__main__":
    main()
