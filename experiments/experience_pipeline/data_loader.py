"""
SkillNet Pipeline - Data Loader

Load and parse SkillNet skills from SKILL.md files.
Each skill directory contains:
  - SKILL.md: Main skill file with YAML frontmatter (name, description) and markdown body
  - references/: Optional directory with reference markdown files
  - scripts/: Optional directory with helper scripts
  - assets/: Optional directory with asset files

Skills are categorized by domain based on their parent directory:
  - alfworld: Household task automation (SOP experience)
  - scienceworld: Scientific experiment procedures (SOP experience)
  - webshop: E-commerce interaction patterns (SOP experience)

Output: Normalized skill samples in a unified format compatible with
the Stack-Planner experience data pipeline.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

# ─── Domain → Experience type mapping ────────────────────────────────
# All SkillNet experiment domains are procedural (SOP) by nature:
# they encode step-by-step action sequences for specific environments.
DOMAIN_EXPERIENCE_TYPE_MAP: Dict[str, str] = {
    "alfworld": "sop",
    "scienceworld": "sop",
    "webshop": "sop",
}

ALL_DOMAINS = list(DOMAIN_EXPERIENCE_TYPE_MAP.keys())

# Default skills root relative to this file: experiments/experience_pipeline -> experiments/src/skills
DEFAULT_SKILLS_ROOT = os.path.join(os.path.dirname(__file__), "..", "src", "skills")


def _parse_yaml_frontmatter(content: str) -> Dict[str, str]:
    """
    Extract YAML frontmatter from a SKILL.md file.

    Frontmatter is delimited by --- lines at the start of the file.
    Returns a dict with 'name' and 'description' fields.
    """
    frontmatter: Dict[str, str] = {"name": "", "description": ""}

    # Match YAML frontmatter between --- delimiters
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return frontmatter

    yaml_block = match.group(1)

    # Parse name field
    name_match = re.search(r"^name:\s*(.+)$", yaml_block, re.MULTILINE)
    if name_match:
        frontmatter["name"] = name_match.group(1).strip()

    # Parse description field (may be multi-line with YAML block scalar)
    desc_match = re.search(
        r"^description:\s*\|?\s*\n((?:\s+.+\n?)+)", yaml_block, re.MULTILINE
    )
    if desc_match:
        # Multi-line description: strip leading whitespace from each line
        lines = desc_match.group(1).strip().split("\n")
        frontmatter["description"] = " ".join(line.strip() for line in lines)
    else:
        # Single-line description
        desc_match = re.search(r"^description:\s*(.+)$", yaml_block, re.MULTILINE)
        if desc_match:
            frontmatter["description"] = desc_match.group(1).strip()

    return frontmatter


def _extract_body(content: str) -> str:
    """
    Extract the markdown body after YAML frontmatter.
    """
    match = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
    if match:
        return content[match.end() :].strip()
    return content.strip()


def _read_references(skill_dir: str) -> List[Dict[str, str]]:
    """
    Read all reference files from a skill's references/ directory.

    Returns a list of dicts with 'filename' and 'content' fields.
    """
    refs_dir = os.path.join(skill_dir, "references")
    references: List[Dict[str, str]] = []

    if not os.path.isdir(refs_dir):
        return references

    for filename in sorted(os.listdir(refs_dir)):
        filepath = os.path.join(refs_dir, filename)
        if os.path.isfile(filepath) and filename.endswith(".md"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    ref_content = f.read()
                references.append(
                    {
                        "filename": filename,
                        "content": ref_content.strip(),
                    }
                )
            except Exception:
                continue

    return references


def _read_scripts(skill_dir: str) -> List[Dict[str, str]]:
    """
    Read all script files from a skill's scripts/ directory.

    Returns a list of dicts with 'filename' and 'content' fields.
    """
    scripts_dir = os.path.join(skill_dir, "scripts")
    scripts: List[Dict[str, str]] = []

    if not os.path.isdir(scripts_dir):
        return scripts

    for filename in sorted(os.listdir(scripts_dir)):
        filepath = os.path.join(scripts_dir, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    script_content = f.read()
                scripts.append(
                    {
                        "filename": filename,
                        "content": script_content.strip(),
                    }
                )
            except Exception:
                continue

    return scripts


def load_skill(skill_dir: str, domain: str, index: int) -> Optional[Dict[str, Any]]:
    """
    Load a single skill from its directory.

    Args:
        skill_dir: Path to the skill directory (containing SKILL.md)
        domain: Domain name (e.g., 'alfworld', 'webshop')
        index: Index within the domain for sample_id generation

    Returns:
        Normalized skill sample dict, or None if SKILL.md is missing/invalid
    """
    skill_md_path = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(skill_md_path):
        return None

    try:
        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    if not content.strip():
        return None

    # Parse frontmatter and body
    frontmatter = _parse_yaml_frontmatter(content)
    body = _extract_body(content)

    skill_name = frontmatter["name"] or os.path.basename(skill_dir)
    description = frontmatter["description"]

    # Read references and scripts
    references = _read_references(skill_dir)
    scripts = _read_scripts(skill_dir)

    experience_type = DOMAIN_EXPERIENCE_TYPE_MAP.get(domain, "sop")

    return {
        "sample_id": f"{domain}_{index}_{skill_name}",
        "dataset": domain,
        "experience_type": experience_type,
        "skill_name": skill_name,
        "description": description,
        "skill_body": body,
        "references": references,
        "scripts": scripts,
        "metadata": {
            "skill_dir": skill_dir,
            "domain": domain,
            "has_references": len(references) > 0,
            "has_scripts": len(scripts) > 0,
            "num_references": len(references),
            "num_scripts": len(scripts),
        },
    }


def load_domain_skills(
    skills_root: str,
    domain: str,
    max_skills: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load all skills from a specific domain directory.

    Args:
        skills_root: Root directory containing domain subdirectories
        domain: Domain name (e.g., 'alfworld')
        max_skills: Maximum number of skills to load (None for all)

    Returns:
        List of normalized skill samples
    """
    domain_dir = os.path.join(skills_root, domain)
    if not os.path.isdir(domain_dir):
        print(f"  Warning: Domain directory not found: {domain_dir}")
        return []

    samples: List[Dict[str, Any]] = []
    skill_dirs = sorted(
        [
            d
            for d in os.listdir(domain_dir)
            if os.path.isdir(os.path.join(domain_dir, d))
        ]
    )

    for idx, skill_dirname in enumerate(skill_dirs):
        if max_skills is not None and idx >= max_skills:
            break

        skill_dir = os.path.join(domain_dir, skill_dirname)
        sample = load_skill(skill_dir, domain, idx)
        if sample is not None:
            samples.append(sample)

    return samples


def load_all_skills(
    skills_root: str,
    domains: Optional[List[str]] = None,
    max_skills_per_domain: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load skills from all specified domains.

    Args:
        skills_root: Root directory containing domain subdirectories
            (e.g., 'experiments/src/skills')
        domains: List of domain names to load (default: all domains)
        max_skills_per_domain: Maximum skills per domain (None for all)

    Returns:
        Combined list of normalized skill samples from all specified domains
    """
    if domains is None:
        domains = ALL_DOMAINS

    all_samples: List[Dict[str, Any]] = []
    for domain_name in domains:
        domain_name = domain_name.lower().strip()
        if domain_name not in DOMAIN_EXPERIENCE_TYPE_MAP:
            raise ValueError(
                f"Unknown domain: {domain_name}. " f"Available: {ALL_DOMAINS}"
            )

        samples = load_domain_skills(
            skills_root, domain_name, max_skills=max_skills_per_domain
        )
        all_samples.extend(samples)
        print(f"  Loaded {len(samples)} skills from {domain_name}")

    return all_samples


def save_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Save skill samples to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def build_skill_content(sample: Dict[str, Any]) -> str:
    """
    Build full skill content text for LLM analysis.

    Combines SKILL.md body with reference files into a single text block.

    Args:
        sample: A normalized skill sample

    Returns:
        Full text content of the skill for LLM processing
    """
    parts: List[str] = []

    # Skill name and description
    parts.append(f"# Skill: {sample['skill_name']}")
    if sample["description"]:
        parts.append(f"\n## Description\n{sample['description']}")

    # Main skill body
    if sample["skill_body"]:
        parts.append(f"\n## Skill Instructions\n{sample['skill_body']}")

    # Reference documents
    for ref in sample.get("references", []):
        parts.append(f"\n## Reference: {ref['filename']}\n{ref['content']}")

    # Script files (content only, for context)
    for script in sample.get("scripts", []):
        parts.append(
            f"\n## Script: {script['filename']}\n```\n{script['content']}\n```"
        )

    return "\n".join(parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and parse SkillNet skills")
    parser.add_argument(
        "--skills-root",
        type=str,
        default=DEFAULT_SKILLS_ROOT,
        help=(
            "Root directory containing domain subdirectories "
            "(default: experiments/src/skills)"
        ),
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=None,
        help=f"Domains to load (default: all). Options: {ALL_DOMAINS}",
    )
    parser.add_argument("--max-skills", type=int, default=None)
    parser.add_argument(
        "--output-path",
        type=str,
        default="./data/skillnet_samples.json",
    )
    args = parser.parse_args()

    samples = load_all_skills(
        skills_root=args.skills_root,
        domains=args.domains,
        max_skills_per_domain=args.max_skills,
    )
    save_samples(samples, args.output_path)
    print(f"\nExtracted {len(samples)} total skills, saved to {args.output_path}")

    # Print distribution
    from collections import Counter

    dist = Counter(s["dataset"] for s in samples)
    type_dist = Counter(s["experience_type"] for s in samples)
    print(f"Domain distribution: {dict(dist)}")
    print(f"Experience type distribution: {dict(type_dist)}")
