"""
SkillNet Pipeline - Data Loader

Fetch skills from the SkillNet Search API and convert them into
the Stack-Planner experience data format.

The SkillNet API (http://api-skillnet.openkg.cn/v1/search) hosts 200,000+
community skills. Each skill has:
  - skill_name, skill_description, author, stars, skill_url, category
  - evaluation: 5-dimension quality scores (safety, completeness,
    executability, cost_awareness, maintainability)

This module paginates through the API to fetch all skills and normalizes
them into a unified sample format compatible with the experience data pipeline.

Optionally, skills can also be loaded from local directories (legacy mode)
for the 121 experiment-level skills in experiments/src/skills/.
"""

import json
import os
import re
import time
import requests
from typing import Any, Dict, List, Optional

# --- SkillNet API Configuration ---
SKILLNET_API_URL = "http://api-skillnet.openkg.cn"
SKILLNET_SEARCH_ENDPOINT = "/v1/search"
DEFAULT_PAGE_SIZE = 50  # Max allowed by API

# --- Category -> Experience type mapping ---
CATEGORY_EXPERIENCE_TYPE_MAP: Dict[str, str] = {
    "Development": "sop",
    "DevOps": "sop",
    "Security": "sop",
    "Productivity": "sop",
    "AIGC": "sop",
    "Science": "sop",
    "Research": "sop",
}

DEFAULT_EXPERIENCE_TYPE = "sop"

# --- Legacy local-file constants (for experiment skills) ---
DOMAIN_EXPERIENCE_TYPE_MAP: Dict[str, str] = {
    "alfworld": "sop",
    "scienceworld": "sop",
    "webshop": "sop",
}
ALL_DOMAINS = list(DOMAIN_EXPERIENCE_TYPE_MAP.keys())
DEFAULT_SKILLS_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "src", "skills"
)


# =====================================================================
# API-based skill loading (primary mode)
# =====================================================================


def fetch_skills_page(
    query: str = "",
    page: int = 1,
    limit: int = DEFAULT_PAGE_SIZE,
    sort_by: str = "stars",
    category: Optional[str] = None,
    min_stars: int = 0,
    api_url: str = SKILLNET_API_URL,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Fetch a single page of skills from the SkillNet Search API.

    Args:
        query: Search query (empty string for all skills)
        page: Page number (1-indexed)
        limit: Results per page (max 50)
        sort_by: Sort field ('stars' or 'recent')
        category: Filter by category (None for all)
        min_stars: Minimum star count filter
        api_url: Base URL of the SkillNet API
        timeout: Request timeout in seconds

    Returns:
        Raw API response dict with 'data', 'meta', 'success' fields
    """
    endpoint = f"{api_url}{SKILLNET_SEARCH_ENDPOINT}"
    params: Dict[str, Any] = {
        "q": query,
        "mode": "keyword",
        "limit": limit,
        "page": page,
        "sort_by": sort_by,
        "min_stars": min_stars,
    }
    if category:
        params["category"] = category

    response = requests.get(endpoint, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_all_skills(
    query: str = "",
    category: Optional[str] = None,
    min_stars: int = 0,
    sort_by: str = "stars",
    max_skills: Optional[int] = None,
    api_url: str = SKILLNET_API_URL,
    page_size: int = DEFAULT_PAGE_SIZE,
    delay_between_pages: float = 0.1,
    resume_from_page: int = 1,
) -> List[Dict[str, Any]]:
    """
    Paginate through the SkillNet API and fetch all matching skills.

    Args:
        query: Search query (empty string for all skills)
        category: Filter by category (None for all)
        min_stars: Minimum star count filter
        sort_by: Sort field ('stars' or 'recent')
        max_skills: Maximum total skills to fetch (None for all)
        api_url: Base URL of the SkillNet API
        page_size: Results per page (max 50)
        delay_between_pages: Seconds to wait between API calls
        resume_from_page: Page number to resume from

    Returns:
        List of raw skill dicts from the API
    """
    all_skills: List[Dict[str, Any]] = []
    page = resume_from_page
    total_available = None

    while True:
        try:
            result = fetch_skills_page(
                query=query,
                page=page,
                limit=page_size,
                sort_by=sort_by,
                category=category,
                min_stars=min_stars,
                api_url=api_url,
            )
        except requests.exceptions.RequestException as e:
            print(f"  API error on page {page}: {e}")
            time.sleep(2)
            try:
                result = fetch_skills_page(
                    query=query,
                    page=page,
                    limit=page_size,
                    sort_by=sort_by,
                    category=category,
                    min_stars=min_stars,
                    api_url=api_url,
                )
            except requests.exceptions.RequestException as e2:
                print(f"  Retry failed on page {page}: {e2}. Stopping.")
                break

        if not result.get("success", False):
            print(f"  API returned success=False on page {page}. Stopping.")
            break

        data = result.get("data", [])
        meta = result.get("meta", {})

        if total_available is None:
            total_available = meta.get("total", 0)
            print(f"  Total skills available: {total_available}")

        if not data:
            break

        all_skills.extend(data)

        if page % 100 == 0 or page == resume_from_page:
            print(
                f"  Fetched page {page}: "
                f"{len(all_skills)} skills so far "
                f"(of {total_available})"
            )

        if max_skills is not None and len(all_skills) >= max_skills:
            all_skills = all_skills[:max_skills]
            break

        if total_available is not None and len(all_skills) >= total_available:
            break

        page += 1
        if delay_between_pages > 0:
            time.sleep(delay_between_pages)

    return all_skills


def normalize_api_skill(
    raw_skill: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """
    Convert a raw API skill dict into the normalized experience data format.

    Args:
        raw_skill: Raw skill dict from the SkillNet API
        index: Global index for sample_id generation

    Returns:
        Normalized skill sample dict
    """
    skill_name = raw_skill.get("skill_name", f"unknown_{index}")
    description = raw_skill.get("skill_description", "")
    author = raw_skill.get("author", "")
    stars = raw_skill.get("stars", 0)
    skill_url = raw_skill.get("skill_url", "")
    category = raw_skill.get("category", "")
    evaluation = raw_skill.get("evaluation", {})

    experience_type = CATEGORY_EXPERIENCE_TYPE_MAP.get(
        category, DEFAULT_EXPERIENCE_TYPE
    )

    # Build evaluation summary as skill_body
    eval_parts: List[str] = []
    if evaluation:
        for dim, info in evaluation.items():
            if isinstance(info, dict):
                level = info.get("level", "N/A")
                reason = info.get("reason", "")
                eval_parts.append(f"- **{dim}**: {level} -- {reason}")

    eval_text = "\n".join(eval_parts) if eval_parts else ""

    body_parts: List[str] = []
    if eval_text:
        body_parts.append(f"## Quality Evaluation\n{eval_text}")

    skill_body = "\n\n".join(body_parts)

    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", skill_name)
    sample_id = f"skillnet_{index}_{clean_name}"

    return {
        "sample_id": sample_id,
        "dataset": category or "uncategorized",
        "experience_type": experience_type,
        "skill_name": skill_name,
        "description": description,
        "skill_body": skill_body,
        "references": [],
        "experience_summary": "",
        "metadata": {
            "source": "skillnet_api",
            "author": author,
            "stars": stars,
            "skill_url": skill_url,
            "category": category,
            "evaluation": evaluation,
            "index": index,
        },
    }


def load_all_skills_from_api(
    query: str = "",
    category: Optional[str] = None,
    min_stars: int = 0,
    sort_by: str = "stars",
    max_skills: Optional[int] = None,
    api_url: str = SKILLNET_API_URL,
    page_size: int = DEFAULT_PAGE_SIZE,
    delay: float = 0.1,
    resume_from_page: int = 1,
) -> List[Dict[str, Any]]:
    """
    Fetch all skills from the SkillNet API and normalize them.

    Returns:
        List of normalized skill samples
    """
    raw_skills = fetch_all_skills(
        query=query,
        category=category,
        min_stars=min_stars,
        sort_by=sort_by,
        max_skills=max_skills,
        api_url=api_url,
        page_size=page_size,
        delay_between_pages=delay,
        resume_from_page=resume_from_page,
    )
    print(f"  Fetched {len(raw_skills)} raw skills from API")

    samples = []
    for idx, raw in enumerate(raw_skills):
        sample = normalize_api_skill(raw, idx)
        samples.append(sample)

    return samples


# =====================================================================
# Local file-based skill loading (legacy mode for experiment skills)
# =====================================================================


def _parse_yaml_frontmatter(content: str) -> Dict[str, str]:
    """Extract YAML frontmatter from a SKILL.md file."""
    frontmatter: Dict[str, str] = {"name": "", "description": ""}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return frontmatter

    yaml_block = match.group(1)
    name_match = re.search(r"^name:\s*(.+)$", yaml_block, re.MULTILINE)
    if name_match:
        frontmatter["name"] = name_match.group(1).strip()

    desc_match = re.search(
        r"^description:\s*\|?\s*\n((?:\s+.+\n?)+)", yaml_block, re.MULTILINE
    )
    if desc_match:
        lines = desc_match.group(1).strip().split("\n")
        frontmatter["description"] = " ".join(line.strip() for line in lines)
    else:
        desc_match = re.search(
            r"^description:\s*(.+)$", yaml_block, re.MULTILINE
        )
        if desc_match:
            frontmatter["description"] = desc_match.group(1).strip()

    return frontmatter


def _extract_body(content: str) -> str:
    """Extract the markdown body after YAML frontmatter."""
    match = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
    if match:
        return content[match.end():].strip()
    return content.strip()


def _read_references(skill_dir: str) -> List[Dict[str, str]]:
    """Read all reference files from a skill's references/ directory."""
    refs_dir = os.path.join(skill_dir, "references")
    references: List[Dict[str, str]] = []
    if not os.path.isdir(refs_dir):
        return references
    for filename in sorted(os.listdir(refs_dir)):
        filepath = os.path.join(refs_dir, filename)
        if os.path.isfile(filepath) and filename.endswith(".md"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    references.append(
                        {"filename": filename, "content": f.read().strip()}
                    )
            except Exception:
                continue
    return references


def _read_scripts(skill_dir: str) -> List[Dict[str, str]]:
    """Read all script files from a skill's scripts/ directory."""
    scripts_dir = os.path.join(skill_dir, "scripts")
    scripts: List[Dict[str, str]] = []
    if not os.path.isdir(scripts_dir):
        return scripts
    for filename in sorted(os.listdir(scripts_dir)):
        filepath = os.path.join(scripts_dir, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    scripts.append(
                        {"filename": filename, "content": f.read().strip()}
                    )
            except Exception:
                continue
    return scripts


def load_skill_from_dir(
    skill_dir: str, domain: str, index: int
) -> Optional[Dict[str, Any]]:
    """Load a single skill from a local directory."""
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

    frontmatter = _parse_yaml_frontmatter(content)
    body = _extract_body(content)
    skill_name = frontmatter["name"] or os.path.basename(skill_dir)
    description = frontmatter["description"]
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
        "experience_summary": "",
        "metadata": {
            "source": "local",
            "skill_dir": skill_dir,
            "domain": domain,
            "has_references": len(references) > 0,
            "has_scripts": len(scripts) > 0,
            "num_references": len(references),
            "num_scripts": len(scripts),
        },
    }


def load_all_skills_from_local(
    skills_root: str,
    domains: Optional[List[str]] = None,
    max_skills_per_domain: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load skills from local directories (legacy mode).

    Args:
        skills_root: Root directory containing domain subdirectories
        domains: Domains to load (default: all)
        max_skills_per_domain: Max skills per domain

    Returns:
        List of normalized skill samples
    """
    if domains is None:
        domains = ALL_DOMAINS

    all_samples: List[Dict[str, Any]] = []
    for domain_name in domains:
        domain_dir = os.path.join(skills_root, domain_name)
        if not os.path.isdir(domain_dir):
            print(f"  Warning: Domain directory not found: {domain_dir}")
            continue

        skill_dirs = sorted(
            d
            for d in os.listdir(domain_dir)
            if os.path.isdir(os.path.join(domain_dir, d))
        )
        count = 0
        for idx, skill_dirname in enumerate(skill_dirs):
            if (
                max_skills_per_domain is not None
                and count >= max_skills_per_domain
            ):
                break
            skill_dir = os.path.join(domain_dir, skill_dirname)
            sample = load_skill_from_dir(skill_dir, domain_name, idx)
            if sample is not None:
                all_samples.append(sample)
                count += 1
        print(f"  Loaded {count} skills from {domain_name}")

    return all_samples


# =====================================================================
# Shared utilities
# =====================================================================


def save_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Save skill samples to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def build_skill_content(sample: Dict[str, Any]) -> str:
    """
    Build full skill content text for LLM analysis.

    For API-sourced skills, combines description + evaluation.
    For local skills, combines SKILL.md body + references + scripts.
    """
    parts: List[str] = []

    parts.append(f"# Skill: {sample['skill_name']}")
    if sample.get("description"):
        parts.append(f"\n## Description\n{sample['description']}")

    if sample.get("skill_body"):
        parts.append(f"\n## Skill Content\n{sample['skill_body']}")

    for ref in sample.get("references", []):
        parts.append(f"\n## Reference: {ref['filename']}\n{ref['content']}")

    for script in sample.get("scripts", []):
        parts.append(
            f"\n## Script: {script['filename']}\n```\n{script['content']}\n```"
        )

    return "\n".join(parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load SkillNet skills (API or local)"
    )
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="Data source: 'api' (SkillNet API) or 'local' (experiment files)",
    )
    parser.add_argument(
        "--max-skills",
        type=int,
        default=100,
        help="Max skills to fetch (default: 100)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category (API mode only)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./data/skillnet_samples.json",
    )
    parser.add_argument(
        "--skills-root",
        type=str,
        default=DEFAULT_SKILLS_ROOT,
        help="Local skills root (local mode only)",
    )
    args = parser.parse_args()

    if args.mode == "api":
        samples = load_all_skills_from_api(
            max_skills=args.max_skills, category=args.category
        )
    else:
        samples = load_all_skills_from_local(
            skills_root=args.skills_root,
            max_skills_per_domain=args.max_skills,
        )

    save_samples(samples, args.output_path)
    print(f"\nExtracted {len(samples)} total skills, saved to {args.output_path}")

    from collections import Counter

    cat_dist = Counter(s["dataset"] for s in samples)
    type_dist = Counter(s["experience_type"] for s in samples)
    print(f"Category distribution: {dict(cat_dist)}")
    print(f"Experience type distribution: {dict(type_dist)}")
