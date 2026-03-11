"""
SkillNet Experience Data Pipeline

End-to-end pipeline for converting SkillNet skills into experience data:
1. Load skill definitions from SkillNet skill directories
2. Build full skill content (SKILL.md + references + scripts)
3. Generate experience summaries via LLM analysis
4. Build final experience data (classified by experience type)

This pipeline directly converts skills into experience data WITHOUT running
them through the Stack-Planner agent workflow. The skills themselves encode
the operational knowledge that would otherwise be derived from benchmark runs.

Experience Types:
- SOP系统层经验 (SOP System Experience): ALFWorld, ScienceWorld, WebShop
  - Step-by-step action procedures, decision patterns, error recovery

Usage:
    # Run from the SkillNet repo root (auto-detects skills location):
    python -m experiments.experience_pipeline.run_pipeline

    # Or specify skills root explicitly:
    python -m experiments.experience_pipeline.run_pipeline \\
        --skills-root ./experiments/src/skills

    # Process specific domains only:
    python -m experiments.experience_pipeline.run_pipeline \\
        --domains alfworld webshop \\
        --output-dir ./results/skillnet
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.experience_pipeline.data_loader import (
    load_all_skills,
    save_samples,
    build_skill_content,
    ALL_DOMAINS,
    DEFAULT_SKILLS_ROOT,
)
from experiments.experience_pipeline.summary_agent import SkillNetSummaryAgent

DEFAULT_SUMMARY_BASE_URL = "http://123.57.228.132:8285/api"
DEFAULT_SUMMARY_MODEL = "deepseek-v3.2-20251201-160k-local"


def build_experience_data(
    skill_samples: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build final experience data from skills and their summaries.

    Classifies experience entries into SOP categories.
    The structure mirrors the Stack-Planner experience data format.

    Args:
        skill_samples: List of normalized skill samples from data_loader
        summaries: List of experience summaries from summary_agent

    Returns:
        Dict with "sop_experiences" (and "factual_experiences" for
        forward-compatibility) lists
    """
    summary_map: Dict[str, Dict[str, Any]] = {}
    for s in summaries:
        key = s.get("sample_id", "")
        if key:
            summary_map[key] = s

    sop_experiences: List[Dict[str, Any]] = []
    factual_experiences: List[Dict[str, Any]] = []

    for sample in skill_samples:
        sample_id = sample["sample_id"]
        domain = sample["dataset"]
        experience_type = sample.get("experience_type", "sop")

        summary_info = summary_map.get(sample_id, {})

        entry: Dict[str, Any] = {
            "sample_id": sample_id,
            "dataset": domain,
            "experience_type": experience_type,
            "skill_name": sample["skill_name"],
            "description": sample["description"],
            "skill_body": sample["skill_body"],
            "references": sample.get("references", []),
            "experience_summary": summary_info.get("summary", ""),
            "metadata": sample.get("metadata", {}),
        }

        if experience_type == "factual":
            factual_experiences.append(entry)
        else:
            sop_experiences.append(entry)

    return {
        "factual_experiences": factual_experiences,
        "sop_experiences": sop_experiences,
    }


def print_statistics(experience_data: Dict[str, Any]) -> None:
    """Print summary statistics for the experience data."""
    sop = experience_data["sop_experiences"]
    factual = experience_data["factual_experiences"]

    print("\n" + "=" * 60)
    print("Experience Data Statistics")
    print("=" * 60)

    print(f"\n[SOP System Experience (SOP系统层经验)]")
    print(f"  Total skills: {len(sop)}")
    if sop:
        domains: Dict[str, int] = {}
        for e in sop:
            ds = e["dataset"]
            domains[ds] = domains.get(ds, 0) + 1
        for ds, count in sorted(domains.items()):
            print(f"  {ds}: {count} skills")

    if factual:
        print(f"\n[Factual Experience (事实性经验)]")
        print(f"  Total skills: {len(factual)}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillNet Experience Data Pipeline")
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
        help=f"Domains to process (default: all). Options: {ALL_DOMAINS}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/skillnet",
        help="Base output directory. A timestamped subdirectory will be created.",
    )
    parser.add_argument(
        "--max-skills",
        type=int,
        default=None,
        help="Max skills per domain (None for all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of parallel threads for summary generation (default: 3)",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip summary generation. Use with --summaries-file to load existing summaries.",
    )
    parser.add_argument(
        "--summaries-file",
        type=str,
        default=None,
        help=(
            "Path to an existing summaries.json file to load when using "
            "--skip-summary. If not provided with --skip-summary, "
            "experience_summary fields will be empty."
        ),
    )
    parser.add_argument(
        "--summary-base-url",
        type=str,
        default=os.getenv("SUMMARY_BASE_URL", DEFAULT_SUMMARY_BASE_URL),
    )
    parser.add_argument(
        "--summary-api-key",
        type=str,
        default=os.getenv("SUMMARY_API_KEY", ""),
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=os.getenv("SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL),
    )
    args = parser.parse_args()

    # Create timestamped output directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    selected_domains = args.domains or ALL_DOMAINS

    print("=" * 60)
    print("SkillNet Experience Data Pipeline")
    print("=" * 60)
    print(f"Skills root: {args.skills_root}")
    print(f"Domains: {selected_domains}")
    print(f"Output dir: {run_dir}")
    print(f"Max skills per domain: {args.max_skills}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Summary model: {args.summary_model}")
    print("=" * 60)

    # ─── Step 1: Load skills ─────────────────────────────────────
    print("\n[Step 1] Loading SkillNet skills...")
    skill_samples = load_all_skills(
        skills_root=args.skills_root,
        domains=selected_domains,
        max_skills_per_domain=args.max_skills,
    )
    print(f"Loaded {len(skill_samples)} total skills")

    if not skill_samples:
        print("No skills found. Check --skills-root path.")
        return

    samples_file = os.path.join(run_dir, "loaded_skills.json")
    save_samples(skill_samples, samples_file)

    # ─── Step 2: Build skill content ─────────────────────────────
    print("\n[Step 2] Building full skill content...")
    skill_contents = [build_skill_content(s) for s in skill_samples]
    print(f"Built content for {len(skill_contents)} skills")

    # ─── Step 3: Generate experience summaries ───────────────────
    summaries: List[Dict[str, Any]] = []
    summaries_file = os.path.join(run_dir, "summaries.json")

    if not args.skip_summary:
        print(
            f"\n[Step 3] Generating experience summaries "
            f"({len(skill_samples)} skills)..."
        )
        summary_agent = SkillNetSummaryAgent(
            base_url=args.summary_base_url,
            api_key=args.summary_api_key,
            model=args.summary_model,
        )
        summaries = summary_agent.summarize_batch(
            skill_samples,
            skill_contents,
            concurrency=args.concurrency,
        )
        with open(summaries_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"  Summaries saved to: {summaries_file}")

        # Report summary generation stats
        failed = sum(
            1
            for s in summaries
            if s.get("summary", "").startswith("Summary generation failed:")
        )
        print(
            f"  Summary success: {len(summaries) - failed}/{len(summaries)}, "
            f"failed: {failed}/{len(summaries)}"
        )
    else:
        print("\n[Step 3] Skipping summary generation (--skip-summary)")
        # Load from explicit --summaries-file if provided
        load_path = args.summaries_file
        if load_path and os.path.exists(load_path):
            with open(load_path, "r", encoding="utf-8") as f:
                summaries = json.load(f)
            print(f"  Loaded existing summaries from: {load_path}")
            print(f"  {len(summaries)} entries")
        elif load_path:
            print(f"  Warning: summaries file not found: {load_path}")
        else:
            print("  No --summaries-file provided; experience_summary will be empty.")

    # ─── Step 4: Build final experience data ─────────────────────
    print("\n[Step 4] Building final experience data...")
    experience_data = build_experience_data(skill_samples, summaries)

    # Save experience data files
    output_file = os.path.join(run_dir, "experience_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(experience_data, f, ensure_ascii=False, indent=2)

    sop_file = os.path.join(run_dir, "sop_experiences.json")
    with open(sop_file, "w", encoding="utf-8") as f:
        json.dump(
            experience_data["sop_experiences"],
            f,
            ensure_ascii=False,
            indent=2,
        )

    factual_file = os.path.join(run_dir, "factual_experiences.json")
    with open(factual_file, "w", encoding="utf-8") as f:
        json.dump(
            experience_data["factual_experiences"],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ─── Print statistics and summary ────────────────────────────
    print_statistics(experience_data)

    print(f"\nPipeline Complete!")
    print(f"All experience data saved to: {run_dir}")
    print(f"  - Combined: {output_file}")
    print(f"  - SOP experiences: {sop_file}")
    print(f"  - Factual experiences: {factual_file}")
    print(f"  - Loaded skills: {samples_file}")
    print(f"  - Summaries: {summaries_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
