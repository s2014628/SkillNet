"""
SkillNet Pipeline - Summary Agent

Generate experience summaries from SkillNet skills by analyzing their content
with an LLM. Converts structured skill definitions (SKILL.md + references)
into reusable experience data entries.

Experience types produced:
- SOP系统层经验 (SOP System Experience): Step-by-step operational procedures
  extracted from environment-specific skills (ALFWorld, ScienceWorld, WebShop).
  These encode action sequences, decision patterns, and error recovery strategies.
"""

import os
from typing import Any, Dict, List

from openai import OpenAI

SUMMARY_BASE_URL = os.getenv("SUMMARY_BASE_URL", "http://123.57.228.132:8285/api")
SUMMARY_API_KEY = os.getenv("SUMMARY_API_KEY", "")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "deepseek-v3.2-20251201-160k-local")


# ─── SOP Experience Summary Prompt ─────────────────────────────────

SOP_SYSTEM_PROMPT = """You are an expert at analyzing agent skills and extracting reusable operational experience patterns.

Your task is to analyze a skill definition (including its instructions, references, and scripts) and extract **SOP system-level experience (SOP系统层经验)**.

SOP system experience focuses on:
1. **Action Sequence Patterns**: Step-by-step procedures that the agent should follow
2. **Decision Tree Patterns**: Key decision points and conditional branching logic
3. **Error Detection & Recovery**: How to identify failures and recover from them
4. **Environment Interaction Patterns**: How to effectively interact with the environment (navigation, object manipulation, information retrieval)
5. **Precondition & Postcondition Patterns**: What must be true before and after executing the skill

Provide your analysis in a structured format with clear sections:
- **Core Procedure**: The essential step-by-step workflow
- **Decision Points**: Key branching logic and conditions
- **Error Handling Patterns**: Common failures and recovery strategies
- **Reusable SOP Summary**: Concise, reusable standard operating procedures that can be applied to similar tasks
- **Integration Notes**: How this skill connects with other skills in the same domain
"""

SOP_USER_TEMPLATE = """Analyze the following agent skill and extract SOP experience patterns.

Domain: {domain}
Skill Name: {skill_name}
Description: {description}

## Full Skill Content:
{skill_content}

Please extract SOP system-level experience patterns. Focus on:
1. What step-by-step procedures does this skill encode?
2. What decision points and conditional logic are involved?
3. What error handling and recovery strategies are described?
4. What reusable standard operating procedures can be derived for similar tasks?

Keep your analysis concise and actionable. The output should be useful as experience data
that an agent can reference when encountering similar tasks in the future.
"""


class SkillNetSummaryAgent:
    """
    Summary agent that generates experience summaries from SkillNet skills.

    Converts skill definitions into structured experience data by analyzing
    the skill content with an LLM and extracting reusable operational patterns.
    """

    def __init__(
        self,
        base_url: str = SUMMARY_BASE_URL,
        api_key: str = SUMMARY_API_KEY,
        model: str = SUMMARY_MODEL,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def summarize_skill(
        self,
        skill_sample: Dict[str, Any],
        skill_content: str,
    ) -> Dict[str, Any]:
        """
        Generate an experience summary for a single skill.

        Args:
            skill_sample: Normalized skill sample from data_loader
            skill_content: Full text content built by build_skill_content()

        Returns:
            Summary dict with experience analysis
        """
        skill_name = skill_sample.get("skill_name", "")
        description = skill_sample.get("description", "")
        domain = skill_sample.get("dataset", "")

        user_message = SOP_USER_TEMPLATE.format(
            domain=domain,
            skill_name=skill_name,
            description=description,
            skill_content=skill_content[:8000],  # Truncate very long content
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SOP_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            content = response.choices[0].message.content
            summary = (
                content.strip()
                if content
                else "Summary generation failed: empty response"
            )
        except Exception as e:
            summary = f"Summary generation failed: {e}"

        return {
            "skill_name": skill_name,
            "domain": domain,
            "experience_type": skill_sample.get("experience_type", "sop"),
            "summary": summary,
        }

    def _summarize_one(self, idx_and_args: tuple) -> Dict[str, Any]:
        """Worker function for summarizing a single skill."""
        idx, total, skill_sample, skill_content = idx_and_args
        skill_name = skill_sample.get("skill_name", "?")
        domain = skill_sample.get("dataset", "?")
        print(f"  Summarizing [{idx + 1}/{total}]: {skill_name} " f"(domain={domain})")
        summary = self.summarize_skill(skill_sample, skill_content)
        return {
            "sample_id": skill_sample.get("sample_id", ""),
            "skill_name": skill_name,
            "dataset": domain,
            "experience_type": skill_sample.get("experience_type", "sop"),
            "description": skill_sample.get("description", ""),
            **summary,
        }

    def summarize_batch(
        self,
        skill_samples: List[Dict[str, Any]],
        skill_contents: List[str],
        concurrency: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Summarize a batch of skills.

        Args:
            skill_samples: List of normalized skill samples
            skill_contents: List of full text contents (parallel to skill_samples)
            concurrency: Number of parallel threads for summary generation

        Returns:
            List of summary dicts
        """
        total = len(skill_samples)
        work_items = [
            (idx, total, sample, content)
            for idx, (sample, content) in enumerate(zip(skill_samples, skill_contents))
        ]

        if concurrency <= 1:
            return [self._summarize_one(item) for item in work_items]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        summaries: List[Dict[str, Any]] = []
        print(f"  Summarizing with {concurrency} threads...")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {}
            for item in work_items:
                future = executor.submit(self._summarize_one, item)
                future_to_idx[future] = item[0]

            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    summaries.append(result)
                except Exception as e:
                    idx = future_to_idx[future]
                    print(f"  Summary failed for index {idx}: {e}")

        return summaries
