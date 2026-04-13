"""
JobSearchAI — multi-provider job search agent
Providers: Anthropic (Claude), OpenAI, Google Gemini
"""

import asyncio
import concurrent.futures
import json
import os
import re
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer, util

# from dotenv import load_dotenv  # Optional
# load_dotenv()  # Optional

## Anthropic agent framework
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, tool
from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk

## OpenAI agent framework
from agents import Agent, Runner, WebSearchTool, function_tool, set_trace_processors
from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor

## Google Gemini agent framework
from google.adk.agents import Agent as GoogleAgent
from google.adk.runners import Runner as GoogleRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from langsmith.integrations.google_adk import configure_google_adk

###### Section 1: Configuration & constants

DEFAULT_JOB_SOURCES = [
    "linkedin.com",
    "jobs.lever.co",
    "greenhouse.io",
    "indeed.com",
]

DEFAULT_COUNTRIES = [
    "United States",
    "Switzerland",
    "Netherlands",
    "Singapore",
    "China",
]

PROVIDER_MODELS = {
    "Anthropic": [
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ],
    "OpenAI": [
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.4-2026-03-05",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.3-codex",
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "o3",
        "o3-mini",
    ],
    "Gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
}

DEFAULT_MODEL = "claude-haiku-4-5"

WORK_DIR = Path.home() / "JobSearchAI"
WORK_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_FILENAME = "JobSearchAI_results.xlsx"

EXCEL_COLUMNS = [
    "company_name",
    "job_title",
    "country_location",
    "post_time",
    "job_description",
    "matched_skills",
    "unmatched_skills",
    "cultural_match_evaluation",
    "job_post_link",
    "source_of_job_post",
    "similarity_score",
]

SKILL_VOCABULARY = {
    "python", "r", "sql", "matlab", "docker", "git", "bash", "scala", "java", "c++", "julia",
    "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "keras", "huggingface",
    "llamafactory", "comfyui", "langchain", "dspy", "textgrad", "adalflow",
    "spark", "hadoop", "airflow", "dbt", "mlflow", "wandb", "kubernetes",
    "machine learning", "deep learning", "reinforcement learning", "supervised learning",
    "unsupervised learning", "transfer learning", "fine-tuning", "prompt engineering",
    "natural language processing", "nlp", "large language models", "llm", "llms",
    "computer vision", "rag", "agents", "multi-agent", "generative ai", "diffusion models",
    "transformers", "bert", "gpt", "embeddings", "vector databases", "semantic search",
    "data science", "data mining", "statistical modeling", "causal inference",
    "time series forecasting", "agent based modeling", "mathematical modeling",
    "network science", "a/b testing", "experimentation", "survey", "bayesian inference",
    "regression", "classification", "clustering", "dimensionality reduction",
    "economics", "political science", "sociology", "communication science",
    "computational social science", "bioinformatics", "neuroscience",
    "research", "teaching", "interdisciplinary", "cross-functional", "collaboration",
    "communication", "leadership", "project management", "agile", "scrum",
}

_seen_job_urls_lock = threading.Lock()
seen_job_urls: set[str] = set()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

###### Section 2: Shared utility functions (CV parsing, job ranking, normalisation)

def extract_text_from_cv(cv_text: str) -> dict[str, Any]:
    """Extract skills from raw CV text using the SKILL_VOCABULARY."""
    lower_cv = cv_text.lower()
    matched_skills = [
        s for s in SKILL_VOCABULARY
        if re.search(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])", lower_cv)
    ]
    return {"raw_text": cv_text, "skills": matched_skills}


def extract_skills_from_text(text: str) -> set[str]:
    """Extract all recognised skills from an arbitrary text string."""
    text_lower = text.lower()
    found = set()
    for skill in SKILL_VOCABULARY:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def find_skill_overlap(
    cv_skills: list[str], job_text: str
) -> tuple[list[str], list[str]]:
    """Return (matched_skills, unmatched_skills) between CV and a job description."""
    job_skills = extract_skills_from_text(job_text)
    cv_skill_set = {s.lower() for s in cv_skills}
    matched = sorted(job_skills & cv_skill_set)
    unmatched = sorted(job_skills - cv_skill_set)
    return matched, unmatched


def simple_culture_match(cv_text: str, job_text: str) -> str:
    """Return a plain-text summary of culture alignment between CV and job."""
    markers = {
        "research": ["research", "publication", "scientific", "phd"],
        "startup": ["fast-paced", "startup", "entrepreneurial"],
        "collaboration": ["cross-functional", "collaborative", "team"],
        "independence": ["self-starter", "independent", "autonomous"],
    }
    cv_lower = cv_text.lower()
    job_lower = job_text.lower()
    notes = []
    for label, words in markers.items():
        cv_hit = any(w in cv_lower for w in words)
        job_hit = any(w in job_lower for w in words)
        if cv_hit and job_hit:
            notes.append(f"Strong {label} fit")
        elif job_hit:
            notes.append(f"Job emphasizes {label}")
    return "; ".join(notes) if notes else "Limited explicit cultural signals detected."


def rank_jobs_by_similarity(
    cv_text: str, jobs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rank jobs by cosine similarity between CV embedding and job title+description."""
    cv_embedding = embedding_model.encode(cv_text, convert_to_tensor=True)
    combined_texts = [
        f"{job.get('job_title', '')}\n{job.get('job_description', '')}"
        for job in jobs
    ]
    job_embeddings = embedding_model.encode(
        combined_texts, convert_to_tensor=True, batch_size=32
    )
    scores = util.cos_sim(cv_embedding, job_embeddings)[0]
    ranked = [
        {**job, "similarity_score": round(float(scores[i].item()), 4)}
        for i, job in enumerate(jobs)
    ]
    ranked.sort(key=lambda x: x["similarity_score"], reverse=True)
    return ranked


def normalize_job_record(job: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw job dict into a consistent schema regardless of source."""
    raw_url = (
        job.get("job_post_link") or job.get("url") or job.get("link") or ""
    )
    return {
        "company_name": (
            job.get("company_name") or job.get("company")
            or job.get("companyName") or job.get("employer") or ""
        ),
        "job_title": (
            job.get("job_title") or job.get("title") or job.get("role") or ""
        ),
        "country_location": (
            job.get("country_location") or job.get("location") or job.get("country") or ""
        ),
        "post_time": (
            job.get("post_time") or job.get("posted_time") or job.get("posted_at")
            or job.get("date_posted") or job.get("published_at") or ""
        ),
        "job_description": (
            job.get("job_description") or job.get("description") or job.get("summary") or ""
        ),
        "job_post_link": raw_url,
        "source_of_job_post": (
            job.get("source_of_job_post") or job.get("source") or job.get("website") or ""
        ),
        "matched_skills": job.get("matched_skills", []),
        "unmatched_skills": job.get("unmatched_skills", []),
        "cultural_match_evaluation": job.get("cultural_match_evaluation", ""),
        "similarity_score": job.get("similarity_score", ""),
    }


def load_cv_text_locally(file_path: str) -> str:
    """Load CV text from a local .pdf, .txt, or .md file."""
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file type: {suffix}")

###### Section 3: System prompt (shared across all providers)

def build_system_prompt(job_sources: list[str], countries: list[str]) -> str:
    """
    Build the agent system prompt.
    """
    return (
        "You are a helpful assistant searching for jobs for a PhD graduate.\n"
        "Find only jobs published within the last 30 days.\n"
        f"Only search these websites: {', '.join(job_sources)}.\n"
        f"Only return jobs in these countries: {', '.join(countries)}.\n\n"

        "You have exactly these tools — use NO others:\n"
        "  • Your built-in web search tool  — search the web for job postings\n"
        "  • parse_cv_tool(cv_text)          — extract skills and role from CV\n"
        "  • rank_jobs_tool(cv_text, jobs_json) — deduplicate and rank jobs\n"
        "  • export_jobs_tool(jobs_json)     — save results to JobSearchAI_results.xlsx\n\n"

        "Exact steps to follow:\n"
        "1. Call parse_cv_tool(cv_text) to extract the candidate's role and skills.\n"
        "2. Use your web search tool to search each allowed site using the 'site:'\n"
        "   operator, e.g. '<role> jobs site:linkedin.com'. Base the role on the CV.\n"
        "   From each search, extract up to 10 job postings and collect them all into\n"
        "   a single JSON list with fields: job_title, company_name, job_description,\n"
        "   job_post_link, source_of_job_post, country_location, post_time\n"
        "3. Once ALL sites are searched, pass the combined list to rank_jobs_tool\n"
        "   in ONE call — do not call it after each individual search.\n"
        "4. If the user asked to save, call export_jobs_tool after ranking.\n\n"

        f"Working directory: {WORK_DIR}\n"
    )


def build_gemini_search_prompt(
    job_sources: list[str], countries: list[str], cv_text: str
) -> str:
    """
    System prompt for the Gemini search-only agent (Step 1 of the two-step pipeline).
    """
    return (
        "You are a job search assistant. Your ONLY job is to search the web for job postings "
        "and return them as a structured JSON list. Do NOT rank, filter or export anything.\n\n"

        f"Only search these websites: {', '.join(job_sources)}.\n"
        f"Only return jobs in these countries: {', '.join(countries)}.\n"
        "Find only jobs published within the last 30 days.\n\n"

        "Instructions:\n"
        f"1. Use google_search to search each allowed site using the 'site:' operator.\n"
        "   Base the job role on the candidate's CV provided below.\n"
        "   Search query example: '<role> jobs site:linkedin.com'\n"
        "2. From each search, extract up to 10 job postings.\n"
        "3. Return ALL extracted jobs as a single JSON array and nothing else.\n"
        "   Each object must have these fields:\n"
        "     job_title, company_name, job_description, job_post_link,\n"
        "     source_of_job_post, country_location, post_time\n\n"

        f"Candidate CV:\n{cv_text}\n"
    )

###### Section 4: Shared tool implementations (called by all three providers)

async def _parse_cv_impl(args: dict[str, Any]) -> dict[str, Any]:
    """Parse CV text and return extracted skills as JSON."""
    parsed = extract_text_from_cv(args["cv_text"])
    return {
        "content": [{"type": "text", "text": json.dumps(parsed, ensure_ascii=False)}]
    }


async def _rank_jobs_impl(args: dict[str, Any]) -> dict[str, Any]:
    """
    Deduplicate, filter, rank and enrich a list of job objects against the CV.
    Returns the top 10 jobs with similarity scores, skill overlap and culture fit.
    """
    global seen_job_urls

    cv_text = args["cv_text"]

    try:
        raw_jobs = json.loads(args["jobs_json"])
    except Exception:
        return {
            "content": [{"type": "text", "text": json.dumps({
                "message": (
                    "jobs_json could not be parsed. "
                    "Search for jobs first, then call rank_jobs_tool with a valid JSON list."
                ),
                "jobs": [], "count": 0,
            }, ensure_ascii=False)}]
        }

    if isinstance(raw_jobs, dict) and "jobs" in raw_jobs:
        raw_jobs = raw_jobs["jobs"]

    if not raw_jobs:
        return {
            "content": [{"type": "text", "text": json.dumps({
                "message": "No jobs provided. Search first, then call rank_jobs_tool.",
                "jobs": [], "count": 0,
            }, ensure_ascii=False)}]
        }

    jobs = [normalize_job_record(j) for j in raw_jobs]

    allowed_countries = args.get("allowed_countries", [])
    if allowed_countries:
        allowed_lower = [c.lower() for c in allowed_countries]
        jobs = [
            j for j in jobs
            if j.get("country_location")
            and any(c in j["country_location"].lower() for c in allowed_lower)
        ]

    allowed_sources = args.get("allowed_sources", [])
    if allowed_sources:
        allowed_lower = [s.lower() for s in allowed_sources]
        jobs = [
            j for j in jobs
            if (
                any(s in j.get("job_post_link", "").lower() for s in allowed_lower)
                or any(s in j.get("source_of_job_post", "").lower() for s in allowed_lower)
            )
        ]

    with _seen_job_urls_lock:
        new_jobs = [
            j for j in jobs
            if (link := str(j.get("job_post_link", "")).strip())
            and link not in seen_job_urls
        ]

    if not new_jobs:
        return {
            "content": [{"type": "text", "text": json.dumps({
                "message": "All provided jobs were already seen in previous searches.",
                "jobs": [], "count": 0,
            }, ensure_ascii=False)}]
        }

    parsed_cv = extract_text_from_cv(cv_text)
    ranked = await asyncio.get_running_loop().run_in_executor(
        None, rank_jobs_by_similarity, cv_text, new_jobs
    )
    top_10 = ranked[:10]

    enriched = []
    urls_to_add = []
    for job in top_10:
        desc = job.get("job_description", "")
        matched_skills, unmatched_skills = find_skill_overlap(parsed_cv["skills"], desc)
        enriched.append({
            **job,
            "matched_skills": matched_skills,
            "unmatched_skills": unmatched_skills,
            "cultural_match_evaluation": simple_culture_match(cv_text, desc),
        })
        link = str(job.get("job_post_link", "")).strip()
        if link:
            urls_to_add.append(link)

    with _seen_job_urls_lock:
        seen_job_urls.update(urls_to_add)

    export_result = await _export_jobs_to_excel_impl({
        "jobs_json": json.dumps(enriched, ensure_ascii=False)
    })
    export_msg = export_result["content"][0]["text"]
    if "Failed" in export_msg:
        print(f"[ERROR] Excel export failed: {export_msg}", flush=True)

    status = (
        "DONE - 10 jobs found. Stop searching immediately."
        if len(enriched) >= 10
        else f"Only {len(enriched)} jobs found so far. You may search more."
    )
    return {
        "content": [{"type": "text", "text": json.dumps(
            {"jobs": enriched, "count": len(enriched), "status": status,
             "export_status": export_msg},
            ensure_ascii=False,
        )}]
    }


async def _export_jobs_to_excel_impl(args: dict[str, Any]) -> dict[str, Any]:
    """
    Export ranked jobs to an Excel file.
    """
    payload = json.loads(args["jobs_json"])
    jobs = payload["jobs"] if isinstance(payload, dict) and "jobs" in payload else payload

    output_path = WORK_DIR / EXCEL_FILENAME
    sheet_name = "JobSearchAI"

    rows = []
    for raw_job in jobs:
        job = normalize_job_record(raw_job)
        rows.append({
            "company_name": job["company_name"],
            "job_title": job["job_title"],
            "country_location": job["country_location"],
            "post_time": job["post_time"],
            "job_description": job["job_description"],
            "matched_skills": ", ".join(job.get("matched_skills", [])),
            "unmatched_skills": ", ".join(job.get("unmatched_skills", [])),
            "cultural_match_evaluation": job.get("cultural_match_evaluation", ""),
            "job_post_link": job["job_post_link"],
            "source_of_job_post": job["source_of_job_post"],
            "similarity_score": job.get("similarity_score", ""),
        })

    new_df = pd.DataFrame(rows)
    for col in EXCEL_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[EXCEL_COLUMNS]

    def _write_excel() -> str:
        try:
            if output_path.exists():
                try:
                    existing_df = pd.read_excel(output_path, sheet_name=sheet_name)
                except Exception:
                    existing_df = pd.DataFrame(columns=EXCEL_COLUMNS)

                rename_map = {
                    "Country/Location": "country_location",
                    "source": "source_of_job_post",
                    "url": "job_post_link",
                    "company": "company_name",
                    "posted_time": "post_time",
                    "posted_at": "post_time",
                    "date_posted": "post_time",
                    "published_at": "post_time",
                }
                existing_df = existing_df.rename(columns=rename_map)
                for col in EXCEL_COLUMNS:
                    if col not in existing_df.columns:
                        existing_df[col] = ""
                existing_df = existing_df[EXCEL_COLUMNS]
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            combined_df = combined_df.fillna("")
            important_cols = [
                "company_name", "job_title", "country_location", "post_time",
                "job_description", "source_of_job_post", "job_post_link",
            ]
            combined_df["_completeness"] = combined_df[important_cols].apply(
                lambda row: sum(str(v).strip() != "" for v in row), axis=1
            )
            combined_df = combined_df.sort_values("_completeness", ascending=False)
            combined_df["job_post_link"] = combined_df["job_post_link"].astype(str).str.strip()
            combined_df = combined_df.drop_duplicates(
                subset=["job_post_link"], keep="first"
            )
            combined_df = combined_df.drop(
                columns=["_completeness"], errors="ignore"
            )
            combined_df = combined_df[EXCEL_COLUMNS]

            with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
                combined_df.to_excel(writer, index=False, sheet_name=sheet_name)

            return f"Excel file updated successfully: {output_path}"
        except Exception as exc:
            traceback.print_exc()
            return f"Failed to save/update Excel file: {type(exc).__name__}: {exc}"

    result_text = await asyncio.get_running_loop().run_in_executor(None, _write_excel)
    return {"content": [{"type": "text", "text": result_text}]}

###### Section 5: Anthropic provider

@tool(
    "parse_cv",
    "Parse a CV text and extract structured information like core skills.",
    {"cv_text": str},
)
async def parse_cv(args: dict[str, Any]) -> dict[str, Any]:
    return await _parse_cv_impl(args)


@tool(
    "rank_jobs",
    (
        "Rank a list of structured job objects against the CV. "
        "Filters duplicates, unseen URLs, allowed countries and sources. "
        "Returns up to 10 ranked jobs."
    ),
    {"cv_text": str, "jobs_json": str, "allowed_countries": list, "allowed_sources": list},
)
async def rank_jobs(args: dict[str, Any]) -> dict[str, Any]:
    return await _rank_jobs_impl(args)


@tool(
    "export_jobs_to_excel",
    "Export ranked jobs to JobSearchAI_results.xlsx. "
    "Appends and deduplicates if the file already exists.",
    {"jobs_json": str},
)
async def export_jobs_to_excel(args: dict[str, Any]) -> dict[str, Any]:
    return await _export_jobs_to_excel_impl(args)


def build_anthropic_options(
    job_sources: list[str],
    countries: list[str],
    model: str = DEFAULT_MODEL,
) -> ClaudeAgentOptions:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Please enter your Anthropic API key in the sidebar."
        )
    return ClaudeAgentOptions(
        model=model,
        max_turns=100,
        cwd=str(WORK_DIR),
        permission_mode="acceptEdits",
        system_prompt=build_system_prompt(job_sources, countries),
        allowed_tools=[
            "Agent",
            "Read",
            "Write",
            "Edit",
            "Bash",
            "WebSearch",
            "WebFetch",
            "Glob",
            "Grep",
            "parse_cv",
            "rank_jobs",
            "export_jobs_to_excel",
        ],
    )


async def _run_anthropic(
    user_input: str,
    job_sources: list[str],
    countries: list[str],
    model: str,
) -> str:
    configure_claude_agent_sdk()

    async with ClaudeSDKClient(
        options=build_anthropic_options(job_sources, countries, model)
    ) as client:
        await client.query(user_input)
        last_message = None
        async for message in client.receive_response():
            last_message = message
            if hasattr(message, "result") and message.result:
                return str(message.result)
        raise RuntimeError(
            f"No final Anthropic result received. Last message: {last_message!r}"
        )

###### Section 6: OpenAI provider

async def _run_openai(
    user_input: str,
    job_sources: list[str],
    countries: list[str],
    model: str,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please enter your OpenAI API key in the sidebar."
        )
    os.environ["OPENAI_API_KEY"] = api_key

    @function_tool
    async def parse_cv_tool(cv_text: str) -> str:
        """Parse CV text and extract skills."""
        result = await _parse_cv_impl({"cv_text": cv_text})
        return result["content"][0]["text"]

    @function_tool
    async def rank_jobs_tool(cv_text: str, jobs_json: str) -> str:
        """Deduplicate and rank collected job objects against the CV."""
        result = await _rank_jobs_impl({
            "cv_text": cv_text,
            "jobs_json": jobs_json,
            "allowed_countries": countries,
            "allowed_sources": job_sources,
        })
        return result["content"][0]["text"]

    @function_tool
    async def export_jobs_tool(jobs_json: str) -> str:
        """Export ranked jobs to JobSearchAI_results.xlsx."""
        result = await _export_jobs_to_excel_impl({"jobs_json": jobs_json})
        return result["content"][0]["text"]

    agent = Agent(
        name="JobSearchAI",
        instructions=build_system_prompt(job_sources, countries),
        model=model,
        tools=[WebSearchTool(), parse_cv_tool, rank_jobs_tool, export_jobs_tool],
    )

    result = await Runner.run(agent, user_input)
    if result.final_output:
        return str(result.final_output)
    raise RuntimeError(f"No final OpenAI output received. Result: {result!r}")

###### Section 7: Google Gemini provider

def _run_coroutine_sync(coro) -> Any:
    """
    Run an async coroutine synchronously from any thread.
    Required because the Gemini ADK runner is async but Step 2 calls
    need to happen after the async Step 1 completes.
    """
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(run_in_thread).result()


async def _run_gemini_search_agent(
    cv_text: str,
    job_sources: list[str],
    countries: list[str],
    model: str,
) -> str:
    """
    Step 1 — Run a Gemini agent with ONLY google_search.
    Returns the agent's final text response, which should be a JSON list of jobs.
    """
    session_service = InMemorySessionService()
    app_name = "job_search_ai_search"
    user_id = "user_123"
    session_id = f"session_{uuid.uuid4().hex}"

    agent = GoogleAgent(
        name="job_search_agent",
        model=model,
        description="Searches the web for job postings using Google Search.",
        instruction=build_gemini_search_prompt(job_sources, countries, cv_text),
        tools=[google_search],
    )

    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    runner = GoogleRunner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
    )

    final_text = None
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Search for jobs based on the CV provided.")],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
            if texts:
                final_text = "\n".join(texts)

    return final_text or "[]"


async def _run_gemini(
    user_input: str,
    job_sources: list[str],
    countries: list[str],
    model: str,
) -> str:
    """
    Two-step Gemini pipeline.
    """
    api_key = (
        os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    ).strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is not set. "
            "Please enter your Gemini API key in the sidebar."
        )

    configure_google_adk()

    cv_text = user_input

    raw_search_output = await _run_gemini_search_agent(
        cv_text, job_sources, countries, model
    )

    json_match = re.search(r"\[.*\]", raw_search_output, re.DOTALL)
    if not json_match:
        return "Gemini search agent returned no job results. Please try again."
    jobs_json = json_match.group(0)

    rank_result = await _rank_jobs_impl({
        "cv_text": cv_text,
        "jobs_json": jobs_json,
        "allowed_countries": countries,
        "allowed_sources": job_sources,
    })

    ranked_payload = json.loads(rank_result["content"][0]["text"])
    count = ranked_payload.get("count", 0)
    jobs = ranked_payload.get("jobs", [])

    if not jobs:
        return ranked_payload.get(
            "message", "No new jobs found after ranking. Try searching again."
        )

    export_status = ranked_payload.get("export_status", "")
    if export_status and "Failed" in export_status:
        save_line = f"\n⚠️  Excel save failed: {export_status}"
    else:
        save_line = f"\nResults saved to {WORK_DIR / EXCEL_FILENAME}"

    lines = [f"Found and ranked {count} job(s):\n"]
    for i, job in enumerate(jobs, 1):
        lines.append(
            f"{i}. {job.get('job_title', 'N/A')} — {job.get('company_name', 'N/A')} "
            f"({job.get('country_location', 'N/A')})\n"
            f"   Score: {job.get('similarity_score', 'N/A')} | "
            f"Link: {job.get('job_post_link', 'N/A')}\n"
            f"   Matched skills: {', '.join(job.get('matched_skills', [])) or 'None'}\n"
            f"   Culture: {job.get('cultural_match_evaluation', 'N/A')}\n"
        )
    lines.append(save_line)
    return "\n".join(lines)

###### Section 8: Public entry point

_RUNNERS = {
    "Anthropic": _run_anthropic,
    "OpenAI": _run_openai,
    "Gemini": _run_gemini,
}


async def run_agent_query(
    user_input: str,
    job_sources: list[str] | None = None,
    countries: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    provider: str = "Anthropic",
    seen_urls: set[str] | None = None,
) -> tuple[str, set[str]]:
    """
    Run a job-search query using the specified provider.
    Returns (final_text, updated_seen_urls).
    """
    global seen_job_urls

    with _seen_job_urls_lock:
        seen_job_urls = seen_urls.copy() if seen_urls is not None else set()

    job_sources = job_sources or DEFAULT_JOB_SOURCES.copy()
    countries = countries or DEFAULT_COUNTRIES.copy()

    runner = _RUNNERS.get(provider)
    if not runner:
        raise ValueError(f"Unsupported provider: {provider}")

    async def _run() -> str:
        return await runner(user_input, job_sources, countries, model)

    if os.environ.get("LANGSMITH_API_KEY"):
        if provider == "OpenAI":
            try:
                set_trace_processors([OpenAIAgentsTracingProcessor()])
            except Exception:
                pass
        final_text = await _run()
    else:
        final_text = await _run()

    with _seen_job_urls_lock:
        return final_text, seen_job_urls.copy()


def clear_seen_job_urls() -> None:
    """Reset the seen-URL deduplication set between independent search sessions."""
    global seen_job_urls
    with _seen_job_urls_lock:
        seen_job_urls.clear()