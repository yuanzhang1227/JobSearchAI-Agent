# JobSearchAI 🔍

A small, self-developed AI job search agent that automatically searches for job postings based on your CV, powered by **Anthropic Claude**, **OpenAI**, or **Google Gemini**, with optional **LangSmith** tracing.

## Features

- **Multi-provider support** — Choose between Anthropic (Claude), OpenAI (GPT), or Google (Gemini) as your AI backend.
- **CV-aware matching** — Upload your CV (PDF, TXT, or MD) and the agent extracts your skills to find the most relevant jobs.
- **Semantic ranking** — Jobs are ranked by cosine similarity between your CV and job description using `sentence-transformers`.
- **Skill overlap analysis** — See exactly which skills in a job posting you have, and which you're missing.
- **Culture fit evaluation** — Quick heuristic analysis of cultural alignment (research-oriented, startup, collaborative, etc.).
- **Excel export** — Results can be saved to the Excel file.
- **Conversational follow-ups** — Chat interface to ask follow-up questions about results or refine the search.
- **LangSmith tracing** — Trace all agent runs for debugging and observability.

---

## Requirements

- Python 3.10+
- An API key for at least one of: Anthropic, OpenAI, or Google Gemini
- A LangSmith API key for tracing

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/JobSearchAI.git
cd JobSearchAI
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `e.g., http://localhost:8501`.

---

## Usage Guide

### Step 1 — Select your AI provider and model

In the **sidebar**, choose your preferred provider:

| Provider  | Example models                        |
|-----------|---------------------------------------|
| Anthropic | `claude-haiku-4-5`, `claude-sonnet-4-6` |
| OpenAI    | `gpt-5`, `gpt-5-mini`           |
| Gemini    | `gemini-2.5-flash`, `gemini-2.5-flash-lite` |

You can also type a custom model name by selecting **"Custom..."** from the model dropdown.

### Step 2 — Enter your API key

Paste your API key for the selected provider into the sidebar field. 

### Step 3 — Load your CV

Either:
- Enter a **local file path** (PDF, TXT, or MD) and click **Load CV from path**, or
- **Upload your CV** directly via the file uploader.

### Step 4 — Configure job search settings

In the sidebar under **Settings**:

- **Job source websites** — Select from `linkedin.com`, `indeed.com`, `jobs.lever.co`, `greenhouse.io`, or add custom domains.
- **Countries** — Select from the default list (US, Switzerland, Netherlands, Singapore, China) or add your own.

### Step 5 — Run a job search

Edit the search request if desired (the default finds jobs from the last 30 days and returns up to 10 ranked results), then click **▶️ Run Job Search**.

### Step 6 — Follow-up and refine

Use the **💬 Conversation** chat box to:
- Ask for more detail on a specific job: *"Tell me more about job #3"*
- Refine the search: *"Search for machine learning engineer roles instead"*
- Request a new search: *"Find more jobs, exclude the ones already shown"*

### Step 7 — Download results

Under **Saved Excel Files**, select the results file, preview it in-app, or click **Download selected Excel file**.

All output is saved to `~/JobSearchAI/` on your machine: `JobSearchAI_results.xlsx` 

---

## LangSmith Tracing 

[LangSmith](https://smith.langchain.com) is a platform for tracing, debugging, and monitoring LLM-powered applications. To enable LangSmith tracing, create your own LangSmith API key in **Settings → API Keys → Create API Key**, then paste it into the **LangSmith API Key** field in the app.

> **Note:** This uses the EU endpoint. If your LangSmith account is on the US region, change `LANGSMITH_ENDPOINT` to `https://api.smith.langchain.com` directly in `app.py`.

---

## Reset Options

In the sidebar:

- **🔄 Reset Job Search** — Clears all seen job URLs, custom sources, countries, and chat history, starting a completely fresh session.
