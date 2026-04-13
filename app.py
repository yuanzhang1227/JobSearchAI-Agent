import asyncio
import os
import threading
from pathlib import Path
import streamlit as st
import pandas as pd

from JobSearchAI import (
    DEFAULT_JOB_SOURCES,
    DEFAULT_COUNTRIES,
    PROVIDER_MODELS,
    WORK_DIR,
    load_cv_text_locally,
    run_agent_query,
    clear_seen_job_urls,
)

st.set_page_config(page_title="JobSearchAI", layout="wide")
st.title("JobSearchAI")
st.write("Search and rank jobs based on your CV.")

# Session state initialisation
for key, default in [
    ("cv_text",            None),
    ("seen_urls",          set()),
    ("custom_job_sources", []),
    ("custom_countries",   []),
    ("chat_history",       []),
    ("search_done",        False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "Anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "OpenAI":    os.environ.get("OPENAI_API_KEY",    ""),
        "Gemini":    os.environ.get("GEMINI_API_KEY",    ""),
        "LangSmith": os.environ.get("LANGSMITH_API_KEY", ""),
    }

def run_async(coro):
    result_holder: dict = {}

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder["value"] = loop.run_until_complete(coro)
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()

    if "error" in result_holder:
        raise result_holder["error"]
    return result_holder["value"]

# Sidebar: provider & model
st.sidebar.header("AI Provider")

provider = st.sidebar.selectbox("AI Provider", options=list(PROVIDER_MODELS.keys()))

model_options   = PROVIDER_MODELS[provider] + ["Custom..."]
model_selection = st.sidebar.selectbox("Model", options=model_options)

if model_selection == "Custom...":
    model = st.sidebar.text_input(
        "Enter custom model name",
        placeholder="e.g. gpt-5.4, gemini-2.0-flash, claude-haiku-4-5",
    )
    if not model:
        st.sidebar.warning("Please enter a custom model name.")
else:
    model = model_selection

# Sidebar: API keys
for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
    os.environ.pop(_k, None)

if provider == "Anthropic":
    entered_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        value=st.session_state.api_keys["Anthropic"],
        key="anthropic_key_input",
    )
    st.session_state.api_keys["Anthropic"] = entered_key
    api_key = entered_key
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

elif provider == "OpenAI":
    entered_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        value=st.session_state.api_keys["OpenAI"],
        key="openai_key_input",
    )
    st.session_state.api_keys["OpenAI"] = entered_key
    api_key = entered_key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

elif provider == "Gemini":
    entered_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        value=st.session_state.api_keys["Gemini"],
        key="gemini_key_input",
    )
    st.session_state.api_keys["Gemini"] = entered_key
    api_key = entered_key
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key

else:
    api_key = ""

entered_ls = st.sidebar.text_input(
    "LangSmith API Key",
    type="password",
    placeholder="lsv2_pt_...",
    value=st.session_state.api_keys["LangSmith"],
    key="langsmith_key_input",
)
st.session_state.api_keys["LangSmith"] = entered_ls

if entered_ls:
    os.environ["LANGSMITH_API_KEY"]  = entered_ls
    os.environ["LANGSMITH_TRACING"]  = "true"
    os.environ["LANGSMITH_PROJECT"]  = "JobSearchAI"
    os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
else:
    for _k in ("LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
        os.environ.pop(_k, None)

# Sidebar: job sources
st.sidebar.header("Settings")

job_sources = st.sidebar.multiselect(
    "Job source websites",
    options=["linkedin.com", "jobs.lever.co", "greenhouse.io", "indeed.com"]
            + st.session_state.custom_job_sources,
    default=DEFAULT_JOB_SOURCES + st.session_state.custom_job_sources,
)

with st.sidebar.expander("➕ Add custom job source"):
    new_source = st.text_input(
        "Website domain", placeholder="e.g. glassdoor.com", key="new_source_input"
    )
    if st.button("Add source", key="add_source_btn"):
        cleaned = new_source.strip().lower()
        if cleaned and cleaned not in st.session_state.custom_job_sources:
            st.session_state.custom_job_sources.append(cleaned)
            st.rerun()
        elif not cleaned:
            st.warning("Please enter a domain.")
        else:
            st.warning(f"'{cleaned}' is already in the list.")

    for src in st.session_state.custom_job_sources:
        c1, c2 = st.columns([3, 1])
        c1.write(f"• {src}")
        if c2.button("✕", key=f"del_src_{src}"):
            st.session_state.custom_job_sources.remove(src)
            st.rerun()

# Sidebar: countries
countries = st.sidebar.multiselect(
    "Countries",
    options=["United States", "Switzerland", "Netherlands", "Singapore", "China"]
            + st.session_state.custom_countries,
    default=DEFAULT_COUNTRIES + st.session_state.custom_countries,
)

with st.sidebar.expander("➕ Add custom country"):
    new_country = st.text_input(
        "Country name", placeholder="e.g. Germany", key="new_country_input"
    )
    if st.button("Add country", key="add_country_btn"):
        cleaned = new_country.strip().title()
        if cleaned and cleaned not in st.session_state.custom_countries:
            st.session_state.custom_countries.append(cleaned)
            st.rerun()
        elif not cleaned:
            st.warning("Please enter a country name.")
        else:
            st.warning(f"'{cleaned}' is already in the list.")

    for country in st.session_state.custom_countries:
        c1, c2 = st.columns([3, 1])
        c1.write(f"• {country}")
        if c2.button("✕", key=f"del_country_{country}"):
            st.session_state.custom_countries.remove(country)
            st.rerun()

# Sidebar: reset
st.sidebar.divider()
if st.sidebar.button("🔄 Reset Job Search (clear seen jobs)"):
    clear_seen_job_urls()
    st.session_state.seen_urls          = set()
    st.session_state.custom_job_sources = []
    st.session_state.custom_countries   = []
    st.session_state.chat_history       = []
    st.session_state.search_done        = False
    st.sidebar.success("Search history cleared.")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.search_done  = False
    st.sidebar.success("Chat cleared.")

# CV input
st.subheader("Load CV")

cv_path       = st.text_input("Local CV path", value="")
uploaded_file = st.file_uploader("Or upload your CV", type=["pdf", "txt", "md"])

col1, col2 = st.columns(2)

with col1:
    if st.button("Load CV from path"):
        if cv_path.strip():
            try:
                st.session_state.cv_text = load_cv_text_locally(cv_path.strip())
                st.success("CV loaded from local path.")
            except Exception as e:
                st.error(f"Failed to load CV: {type(e).__name__}: {e}")
        else:
            st.warning("Please enter a CV path.")

with col2:
    if uploaded_file is not None:
        temp_path = Path(WORK_DIR) / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            st.session_state.cv_text = load_cv_text_locally(str(temp_path))
            st.success("CV uploaded and loaded.")
        except Exception as e:
            st.error(f"Failed to load uploaded CV: {type(e).__name__}: {e}")

if st.session_state.cv_text:
    with st.expander("Preview loaded CV text"):
        st.text(st.session_state.cv_text[:3000])

# Validation & display helpers
def validate_inputs() -> bool:
    if not api_key:
        st.warning(f"Please enter your {provider} API key in the sidebar.")
        return False
    if not model:
        st.warning("Please select or enter a model name.")
        return False
    if not st.session_state.cv_text:
        st.warning("Please load a CV first.")
        return False
    if not countries:
        st.warning("Please select at least one country.")
        return False
    if not job_sources:
        st.warning("Please select at least one job source.")
        return False
    return True


def show_active_filters() -> None:
    st.info(
        f"🤖 Provider: **{provider}** | Model: **{model}** | "
        f"🔍 Countries: **{', '.join(countries)}** | "
        f"Sources: **{', '.join(job_sources)}**"
    )


def build_strict_filters() -> str:
    return (
        f"STRICT FILTER — only return jobs in: {', '.join(countries)}. "
        f"Reject any job outside these countries.\n"
        f"STRICT FILTER — only search: {', '.join(job_sources)}. "
        f"Do not use any other websites.\n"
        f"STOP as soon as you have 10 ranked jobs."
    )


def build_chat_context() -> str:
    """
    Summarise the conversation so far into a context block the agent can read.
    Only includes the last 10 exchanges to avoid token bloat.
    """
    if not st.session_state.chat_history:
        return ""
    recent = st.session_state.chat_history[-20:]
    lines  = ["Previous conversation context:"]
    for msg in recent:
        role  = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _run_search(user_message: str, is_followup: bool = False) -> None:
    """
    Run the agent with the given user message.
    """
    st.session_state.chat_history.append({"role": "user", "content": user_message})

    with st.spinner(f"Searching using {provider} / {model}..."):
        try:
            if is_followup and st.session_state.chat_history:
                chat_context = build_chat_context()
                prompt = (
                    f"Here is my CV text:\n\n{st.session_state.cv_text}\n\n"
                    f"{chat_context}\n\n"
                    f"User follow-up: {user_message}\n\n"
                    f"{build_strict_filters()}"
                )
            else:
                already_seen = len(st.session_state.seen_urls)
                seen_note = (
                    f"\nNOTE: {already_seen} job(s) have already been found in previous "
                    f"searches this session. Find NEW jobs not seen before and append them "
                    f"to the same Excel file."
                    if already_seen > 0 else ""
                )
                prompt = (
                    f"Here is my CV text:\n\n{st.session_state.cv_text}\n\n"
                    f"User request: {user_message}{seen_note}\n\n"
                    f"{build_strict_filters()}"
                )

            result, updated_urls = run_async(
                run_agent_query(
                    prompt,
                    job_sources,
                    countries,
                    model,
                    provider,
                    seen_urls=st.session_state.seen_urls,
                )
            )
            st.session_state.seen_urls = updated_urls
            st.session_state.search_done = True

            st.session_state.chat_history.append(
                {"role": "assistant", "content": result}
            )

        except ValueError as e:
            error_msg = f"Configuration error: {e}"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_msg}
            )
            st.error(error_msg)
        except Exception as e:
            error_msg = f"Agent failed: {type(e).__name__}: {e}"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_msg}
            )
            st.error(error_msg)

# Job search trigger UI
st.subheader("Job Search Request")

default_prompt = (
    "Find jobs posted within the last 30 days, "
    "search up to 10 matched jobs, rank them, and save the Excel file."
)
user_request = st.text_area("Your request", value=default_prompt, height=100)

if st.button("▶️ Run Job Search", use_container_width=True):
    if validate_inputs():
        show_active_filters()
        _run_search(user_request, is_followup=False)
        st.rerun()

if st.session_state.seen_urls:
    st.caption(f"🗂️ Total unique jobs seen this session: {len(st.session_state.seen_urls)}")

st.divider()

# Chat interface
st.subheader("💬 Conversation")

# Render all past messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(
    "Ask a follow-up question, e.g. 'Tell me more about job #2' or 'Search for ML jobs instead'...",
    key="chat_input",
):
    if not validate_inputs():
        st.stop()

    search_keywords = {
        "search", "find", "look for", "more jobs", "different jobs",
        "new jobs", "other jobs", "show me jobs",
    }
    is_new_search = any(kw in prompt.lower() for kw in search_keywords)

    _run_search(prompt, is_followup=not is_new_search)
    st.rerun()

if not st.session_state.chat_history:
    st.caption(
        "Run a job search above to get started, then use the chat box "
        "to ask follow-up questions about the results."
    )

st.divider()

# Saved Excel files
st.subheader("Saved Excel Files")

xlsx_files = sorted(Path(WORK_DIR).glob("*.xlsx"))
if xlsx_files:
    selected_file = st.selectbox("Saved files", [f.name for f in xlsx_files])
    selected_path = Path(WORK_DIR) / selected_file

    if st.button("Preview selected Excel file"):
        try:
            st.dataframe(pd.read_excel(selected_path), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read file: {type(e).__name__}: {e}")

    with open(selected_path, "rb") as f:
        st.download_button(
            label="Download selected Excel file",
            data=f,
            file_name=selected_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("No Excel files found yet.")