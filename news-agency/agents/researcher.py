import json
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from state import ResearcherState, ResearchNote
from tools.research_tools import (
    search_news,
    search_headlines,
    fetch_page,
    ResearchToolError,
)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _safe_json_loads(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    return json.loads(text)


def reason_node(state: ResearcherState) -> dict:
    topic = state["topic"]
    angle = state.get("angle", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 4)
    sources = state.get("sources", [])
    notes = state.get("research_notes", [])
    open_questions = state.get("open_questions", [])
    observation = state.get("observation", "")

    objective = state.get(
        "objective",
        f"Research the topic '{topic}' for a news article. Gather source-backed facts only.",
    )

    prompt = f"""
You are a newsroom researcher agent.

Your job:
- decide the next best step
- you may choose one action: search, headlines, fetch, or finish
- do not write the article
- gather source-backed facts only

Current topic: {topic}
Angle: {angle}
Objective: {objective}
Iteration: {iteration}
Max iterations: {max_iterations}

Known sources:
{sources}

Known notes:
{notes}

Open questions:
{open_questions}

Latest observation:
{observation}

Return ONLY valid JSON:
{{
  "reasoning": "brief reasoning",
  "action": "search" | "headlines" | "fetch" | "finish",
  "action_input": "query or url or empty string",
  "done": true | false
}}

Guidance:
- choose "search" for broader discovery
- choose "headlines" for recent coverage
- choose "fetch" if there is a promising URL already in known sources
- choose "finish" when you have enough research for a first draft
- if iteration >= max_iterations, choose "finish"
""".strip()

    response = llm.invoke(prompt)
    parsed = _safe_json_loads(response.content)

    action = parsed.get("action", "finish")
    done = parsed.get("done", False)

    if iteration >= max_iterations:
        action = "finish"
        done = True

    return {
        "objective": objective,
        "reasoning": parsed.get("reasoning", ""),
        "action": action,
        "action_input": parsed.get("action_input", ""),
        "done": done,
        "logs": [f"Reason decided action={action} at iteration={iteration}"],
    }


def _merge_sources(existing: list[dict], new_results: list[dict]) -> list[dict]:
    merged = existing.copy()
    seen = {item.get("url", "") for item in existing}

    for item in new_results:
        url = item.get("url", "")
        if url and url not in seen:
            merged.append(item)
            seen.add(url)

    return merged


def search_node(state: ResearcherState) -> dict:
    query = state.get("action_input") or state["topic"]

    try:
        results = search_news(query)
        merged_sources = _merge_sources(state.get("sources", []), results)
        observation = f"Search for '{query}' returned {len(results)} result(s)."
        return {
            "search_results": results,
            "sources": merged_sources,
            "observation": observation,
            "logs": [observation],
        }
    except ResearchToolError as e:
        return {
            "observation": f"Search tool error: {e}",
            "logs": [f"Search tool error: {e}"],
        }
    except Exception as e:
        return {
            "observation": f"Unexpected search error: {e}",
            "logs": [f"Unexpected search error: {e}"],
        }


def headlines_node(state: ResearcherState) -> dict:
    query = state.get("action_input") or state["topic"]

    try:
        results = search_headlines(query)
        merged_sources = _merge_sources(state.get("sources", []), results)
        observation = f"Headline search for '{query}' returned {len(results)} result(s)."
        return {
            "search_results": results,
            "sources": merged_sources,
            "observation": observation,
            "logs": [observation],
        }
    except ResearchToolError as e:
        return {
            "observation": f"Headline tool error: {e}",
            "logs": [f"Headline tool error: {e}"],
        }
    except Exception as e:
        return {
            "observation": f"Unexpected headline error: {e}",
            "logs": [f"Unexpected headline error: {e}"],
        }


def fetch_node(state: ResearcherState) -> dict:
    url = state.get("action_input", "").strip()

    if not url:
        return {
            "observation": "No URL provided for fetch.",
            "logs": ["Fetch skipped because no URL was provided."],
        }

    try:
        page_text = fetch_page(url)
        fetched_pages = dict(state.get("fetched_pages", {}))
        fetched_pages[url] = page_text

        return {
            "fetched_pages": fetched_pages,
            "observation": f"Fetched page content from {url}",
            "logs": [f"Fetched page: {url}"],
        }
    except ResearchToolError as e:
        return {
            "observation": f"Fetch tool error: {e}",
            "logs": [f"Fetch tool error: {e}"],
        }
    except Exception as e:
        return {
            "observation": f"Unexpected fetch error: {e}",
            "logs": [f"Unexpected fetch error: {e}"],
        }


def observe_node(state: ResearcherState) -> dict:
    topic = state["topic"]
    angle = state.get("angle", "")
    sources = state.get("sources", [])
    fetched_pages = state.get("fetched_pages", {})
    notes = state.get("research_notes", [])
    open_questions = state.get("open_questions", [])
    iteration = state.get("iteration", 0)

    prompt = f"""
You are a newsroom research analyst.

Topic: {topic}
Angle: {angle}

Known sources:
{sources}

Fetched pages:
{fetched_pages}

Existing notes:
{notes}

Existing open questions:
{open_questions}

Return ONLY valid JSON:
{{
  "research_notes": [
    {{
      "fact": "fact text",
      "source_url": "https://...",
      "confidence": "high|medium|low"
    }}
  ],
  "open_questions": ["question 1", "question 2"]
}}

Rules:
- every fact must cite a real source_url from known sources or fetched pages
- do not invent sources
- keep notes concise
""".strip()

    response = llm.invoke(prompt)
    parsed = _safe_json_loads(response.content)

    new_notes = parsed.get("research_notes", [])
    new_questions = parsed.get("open_questions", [])

    combined_notes: list[ResearchNote] = notes.copy()
    for note in new_notes:
        if note not in combined_notes:
            combined_notes.append(note)

    combined_questions = open_questions.copy()
    for q in new_questions:
        if q not in combined_questions:
            combined_questions.append(q)

    return {
        "research_notes": combined_notes,
        "open_questions": combined_questions,
        "iteration": iteration + 1,
        "logs": [
            f"Observe produced {len(new_notes)} new note(s) and {len(new_questions)} open question(s)."
        ],
    }


def finalize_node(state: ResearcherState) -> dict:
    return {
        "done": True,
        "logs": ["Researcher finalized output."],
    }


def route_after_reason(state: ResearcherState) -> str:
    action = state.get("action", "finish")

    if action == "search":
        return "search"
    if action == "headlines":
        return "headlines"
    if action == "fetch":
        return "fetch"
    return "finalize"


def build_researcher_graph():
    builder = StateGraph(ResearcherState)

    builder.add_node("reason", reason_node)
    builder.add_node("search", search_node)
    builder.add_node("headlines", headlines_node)
    builder.add_node("fetch", fetch_node)
    builder.add_node("observe", observe_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "reason")

    builder.add_conditional_edges(
        "reason",
        route_after_reason,
        {
            "search": "search",
            "headlines": "headlines",
            "fetch": "fetch",
            "finalize": "finalize",
        },
    )

    builder.add_edge("search", "observe")
    builder.add_edge("headlines", "observe")
    builder.add_edge("fetch", "observe")
    builder.add_edge("observe", "reason")
    builder.add_edge("finalize", END)

    return builder.compile()


researcher_graph = build_researcher_graph()