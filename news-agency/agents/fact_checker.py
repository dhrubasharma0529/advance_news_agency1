import json
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from state import FactCheckerState, VerifiedClaim, FactCheckIssue
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


def extract_claims_node(state: FactCheckerState) -> dict:
    article_draft = state.get("article_draft", "")
    research_notes = state.get("research_notes", [])

    prompt = f"""
You are a newsroom fact-checking assistant.

Extract the most important factual claims from the article draft that should be verified.

Article draft:
{article_draft}

Research notes:
{research_notes}

Return ONLY valid JSON:
{{
  "extracted_claims": [
    "claim 1",
    "claim 2",
    "claim 3"
  ]
}}

Rules:
- extract only factual claims, not opinions
- keep claims concise
- prioritize claims about dates, launches, availability, names, organizations, and product details
- extract at most 7 claims
""".strip()

    response = llm.invoke(prompt)
    parsed = _safe_json_loads(response.content)

    claims = parsed.get("extracted_claims", [])

    return {
        "extracted_claims": claims,
        "logs": [f"Fact-checker extracted {len(claims)} claims."],
    }


def reason_node(state: FactCheckerState) -> dict:
    topic = state["topic"]
    angle = state.get("angle", "")
    claims = state.get("extracted_claims", [])
    sources = state.get("sources", [])
    verified_claims = state.get("verified_claims", [])
    issues = state.get("fact_check_issues", [])
    fetched_pages = state.get("fetched_pages", {})
    observation = state.get("observation", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 4)

    prompt = f"""
You are a newsroom fact-checker agent.

Your job:
- decide the next best verification step
- you may choose one action: fetch, search, headlines, or finish
- do not rewrite the article
- focus on verifying important claims

Topic: {topic}
Angle: {angle}

Claims to verify:
{claims}

Known sources:
{sources}

Fetched pages:
{list(fetched_pages.keys())}

Already verified claims:
{verified_claims}

Current issues:
{issues}

Latest observation:
{observation}

Iteration: {iteration}
Max iterations: {max_iterations}

Return ONLY valid JSON:
{{
  "reasoning": "brief reasoning",
  "action": "fetch" | "search" | "headlines" | "finish",
  "action_input": "url or query or empty string",
  "done": true | false
}}

Guidance:
- choose "fetch" if a promising source URL already exists
- choose "search" if you need broader verification
- choose "headlines" if recency matters
- choose "finish" when enough important claims have been checked
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
        "reasoning": parsed.get("reasoning", ""),
        "action": action,
        "action_input": parsed.get("action_input", ""),
        "done": done,
        "logs": [f"Fact-checker chose action={action} at iteration={iteration}"],
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


def fetch_node(state: FactCheckerState) -> dict:
    url = state.get("action_input", "").strip()

    if not url:
        return {
            "observation": "No URL provided for fetch.",
            "logs": ["Fact-check fetch skipped because no URL was provided."],
        }

    try:
        page_text = fetch_page(url)
        fetched_pages = dict(state.get("fetched_pages", {}))
        fetched_pages[url] = page_text

        return {
            "fetched_pages": fetched_pages,
            "observation": f"Fetched page content from {url}",
            "logs": [f"Fact-checker fetched page: {url}"],
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


def search_node(state: FactCheckerState) -> dict:
    query = state.get("action_input", "").strip() or state["topic"]

    try:
        results = search_news(query)
        merged_sources = _merge_sources(state.get("sources", []), results)

        return {
            "search_results": results,
            "sources": merged_sources,
            "observation": f"Verification search for '{query}' returned {len(results)} result(s).",
            "logs": [f"Verification search for '{query}' returned {len(results)} result(s)."],
        }
    except ResearchToolError as e:
        return {
            "observation": f"Verification search tool error: {e}",
            "logs": [f"Verification search tool error: {e}"],
        }
    except Exception as e:
        return {
            "observation": f"Unexpected verification search error: {e}",
            "logs": [f"Unexpected verification search error: {e}"],
        }


def headlines_node(state: FactCheckerState) -> dict:
    query = state.get("action_input", "").strip() or state["topic"]

    try:
        results = search_headlines(query)
        merged_sources = _merge_sources(state.get("sources", []), results)

        return {
            "search_results": results,
            "sources": merged_sources,
            "observation": f"Recent coverage search for '{query}' returned {len(results)} result(s).",
            "logs": [f"Recent coverage search for '{query}' returned {len(results)} result(s)."],
        }
    except ResearchToolError as e:
        return {
            "observation": f"Recent coverage tool error: {e}",
            "logs": [f"Recent coverage tool error: {e}"],
        }
    except Exception as e:
        return {
            "observation": f"Unexpected recent coverage error: {e}",
            "logs": [f"Unexpected recent coverage error: {e}"],
        }


def observe_node(state: FactCheckerState) -> dict:
    topic = state["topic"]
    angle = state.get("angle", "")
    article_draft = state.get("article_draft", "")
    claims = state.get("extracted_claims", [])
    sources = state.get("sources", [])
    fetched_pages = state.get("fetched_pages", {})
    verified_claims = state.get("verified_claims", [])
    issues = state.get("fact_check_issues", [])
    iteration = state.get("iteration", 0)

    prompt = f"""
You are a newsroom fact-check analyst.

Topic: {topic}
Angle: {angle}

Article draft:
{article_draft}

Claims to verify:
{claims}

Known sources:
{sources}

Fetched pages:
{fetched_pages}

Existing verified claims:
{verified_claims}

Existing issues:
{issues}

Return ONLY valid JSON:
{{
  "verified_claims": [
    {{
      "claim": "claim text",
      "supporting_source_url": "https://...",
      "status": "supported|weak|unsupported"
    }}
  ],
  "fact_check_issues": [
    {{
      "claim": "claim text",
      "problem": "what is wrong or unverified",
      "severity": "low|medium|high",
      "suggested_fix": "how to revise or soften the claim"
    }}
  ],
  "fact_check_summary": "brief summary"
}}

Rules:
- only mark a claim verified if there is support in the sources or fetched pages
- do not invent sources
- if a claim cannot be fully verified, either mark it partially_verified or create an issue
- focus on meaningful issues, not tiny stylistic points
""".strip()

    response = llm.invoke(prompt)
    parsed = _safe_json_loads(response.content)

    new_verified = parsed.get("verified_claims", [])
    new_issues = parsed.get("fact_check_issues", [])
    summary = parsed.get("fact_check_summary", "")

    combined_verified: list[VerifiedClaim] = verified_claims.copy()
    for item in new_verified:
        if item not in combined_verified:
            combined_verified.append(item)

    combined_issues: list[FactCheckIssue] = issues.copy()
    for item in new_issues:
        if item not in combined_issues:
            combined_issues.append(item)

    return {
        "verified_claims": combined_verified,
        "fact_check_issues": combined_issues,
        "fact_check_summary": summary,
        "iteration": iteration + 1,
        "logs": [
            f"Fact-check observe produced {len(new_verified)} verified claim(s) and {len(new_issues)} issue(s)."
        ],
    }


def finalize_node(state: FactCheckerState) -> dict:
    return {
        "done": True,
        "logs": ["Fact-checker finalized output."],
    }


def route_after_reason(state: FactCheckerState) -> str:
    action = state.get("action", "finish")

    if action == "fetch":
        return "fetch"
    if action == "search":
        return "search"
    if action == "headlines":
        return "headlines"
    return "finalize"


def build_fact_checker_graph():
    builder = StateGraph(FactCheckerState)

    builder.add_node("extract_claims", extract_claims_node)
    builder.add_node("reason", reason_node)
    builder.add_node("fetch", fetch_node)
    builder.add_node("search", search_node)
    builder.add_node("headlines", headlines_node)
    builder.add_node("observe", observe_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "extract_claims")
    builder.add_edge("extract_claims", "reason")

    builder.add_conditional_edges(
        "reason",
        route_after_reason,
        {
            "fetch": "fetch",
            "search": "search",
            "headlines": "headlines",
            "finalize": "finalize",
        },
    )

    builder.add_edge("fetch", "observe")
    builder.add_edge("search", "observe")
    builder.add_edge("headlines", "observe")
    builder.add_edge("observe", "reason")
    builder.add_edge("finalize", END)

    return builder.compile()


fact_checker_graph = build_fact_checker_graph()