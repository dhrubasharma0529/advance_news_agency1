import json
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import re

from state import WriterState


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def _safe_json_loads(text: str) -> dict:
    text = text.strip()

    # Remove code block markdown if present
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    # Replace smart quotes with normal quotes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Remove trailing commas before closing brackets/braces
    text = re.sub(r',(\s*[\]}])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        # Return empty dict if still invalid
        return {}


def plan_node(state: WriterState) -> dict:
    topic = state["topic"]
    angle = state.get("angle", "")
    research_notes = state.get("research_notes", [])
    open_questions = state.get("open_questions", [])
    fact_check_issues = state.get("fact_check_issues", [])
    fact_check_summary = state.get("fact_check_summary", "")
    revision_count = state.get("revision_count", 0)

    prompt = f"""
You are a newsroom writer.

Your job is to prepare a writing plan for a news article.

Topic: {topic}
Angle: {angle}
Revision count: {revision_count}

Research notes:
{research_notes}

Open questions:
{open_questions}

Fact check summary:
{fact_check_summary}

Fact check issues:
{fact_check_issues}

Return ONLY valid JSON:
{{
  "writing_plan": "short paragraph describing the writing approach",
  "article_outline": "bullet-style outline as plain text",
  "headline": "proposed headline",
  "subheadline": "proposed subheadline"
}}

Rules:
- use only the provided research notes
- do not invent facts
- if something is uncertain, avoid stating it as fact
- if fact-check issues exist, plan around fixing them
- the article should feel like a professional news report
""".strip()

    response = llm.invoke(prompt)
    parsed = _safe_json_loads(response.content)

    return {
        "writing_plan": parsed.get("writing_plan", ""),
        "article_outline": parsed.get("article_outline", ""),
        "headline": parsed.get("headline", ""),
        "subheadline": parsed.get("subheadline", ""),
        "logs": ["Writer created plan, outline, and headline package."],
    }


def draft_node(state: WriterState) -> dict:
    topic = state["topic"]
    angle = state.get("angle", "")
    headline = state.get("headline", "")
    subheadline = state.get("subheadline", "")
    article_outline = state.get("article_outline", "")
    article_draft = state.get("article_draft", "")
    research_notes = state.get("research_notes", [])
    sources = state.get("sources", [])
    open_questions = state.get("open_questions", [])
    fact_check_issues = state.get("fact_check_issues", [])
    fact_check_summary = state.get("fact_check_summary", "")
    revision_count = state.get("revision_count", 0)

    mode = "revise" if revision_count > 0 and article_draft else "draft"

    prompt = f"""
You are a newsroom writer working on a news article.

Mode: {mode}
Topic: {topic}
Angle: {angle}
Headline: {headline}
Subheadline: {subheadline}

Outline:
{article_outline}

Current draft:
{article_draft}

Research notes:
{research_notes}

Sources:
{sources}

Open questions:
{open_questions}

Fact check summary:
{fact_check_summary}

Fact check issues:
{fact_check_issues}

Return ONLY valid JSON:
{{
  "article_draft": "full article text in markdown"
}}

Rules:
- rely only on the provided research notes and sources
- do not invent details
- do not claim uncertain information as confirmed fact
- if revising, fix the fact-check issues directly in the article
- write clearly and professionally
- do not include a bibliography section
""".strip()

    response = llm.invoke(prompt)
    parsed = _safe_json_loads(response.content)

    return {
        "article_draft": parsed.get("article_draft", article_draft),
        "logs": [f"Writer produced {mode} draft."],
    }


def polish_node(state: WriterState) -> dict:
    headline = state.get("headline", "")
    subheadline = state.get("subheadline", "")
    article_draft = state.get("article_draft", "")
    research_notes = state.get("research_notes", [])
    fact_check_issues = state.get("fact_check_issues", [])

    prompt = f"""
You are a newsroom editor polishing a writer's draft.

Headline:
{headline}

Subheadline:
{subheadline}

Draft:
{article_draft}

Research notes:
{research_notes}

Fact-check issues:
{fact_check_issues}

Return ONLY valid JSON:
{{
  "headline": "improved headline",
  "subheadline": "improved subheadline",
  "article_draft": "polished article text in markdown"
}}

Rules:
- improve clarity, transitions, and readability
- preserve factual meaning
- do not add unsupported claims
- do not add facts not present in the research notes
- ensure the revised draft does not repeat fact-check problems
""".strip()

    response = llm.invoke(prompt)
    parsed = _safe_json_loads(response.content)

    return {
        "headline": parsed.get("headline", headline),
        "subheadline": parsed.get("subheadline", subheadline),
        "article_draft": parsed.get("article_draft", article_draft),
        "logs": ["Writer polished the draft."],
    }


def finalize_node(state: WriterState) -> dict:
    return {
        "done": True,
        "logs": ["Writer finalized output."],
    }


def build_writer_graph():
    builder = StateGraph(WriterState)

    builder.add_node("plan", plan_node)
    builder.add_node("draft", draft_node)
    builder.add_node("polish", polish_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "draft")
    builder.add_edge("draft", "polish")
    builder.add_edge("polish", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


writer_graph = build_writer_graph()