import json
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from state import EditorState


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def _safe_json_loads(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    return json.loads(text)


def review_node(state: EditorState) -> dict:
    topic = state["topic"]
    angle = state.get("angle", "")
    headline = state.get("headline", "")
    subheadline = state.get("subheadline", "")
    article_draft = state.get("article_draft", "")
    fact_check_summary = state.get("fact_check_summary", "")
    fact_check_issues = state.get("fact_check_issues", [])
    research_notes = state.get("research_notes", [])

    prompt = f"""
You are a newsroom editor.

Review the article and identify editorial improvements only.

Topic: {topic}
Angle: {angle}

Headline:
{headline}

Subheadline:
{subheadline}

Article draft:
{article_draft}

Fact-check summary:
{fact_check_summary}

Fact-check issues:
{fact_check_issues}

Research notes:
{research_notes}

Return ONLY valid JSON:
{{
  "editor_notes": [
    "note 1",
    "note 2",
    "note 3"
  ]
}}

Rules:
- focus on clarity, flow, readability, structure, transitions, and tone
- do not ask for factual additions beyond the provided material
- do not introduce new claims
- if fact-check issues exist, avoid reintroducing them
- keep notes concrete and actionable
- provide at most 5 notes
""".strip()

    response = llm.invoke(prompt)
    parsed = _safe_json_loads(response.content)

    return {
        "editor_notes": parsed.get("editor_notes", []),
        "logs": ["Editor reviewed the draft and produced editorial notes."],
    }


def polish_node(state: EditorState) -> dict:
    headline = state.get("headline", "")
    subheadline = state.get("subheadline", "")
    article_draft = state.get("article_draft", "")
    editor_notes = state.get("editor_notes", [])
    research_notes = state.get("research_notes", [])
    fact_check_issues = state.get("fact_check_issues", [])

    prompt = f"""
You are a newsroom editor polishing an already fact-checked article.

Headline:
{headline}

Subheadline:
{subheadline}

Article draft:
{article_draft}

Editorial notes:
{editor_notes}

Research notes:
{research_notes}

Fact-check issues:
{fact_check_issues}

Return ONLY valid JSON:
{{
  "headline": "improved headline",
  "subheadline": "improved subheadline",
  "article_draft": "improved article draft in markdown"
}}

Rules:
- improve readability, flow, transitions, and tone
- preserve factual meaning
- do not add unsupported claims
- do not introduce new facts
- keep the article newsroom-like and concise
- do not add citations or a bibliography section
""".strip()

    response = llm.invoke(prompt)
    parsed = _safe_json_loads(response.content)

    return {
        "headline": parsed.get("headline", headline),
        "subheadline": parsed.get("subheadline", subheadline),
        "article_draft": parsed.get("article_draft", article_draft),
        "logs": ["Editor polished the article."],
    }


def finalize_node(state: EditorState) -> dict:
    return {
        "done": True,
        "logs": ["Editor finalized output."],
    }


def build_editor_graph():
    builder = StateGraph(EditorState)

    builder.add_node("review", review_node)
    builder.add_node("polish", polish_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "review")
    builder.add_edge("review", "polish")
    builder.add_edge("polish", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


editor_graph = build_editor_graph()