from state import NewsState

def publisher_node(state: NewsState) -> dict:
    headline = state.get("headline", "").strip()
    subheadline = state.get("subheadline", "").strip()
    article_draft = state.get("article_draft", "").strip()

    sources = state.get("sources", [])
    verified_claims = state.get("verified_claims", [])
    fact_check_issues = state.get("fact_check_issues", [])
    fact_check_summary = state.get("fact_check_summary", "")
    open_questions = state.get("open_questions", [])

    publish_notes: list[str] = []

    if not headline:
        publish_notes.append("Missing headline.")

    if not subheadline:
        publish_notes.append("Missing subheadline.")

    if not article_draft:
        publish_notes.append("Article draft is empty.")

    if len(sources) < 2:
        publish_notes.append("Not enough sources for publication.")

    high_issues = [issue for issue in fact_check_issues if issue.get("severity") == "high"]
    medium_issues = [issue for issue in fact_check_issues if issue.get("severity") == "medium"]

    if high_issues:
        publish_notes.append(f"Found {len(high_issues)} high-severity fact-check issue(s).")

    if len(verified_claims) == 0:
        publish_notes.append("No verified claims were recorded.")

    unresolved_open_questions = len(open_questions)
    if unresolved_open_questions > 3:
        publish_notes.append("Too many open questions remain unresolved.")

    publish_ready = (
        bool(headline)
        and bool(subheadline)
        and bool(article_draft)
        and len(sources) >= 2
        and len(high_issues) == 0
    )

    if publish_ready:
        publish_decision = "approved"
        final_article = f"# {headline}\n\n## {subheadline}\n\n{article_draft}"
        
        

       

        if medium_issues:
            publish_notes.append(
                f"Approved with {len(medium_issues)} medium-severity issue(s) noted."
            )
        if fact_check_summary:
            publish_notes.append(f"Fact-check summary: {fact_check_summary}")
    else:
        publish_decision = "rejected"
        final_article = ""

    return {
        "publish_ready": publish_ready,
        "publish_decision": publish_decision,
        "publish_notes": publish_notes,
        "final_article": final_article,
        "logs": [
            f"Publisher decision: {publish_decision}",
            *[f"Publisher note: {note}" for note in publish_notes],
        ],
    }