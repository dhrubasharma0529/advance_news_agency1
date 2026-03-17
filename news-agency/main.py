from pprint import pprint
from graph import graph


if __name__ == "__main__":
    result = graph.invoke(
        {
            "topic": "OpenAI launches a new tool for developers",
            "angle": "What launched and why it matters",
            "revision_count": 0,
            "max_revisions": 2,
            "logs": [],
        }
    )

    pprint(result)

    print("\n=== HEADLINE ===")
    print(result.get("headline", ""))

    print("\n=== SUBHEADLINE ===")
    print(result.get("subheadline", ""))

    print("\n=== ARTICLE DRAFT ===")
    print(result.get("article_draft", ""))

    print("\n=== EDITOR NOTES ===")
    for note in result.get("editor_notes", []):
        print("-", note)

    print("\n=== FACT CHECK SUMMARY ===")
    print(result.get("fact_check_summary", ""))

    print("\n=== PUBLISH DECISION ===")
    print(result.get("publish_decision", ""))

    print("\n=== PUBLISH READY ===")
    print(result.get("publish_ready", False))

    print("\n=== PUBLISH NOTES ===")
    for note in result.get("publish_notes", []):
        print("-", note)

    print("\n=== FINAL ARTICLE ===")
    print(result.get("final_article", ""))

    print("\n=== LOGS ===")
    for log in result.get("logs", []):
        print("-", log)