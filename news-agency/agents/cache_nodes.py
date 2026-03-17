
from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.store.base import BaseStore

from state import NewsState


# ── Similarity threshold ───────────────────────────────────────────────────
# Topics with cosine similarity >= this value are treated as "already covered".
# 0.85 is a good default:  "Ukraine ceasefire talks" ~ "Russia Ukraine peace"
# Raise to 0.92 for near-exact matches only.
SIMILARITY_THRESHOLD = 0.85

# Namespace used for all cached articles inside the store
_NAMESPACE = ("news_articles",)

# State keys whose values we persist and restore on a cache hit
_CACHE_KEYS = [
    "headline",
    "subheadline",
    "article_draft",
    "final_article",
    "sources",
    "verified_claims",
    "fact_check_issues",
    "fact_check_summary",
    "open_questions",
    "publish_ready",
    "publish_decision",
    "publish_notes",
    "angle",
    "research_notes",
    "plan_steps",
]


#── Embed function factory ─────────────────────────────────────────────────

def build_embed_fn(model: str = "text-embedding-3-small"):
    """
    Returns a coroutine-based embed function compatible with InMemoryStore.

    InMemoryStore expects:   async (texts: list[str]) -> list[list[float]]
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=model)

    async def embed(texts: list[str]) -> list[list[float]]:
        return await embeddings.aembed_documents(texts)

    return embed


# ── Helpers ────────────────────────────────────────────────────────────────

def _article_key(topic: str) -> str:
    """Stable key derived from the topic string."""
    return topic.lower().strip().replace(" ", "_")[:120]


# ── cache_check_node ───────────────────────────────────────────────────────

async def cache_check_node(state: NewsState, *, store: BaseStore) -> dict:
    """
    Entry node.  Queries InMemoryStore for a semantically similar topic.

    On HIT  → merges cached publisher state so the publisher can re-run
              (or the graph can jump straight past it).
    On MISS → sets cache_hit=False and lets the full pipeline run.
    """
    topic: str = state["topic"]
    print(f"[cache_check] Querying cache for: '{topic}'")

    results = await store.asearch(
        _NAMESPACE,
        query=topic,
        limit=1,                        # we only need the best match
    )

    if results:
        best = results[0]
        # LangGraph SearchItem exposes .score (cosine similarity, 0–1)
        score: float = best.score if hasattr(best, "score") else 0.0

        if score >= SIMILARITY_THRESHOLD:
            matched_topic: str = best.value.get("topic", "")
            cached_state: dict = best.value.get("state", {})

            print(
                f"[cache_check] HIT — matched '{matched_topic}' "
                f"(similarity={score:.3f})"
            )

            return {
                **cached_state,                  # restore all publisher outputs
                "cache_hit":           True,
                "cache_similarity":    round(score, 4),
                "cache_matched_topic": matched_topic,
                "current_agent":       "cache_check",
                "status":              "cache_hit",
                "logs": [
                    f"Cache HIT: matched '{matched_topic}' "
                    f"with similarity={score:.3f}"
                ],
            }

    print("[cache_check] MISS — running full pipeline.")
    return {
        "cache_hit":           False,
        "cache_similarity":    0.0,
        "cache_matched_topic": "",
        "current_agent":       "cache_check",
        "status":              "cache_miss",
        "logs": ["Cache MISS: topic not found, running full pipeline."],
    }


# ── route_after_cache_check ────────────────────────────────────────────────

def route_after_cache_check(state: NewsState) -> str:
    """Conditional edge: 'cache_hit' → publisher, 'cache_miss' → planner."""
    return "cache_hit" if state.get("cache_hit") else "cache_miss"


# ── cache_store_node ───────────────────────────────────────────────────────

async def cache_store_node(state: NewsState, *, store: BaseStore) -> dict:
    """
    Terminal node (runs after publisher).

    Persists the approved article to InMemoryStore ONLY when:
      - publish_decision == "approved"
      - this result was NOT itself served from the cache
        (prevents storing stale re-served content)

    The stored value is:
        {
            "topic": <original topic string>,
            "state": { ...publisher outputs... }
        }

    InMemoryStore will embed the topic string for future similarity queries.
    """
    topic: str = state["topic"]

    # Skip if this was a cache-served result
    if state.get("cache_hit"):
        print("[cache_store] Skipping — result was served from cache.")
        return {"logs": ["Cache store skipped (result was a cache hit)."]}

    # Skip if the article was rejected
    if state.get("publish_decision") != "approved":
        print(f"[cache_store] Skipping — article was rejected.")
        return {"logs": ["Cache store skipped (article not approved)."]}

    payload = {
        "topic": topic,
        "state": {k: state.get(k) for k in _CACHE_KEYS},
    }

    key = _article_key(topic)

    await store.aput(
        _NAMESPACE,
        key,
        payload,
        # index=True tells InMemoryStore to embed this item.
        # By default it embeds the entire value; to embed only the topic
        # field pass:  index=["topic"]
        index=["topic"],
    )

    print(f"[cache_store] Stored '{topic}' under key '{key}'.")
    return {
        "logs": [f"Stored topic '{topic}' in LangGraph InMemoryStore (key='{key}')."],
    }
