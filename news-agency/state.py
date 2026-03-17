from typing import Annotated, Literal
from typing_extensions import TypedDict
import operator


class SourceItem(TypedDict):
    title: str
    url: str
    snippet: str
    source_type: str


class ResearchNote(TypedDict):
    fact: str
    source_url: str
    confidence: str


class VerifiedClaim(TypedDict):
    claim: str
    supporting_source_url: str
    status: str  # verified | partially_verified


class FactCheckIssue(TypedDict):
    claim: str
    problem: str
    severity: str  # low | medium | high
    suggested_fix: str


class NewsState(TypedDict, total=False):
    topic: str
    angle: str

    sources: list[SourceItem]
    research_notes: list[ResearchNote]
    open_questions: list[str]
    
    plan_steps: list[str]
    research_questions: list[str]
    agent_guidance: dict[str, str]
    current_agent: str
    status: str

    headline: str
    subheadline: str
    article_outline: str
    article_draft: str

    verified_claims: list[VerifiedClaim]
    fact_check_issues: list[FactCheckIssue]
    fact_check_summary: str

    revision_count: int
    max_revisions: int
    route_decision: str

    editor_notes: list[str]

    publish_ready: bool
    publish_decision: str
    publish_notes: list[str]
    final_article: str

    cache_hit:            bool   
    cache_similarity:     float  
    cache_matched_topic:  str    

    reddit_post_url:      str    
    reddit_post_status:   str    

    logs: Annotated[list[str], operator.add]


class ResearcherState(TypedDict, total=False):
    topic: str
    angle: str
    objective: str

    reasoning: str
    action: Literal["search", "headlines", "fetch", "finish"]
    action_input: str
    observation: str

    search_results: list[SourceItem]
    fetched_pages: dict[str, str]

    sources: list[SourceItem]
    research_notes: list[ResearchNote]
    open_questions: list[str]

    iteration: int
    max_iterations: int
    done: bool

    logs: Annotated[list[str], operator.add]


class WriterState(TypedDict, total=False):
    topic: str
    angle: str
    sources: list[SourceItem]
    research_notes: list[ResearchNote]
    open_questions: list[str]

    fact_check_issues: list[FactCheckIssue]
    fact_check_summary: str
    revision_count: int

    writing_plan: str
    article_outline: str
    headline: str
    subheadline: str
    article_draft: str

    iteration: int
    max_iterations: int
    done: bool

    logs: Annotated[list[str], operator.add]


class FactCheckerState(TypedDict, total=False):
    topic: str
    angle: str

    article_draft: str
    sources: list[SourceItem]
    research_notes: list[ResearchNote]

    extracted_claims: list[str]

    reasoning: str
    action: Literal["fetch", "search", "headlines", "finish"]
    action_input: str
    observation: str

    fetched_pages: dict[str, str]
    search_results: list[SourceItem]

    verified_claims: list[VerifiedClaim]
    fact_check_issues: list[FactCheckIssue]
    fact_check_summary: str

    iteration: int
    max_iterations: int
    done: bool

    logs: Annotated[list[str], operator.add]


class EditorState(TypedDict, total=False):
    topic: str
    angle: str

    headline: str
    subheadline: str
    article_draft: str

    research_notes: list[ResearchNote]
    fact_check_summary: str
    fact_check_issues: list[FactCheckIssue]

    editor_notes: list[str]
    done: bool

    logs: Annotated[list[str], operator.add]