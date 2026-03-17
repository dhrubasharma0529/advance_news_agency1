# news_agency

## Project Description

`news_agency` is a Python-based news automation pipeline using **LangGraph** and **LangChain**
with OpenAI models. It processes news topics through a multi-agent graph:

- **Cache Check** – Queries an in-memory vector store to detect if a semantically similar
  topic has already been processed. On a cache hit, the pipeline skips directly to the
  publisher; on a miss, the full pipeline runs.
- **Planner** – Generates a structured set of plan steps and research questions that guide
  the rest of the pipeline.
- **Researcher** – An agentic subgraph that iteratively searches news sources, fetches pages,
  and builds a set of source-backed research notes.
- **Writer** – An agentic subgraph that plans, drafts, and polishes the article using only
  the verified research notes.
- **Fact-Checker** – An agentic subgraph that extracts claims from the draft and
  iteratively verifies them against live sources.
- **Review Router** – Evaluates fact-check issues and routes the article back to the
  researcher or writer for revision, or forwards it to the editor.
- **Editor** – An agentic subgraph that reviews the fact-checked draft and polishes it
  for tone, clarity, and flow.
- **Publisher** – Validates the article against quality gates (headline, sources,
  severity of issues) and emits the final formatted article or rejects it.
- **Cache Store** – Embeds and stores the finished article in the in-memory vector store
  so future runs on similar topics can retrieve it instantly.

The pipeline uses a state-driven `StateGraph` for modular, conditional processing.
Colored terminal logs indicate different agents and phases for easy tracking.

---

## Setup & Run

```bash
# 1. Clone the repository
git clone git@github.com:dhrubasharma0529/advance_news_agency1.git
cd news_agency

# 2. Create and activate a virtual environment

# Windows
python -m venv myenv
myenv\Scripts\activate

# macOS/Linux
python -m venv myenv
source myenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Edit .env file with your OpenAI credentials
echo "OPENAI_API_KEY=sk-..." >> .env
echo "OPENAI_MODEL=gpt-4o-mini" >> .env  # or gpt-4o, gpt-3.5-turbo, etc.

# 5. Run the pipeline
python main.py
```

---

## Sample Run

```
================================================================
NEWS AGENCY PIPELINE
Topic: RSP victory in Nepal parliamentary elections 2026
================================================================

[NODE 1 - CACHE CHECK]
{
  "cache_hit": false,
  "cache_key": "rsp_nepal_elections_2026",
  "message": "No similar article found. Proceeding to planner."
}
MISS: No cache hit. Running full pipeline.

[NODE 2 - PLANNER]
{
  "plan_steps": [
    "Identify key events and background of the topic",
    "Find credible sources and collect evidence",
    "Extract important facts and context",
    "Draft a structured news article",
    "Verify claims through fact checking",
    "Edit and polish the article before publishing"
  ],
  "research_questions": [
    "What exactly happened regarding RSP victory in Nepal?",
    "When and where did the election take place?",
    "Who are the key individuals or organizations involved?",
    "What are the causes and consequences of this result?",
    "What information is still unclear or disputed?"
  ]
}

[NODE 3 - RESEARCHER]

  [RESEARCHER - reason]
  {
    "reasoning": "Start with a broad search on the RSP election result.",
    "action": "search",
    "action_input": "Rastriya Swatantra Party Nepal election 2026 results",
    "done": false
  }

  [RESEARCHER - search]
  Search for 'Rastriya Swatantra Party Nepal election 2026 results'
  returned 5 result(s).

  [RESEARCHER - observe]
  Observe produced 4 new note(s) and 2 open question(s).

  [RESEARCHER - reason]
  {
    "reasoning": "Fetch the most detailed source for seat counts.",
    "action": "fetch",
    "action_input": "https://kathmandupost.com/national/2026/03/06/rsp-wins-182-seats",
    "done": false
  }

  [RESEARCHER - fetch]
  Fetched page content from
  https://kathmandupost.com/national/2026/03/06/rsp-wins-182-seats

  [RESEARCHER - observe]
  Observe produced 3 new note(s) and 1 open question(s).

  [RESEARCHER - reason]
  {
    "reasoning": "Enough notes gathered for a strong first draft.",
    "action": "finish",
    "action_input": "",
    "done": true
  }

  [RESEARCHER - finalize]
  Researcher finalized output.

[NODE 4 - WRITER]

  [WRITER - plan]
  Writer created plan, outline, and headline package.

  [WRITER - draft]
  Writer produced draft.

  [WRITER - polish]
  Writer polished the draft.

  [WRITER - finalize]
  Writer finalized output.

[NODE 5 - FACT-CHECKER]

  [FACT-CHECKER - extract_claims]
  Fact-checker extracted 5 claims.

  [FACT-CHECKER - reason]
  {
    "reasoning": "Search to verify the seat count claim.",
    "action": "search",
    "action_input": "RSP 182 seats Nepal parliament March 2026",
    "done": false
  }

  [FACT-CHECKER - search]
  Verification search for 'RSP 182 seats Nepal parliament March 2026'
  returned 4 result(s).

  [FACT-CHECKER - observe]
  Fact-check observe produced 3 verified claim(s) and 0 issue(s).

  [FACT-CHECKER - reason]
  {
    "reasoning": "Verify Balendra Shah's background claim.",
    "action": "headlines",
    "action_input": "Balendra Shah Kathmandu mayor rapper RSP leader",
    "done": false
  }

  [FACT-CHECKER - headlines]
  Recent coverage search for 'Balendra Shah Kathmandu mayor rapper RSP'
  returned 3 result(s).

  [FACT-CHECKER - observe]
  Fact-check observe produced 2 verified claim(s) and 0 issue(s).

  [FACT-CHECKER - reason]
  {
    "reasoning": "All key claims verified. Ready to finalize.",
    "action": "finish",
    "action_input": "",
    "done": true
  }

  [FACT-CHECKER - finalize]
  Fact-checker finalized output.

[NODE 6 - REVIEW ROUTER]
{
  "route_decision": "editor",
  "logs": [
    "Router sent flow to editor because there are no fact-check issues."
  ]
}

[NODE 7 - EDITOR]

  [EDITOR - review]
  Editor reviewed the draft and produced editorial notes.
  {
    "editor_notes": [
      "Strengthen the opening sentence for immediacy.",
      "Clarify the timeline of the September 2025 protests.",
      "Tighten the paragraph on opposition party results.",
      "Improve transition between protest context and election outcome."
    ]
  }

  [EDITOR - polish]
  Editor polished the article.

  [EDITOR - finalize]
  Editor finalized output.

[NODE 8 - PUBLISHER]
{
  "publish_decision": "approved",
  "publish_ready": true,
  "publish_notes": [
    "Approved with 0 medium-severity issue(s) noted.",
    "Fact-check summary: All major claims verified against live sources."
  ],
  "word_count": 198
}
OK: Article approved for publication.

[NODE 9 - CACHE STORE]
{
  "cache_key": "rsp_nepal_elections_2026",
  "stored": true,
  "message": "Article embedded and stored for future retrieval."
}
STORED: Article cached under key 'rsp_nepal_elections_2026'.

================================================================
FINAL PUBLISHED ARTICLE
================================================================

# Rastriya Swatantra Party Secures Historic Victory
# in Nepal's Parliamentary Elections

## Led by former rapper Balendra Shah, the RSP's triumph marks
## a transformative moment in Nepal's political landscape
## following youth-led protests against corruption.

In a historic turn of events, the Rastriya Swatantra Party (RSP),
led by former rapper Balendra Shah, won 182 out of 275 seats in
Nepal's parliamentary elections held on March 5, 2026. This victory
signifies a significant shift in the country's political landscape,
occurring in the wake of youth-led protests against corruption that
erupted in September 2025 and resulted in the ousting of former
Prime Minister K.P. Sharma Oli.

Balendra Shah, who previously served as the mayor of Kathmandu,
emerged as a prominent figure during these protests. He campaigned
vigorously on a platform focused on combating corruption, enhancing
governance, and promoting transparency, resonating with a
disillusioned electorate eager for change. His efforts directly
contributed to the RSP's overwhelming electoral success.

The election results reflect a dramatic shift in power dynamics,
with the Nepali Congress party finishing a distant second with only
38 seats. The Communist Party of Nepal (Unified Marxist-Leninist)
secured 25 seats, further illustrating the RSP's dominant support.

This election not only highlights the growing influence of younger
generations in Nepal's politics but also sets the stage for
potential reforms aimed at addressing long-standing issues such as
corruption, governance, and economic development.
```