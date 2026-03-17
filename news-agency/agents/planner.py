from state import NewsState




def planner_node(state: NewsState) -> dict:
    print("planner node executed.")
    topic = state["topic"]
    plan_steps = [
        "Identify key events and background of the topic",
        "Find credible sources and collect evidence",
        "Extract important facts and context",
        "Draft a structured news article",
        "Verify claims through fact checking",
        "Edit and polish the article before publishing"
    ]

    # decide what information is needed
    research_questions = [
        f"What exactly happened regarding {topic}?",
        f"When and where did the event related to {topic} occur?",
        f"Who are the key individuals or organizations involved in {topic}?",
        f"What are the causes and consequences of {topic}?",
        f"What information about {topic} is still unclear or disputed?"
    ]

    return {
    "plan_steps": plan_steps,
    "research_questions": research_questions,
    "open_questions": research_questions,  # researcher will use this
    "current_agent": "planner",
    "status": "planned",
    "logs": [f"Planner generated {len(research_questions)} research questions"]
}