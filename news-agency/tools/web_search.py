from typing import List
from state import SourceItem


def search_news(topic: str) -> List[SourceItem]:
    """
    Search the news with tavily.
    """
    return [
        {
            "title": f"Sample source about {topic}",
            "url": "https://api.tavily.com/search",
            "snippet": f"This is a placeholder source for {topic}.",
            "source_type": "news",
        }
    ]