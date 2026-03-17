# import os
# import requests
# from state import NewsState

# DEVVIT_BASE_URL = os.environ["DEVVIT_APP_URL"]  # e.g. https://your-devvit-app.com
# SUBREDDIT_NAME = os.environ.get("REDDIT_SUBREDDIT", "your_subreddit")


# def reddit_node(state: NewsState) -> dict:
#     final_article = state.get("final_article", "").strip()
#     headline = state.get("headline", "").strip()
#     publish_decision = state.get("publish_decision", "")

#     # Only post if publisher approved
#     if publish_decision != "approved" or not final_article:
#         return {
#             "logs": ["Reddit node skipped: article not approved or empty."],
#         }

#     post_data = {
#         "headline":        headline,
#         "final_article":   final_article,
#         "sources":         state.get("sources", []),
#         "fact_check_summary": state.get("fact_check_summary", ""),
#         "verified_claims": state.get("verified_claims", []),
#         "publish_decision": publish_decision,
#     }

#     # Validate 2KB limit before sending
#     import json
#     payload_size = len(json.dumps(post_data).encode("utf-8"))
#     if payload_size > 2048:
#         return {
#             "logs": [
#                 f"Reddit node skipped: postData too large ({payload_size} bytes, max 2048)."
#             ],
#         }

#     response = requests.post(
#         f"{DEVVIT_BASE_URL}/api/create-post",
#         json={
#             "subredditName": SUBREDDIT_NAME,
#             "title": headline,
#             "postData": post_data,
#         },
#         headers={"Content-Type": "application/json"},
#         timeout=15,
#     )
#     response.raise_for_status()
#     result = response.json()

#     return {
#         "logs": [
#             f"Devvit node posted article. Post ID: {result.get('postId')}",
#         ],
#     }