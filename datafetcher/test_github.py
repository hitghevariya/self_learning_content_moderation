from datafetcher.github_search import search_github_datasets

results = search_github_datasets(
    keywords=[
        "badwords",
        "profanity",
        "toxic speech",
        "offensive words",
        "hate speech"
    ],
    limit=5
)

print("RESULTS:", results)
