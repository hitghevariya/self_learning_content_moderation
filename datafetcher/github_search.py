import requests
from config import GITHUB_TOKEN

print("LOADED github_search.py FROM:", __file__)

GITHUB_API = "https://api.github.com/search/repositories"


def search_github_datasets(
    keywords,
    min_stars: int = 0,
    limit: int = 5
):
    headers = {
        "Accept": "application/vnd.github+json"
    }

    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    repos = set()

    for kw in keywords:
        if not kw or not isinstance(kw, str):
            continue

        query = f'{kw} in:name,description,readme'

        params = {
            "q": query,
            "per_page": limit
        }

        print(f"🔍 GitHub search query: {query}")

        r = requests.get(
            GITHUB_API,
            params=params,
            headers=headers,
            timeout=15
        )

        print("GitHub status:", r.status_code)

        if r.status_code != 200:
            print("GitHub error:", r.text)
            continue

        items = r.json().get("items", [])
        print(f"Found {len(items)} repos")

        for repo in items:
            if repo.get("stargazers_count", 0) >= min_stars:
                repos.add(repo["html_url"])

    return list(repos)
