import pandas as pd
import requests
import json
from io import StringIO

from datafetcher.github_search import search_github_datasets
from datafetcher.auto_query import build_github_queries
from database.novel_inputs import fetch_unused_novel_tokens,has_unused_novel_inputs
from database.used_repos import is_repo_checked, mark_repo_checked


def generate_raw_urls(repo_url):
    """
    Try common raw file paths for profanity datasets
    """
    branches = ["master", "main"]
    filenames = [
        "en",
        "badwords.txt",
        "bad_words.txt",
        "profanity.txt",
        "words.txt",
        "list.txt"
    ]

    raw_base = repo_url.replace("github.com", "raw.githubusercontent.com")

    for branch in branches:
        for fname in filenames:
            yield f"{raw_base}/{branch}/{fname}"
def parse_open_text(text):
    """
    Parse text into [{text, label}] records
    """
    rows = []

    # 1️⃣ Try CSV
    try:
        df = pd.read_csv(StringIO(text))
        if "text" in df.columns:
            return df[["text"]].assign(label=1).to_dict("records")
    except Exception:
        pass

    # 2️⃣ Try JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for x in data:
                rows.append({"text": str(x), "label": 1})
            return rows
    except Exception:
        pass

    # 3️⃣ Fallback: line-based
    for line in text.splitlines():
        w = line.strip()
        if w and not w.startswith("#"):
            rows.append({"text": w, "label": 1})

    return rows



def fetch_open_data():
    """
    Fetch open-source bad-word data automatically using OOD-driven GitHub queries
    """

    # 1️⃣ Decide query source
    if has_unused_novel_inputs():
        novel_tokens = fetch_unused_novel_tokens()
    else:
        print(" No OOD signals found — running bootstrap open-data fetch")
        novel_tokens = []

    queries = build_github_queries(novel_tokens)

    if not queries:
        queries = [
            "badwords",
            "profanity list",
            "toxic speech dataset",
            "offensive words",
            "hate speech dataset"
        ]
        print(" Using fallback GitHub queries:", queries)

    # 2️⃣ Discover repos
    repo_urls = search_github_datasets(queries)

    if not repo_urls:
        print("No GitHub repositories found")
        return None

    rows = []
    parsed_any = False

    # 3️⃣ Process repos
    for repo_url in repo_urls:

        if is_repo_checked(repo_url):
            print(f"⏭️ Repo already checked, skipping: {repo_url}")
            continue

        print(f"Checking repo: {repo_url}")
        repo_parsed = False

        try:
            # 🔑 TRY MULTIPLE RAW FILES
            for raw_url in generate_raw_urls(repo_url):

                print(f"Fetching: {raw_url}")
                resp = requests.get(raw_url, timeout=15)

                if resp.status_code != 200:
                    continue

                parsed_rows = parse_open_text(resp.text)

                if not parsed_rows:
                    continue

                rows.extend(parsed_rows)
                mark_repo_checked(repo_url, "parsed")
                repo_parsed = True
                parsed_any = True
                break  # stop after first valid file

            if not repo_parsed:
                mark_repo_checked(repo_url, "failed", "no_known_paths")

        except Exception as e:
            mark_repo_checked(repo_url, "failed", str(e))

    # 4️⃣ Final guard
    if not parsed_any:
        print("No open data parsed")
        return None

    df_final = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    print(f"Parsed {len(df_final)} open samples")

    return df_final
