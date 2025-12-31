def build_github_queries(novel_tokens):
    """
    Generate GitHub search queries from OOD slang
    """
    base_terms = ["bad words", "profanity", "toxic", "abuse"]

    queries = set()

    for token in novel_tokens:
        if len(token) < 4:
            continue
        if token.isdigit():
            continue

        for base in base_terms:
            queries.add(f"{token} {base}")

    return list(queries)[:20]  
