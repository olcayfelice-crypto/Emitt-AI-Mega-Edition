import json
import re
import urllib.parse
import urllib.request


WIKIPEDIA_API = "https://tr.wikipedia.org/w/api.php"
USER_AGENT = "Emitt/2.0"


STOP_WORDS = {
    "nedir",
    "ne",
    "midir",
    "mi",
    "mı",
    "mu",
    "mü",
    "kimdir",
    "kim",
    "nerede",
    "neresi",
    "nasıl",
    "neden",
    "niçin",
    "hangi",
    "kaç",
    "olan",
    "bir",
    "ve",
    "ile",
    "için",
    "bu",
    "şu",
    "o",
}


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_keywords(query):
    words = re.findall(
        r"\w+",
        query.lower(),
        flags=re.UNICODE
    )

    keywords = []

    for word in words:
        if word in STOP_WORDS:
            continue

        if len(word) < 2:
            continue

        if word not in keywords:
            keywords.append(word)

    return keywords


def build_search_query(query):
    keywords = extract_keywords(query)

    if keywords:
        return " ".join(keywords)

    return query.strip()


def search_wikipedia(query):
    search_query = build_search_query(query)

    print(
        "[Internet] Aranıyor:",
        search_query
    )

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": search_query,
        "srnamespace": "0",
        "srlimit": "5",
    }

    url = (
        WIKIPEDIA_API
        + "?"
        + urllib.parse.urlencode(params)
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT
        }
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=8
        ) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

    except Exception as error:
        print(
            "[Internet] Arama hatası:",
            error
        )
        return []

    return (
        data
        .get("query", {})
        .get("search", [])
    )


def get_page_extract(title):
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exintro": "1",
        "exchars": "2500",
        "titles": title,
    }

    url = (
        WIKIPEDIA_API
        + "?"
        + urllib.parse.urlencode(params)
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT
        }
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=8
        ) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

    except Exception as error:
        print(
            "[Internet] Sayfa hatası:",
            error
        )
        return ""

    pages = (
        data
        .get("query", {})
        .get("pages", {})
    )

    for page in pages.values():
        extract = page.get(
            "extract",
            ""
        )

        if extract:
            return clean_text(extract)

    return ""


def score_result(title, query):
    title_words = set(
        extract_keywords(title)
    )

    query_words = set(
        extract_keywords(query)
    )

    if not title_words or not query_words:
        return 0

    return len(
        title_words & query_words
    )


def choose_result(results, query):
    if not results:
        return None

    ranked = sorted(
        results,
        key=lambda result: score_result(
            result.get("title", ""),
            query
        ),
        reverse=True
    )

    return ranked[0]


def split_sentences(text):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]


def find_relevant_sentences(
    text,
    query,
    limit=5
):
    keywords = extract_keywords(query)

    sentences = split_sentences(text)

    scored = []

    for sentence in sentences:
        sentence_lower = sentence.lower()

        score = 0

        for keyword in keywords:
            if keyword in sentence_lower:
                score += 1

        if score > 0:
            scored.append(
                (score, sentence)
            )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        sentence
        for _, sentence in scored[:limit]
    ]


def search_web(query):
    results = search_wikipedia(query)

    if not results:
        print(
            "[Internet] Sonuç bulunamadı."
        )

        return {
            "title": "",
            "text": "",
            "sentences": [],
            "keywords": extract_keywords(query),
            "words": []
        }

    best = choose_result(
        results,
        query
    )

    if best is None:
        return {
            "title": "",
            "text": "",
            "sentences": [],
            "keywords": extract_keywords(query),
            "words": []
        }

    title = best.get(
        "title",
        ""
    )

    print(
        "[Internet] Kaynak:",
        title
    )

    text = get_page_extract(
        title
    )

    if not text:
        text = clean_text(
            best.get(
                "snippet",
                ""
            )
        )

    keywords = extract_keywords(
        query
    )

    sentences = find_relevant_sentences(
        text,
        query
    )

    words = re.findall(
        r"\w+",
        text,
        flags=re.UNICODE
    )

    print(
        "[Internet] Anahtar kelimeler:",
        ", ".join(keywords)
    )

    print(
        "[Internet] İlgili cümle:",
        len(sentences)
    )

    return {
        "title": title,
        "text": text,
        "sentences": sentences,
        "keywords": keywords,
        "words": words
    }