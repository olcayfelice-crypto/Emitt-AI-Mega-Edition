import json
import os
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

CORPUS_PATH = os.path.join(
    DATA_DIR,
    "corpus.txt"
)


# ============================================================
# EMITT CORPUS KONULARI
# ============================================================

TOPICS = [
    "Yapay zeka",
    "Makine öğrenmesi",
    "Bilgisayar",
    "Programlama",
    "Python",
    "JavaScript",
    "C++",
    "Linux",
    "Windows",
    "İnternet",
    "Web",
    "Veri",
    "Algoritma",
    "Robot",
    "Elektronik",
    "Matematik",
    "Fizik",
    "Kimya",
    "Biyoloji",
    "Astronomi",
    "Uzay",
    "Güneş",
    "Dünya",
    "Mars",
    "Ay",
    "Güneş Sistemi",
    "Galaksi",
    "Samanyolu",
    "Kara delik",
    "Yıldız",
    "Türkiye",
    "Ankara",
    "İstanbul",
    "İzmir",
    "Tarih",
    "Osmanlı İmparatorluğu",
    "Roma İmparatorluğu",
    "Antik Yunan",
    "Coğrafya",
    "Okyanus",
    "Dağ",
    "Nehir",
    "İklim",
    "Ekoloji",
    "Hayvan",
    "Kedi",
    "Köpek",
    "Bitki",
    "Tarım",
    "Mühendislik",
    "Makine mühendisliği",
    "Elektrik mühendisliği",
    "Bilgisayar mühendisliği",
    "Ekonomi",
    "Enerji",
    "Güneş enerjisi",
    "Nükleer enerji",
    "Felsefe",
    "Sanat",
    "Müzik",
    "Edebiyat",
    "Dil",
    "Psikoloji",
    "Sosyoloji",
    "Eğitim",
    "Sağlık",
    "Beslenme",
    "Spor",
    "Futbol",
    "Basketbol",
    "Satranç",
]


# ============================================================
# WIKIPEDIA API
# ============================================================

def fetch_wikipedia(title):

    encoded_title = quote(
        title.replace(" ", "_")
    )

    url = (
        "https://tr.wikipedia.org/w/api.php"
        "?action=query"
        "&prop=extracts"
        "&explaintext=1"
        "&format=json"
        "&redirects=1"
        "&titles="
        + encoded_title
    )

    request = Request(
        url,
        headers={
            "User-Agent":
                "Emitt-AI/0.1"
        }
    )

    try:

        with urlopen(
            request,
            timeout=15
        ) as response:

            raw = response.read()

        data = json.loads(
            raw.decode("utf-8")
        )

    except Exception as error:

        print(
            "İndirme hatası:",
            title,
            error
        )

        return None

    pages = (
        data
        .get("query", {})
        .get("pages", {})
    )

    for page in pages.values():

        extract = page.get(
            "extract"
        )

        if extract:
            return extract.strip()

    return None


# ============================================================
# CORPUS OLUŞTUR
# ============================================================

def main():

    print()
    print("=================================")
    print("       EMITT CORPUS BUILDER")
    print("=================================")
    print()

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    successful = 0
    failed = 0

    with open(
        CORPUS_PATH,
        "w",
        encoding="utf-8"
    ) as corpus:

        for index, topic in enumerate(
            TOPICS,
            start=1
        ):

            print(
                f"[{index:03d}/{len(TOPICS):03d}] "
                f"{topic}"
            )

            text = fetch_wikipedia(
                topic
            )

            if not text:

                failed += 1
                continue

            corpus.write(
                "\n\n"
                "===== "
                + topic
                + " =====\n\n"
            )

            corpus.write(
                text
            )

            corpus.write(
                "\n"
            )

            successful += 1

            # API'yi gereksiz yere hızlı
            # sorgulamamak için kısa bekleme.
            time.sleep(
                0.25
            )

    print()
    print("=================================")
    print("       CORPUS TAMAMLANDI")
    print("=================================")
    print()

    print(
        "Başarılı:",
        successful
    )

    print(
        "Başarısız:",
        failed
    )

    print(
        "Corpus:",
        CORPUS_PATH
    )

    if os.path.exists(
        CORPUS_PATH
    ):

        size = os.path.getsize(
            CORPUS_PATH
        )

        print(
            "Boyut:",
            size,
            "byte"
        )

    print()


if __name__ == "__main__":
    main()
