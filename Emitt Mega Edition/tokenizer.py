import re
import json
from collections import Counter


class EmittTokenizer:

    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ]

    def __init__(self, max_vocabulary_size=10000):
        self.max_vocabulary_size = max(
            int(max_vocabulary_size),
            len(self.SPECIAL_TOKENS)
        )

        self.token_to_id = {}
        self.id_to_token = {}

        self._create_special_tokens()

    # ========================================================
    # SPECIAL TOKENS
    # ========================================================

    def _create_special_tokens(self):

        for token in self.SPECIAL_TOKENS:
            self._add_token(token)

    # ========================================================
    # TOKEN EKLE
    # ========================================================

    def _add_token(self, token):

        if token in self.token_to_id:
            return

        if len(self.token_to_id) >= self.max_vocabulary_size:
            return

        token_id = len(self.token_to_id)

        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token

    # ========================================================
    # METİN PARÇALAMA
    # ========================================================

    def split_text(self, text):

        if not isinstance(text, str):
            return []

        text = text.strip()

        if not text:
            return []

        return re.findall(
            r"\w+|[^\w\s]",
            text,
            flags=re.UNICODE
        )

    # ========================================================
    # VOCABULARY OLUŞTUR
    # ========================================================

    def build_vocabulary(
        self,
        texts,
        min_frequency=1
    ):

        counter = Counter()

        for text in texts:

            if not isinstance(text, str):
                continue

            tokens = self.split_text(text)

            for token in tokens:
                counter[token] += 1

        # En sık kullanılan tokenlar önce gelir.
        sorted_tokens = sorted(
            counter.items(),
            key=lambda item: (
                -item[1],
                item[0]
            )
        )

        for token, frequency in sorted_tokens:

            if frequency < min_frequency:
                continue

            self._add_token(token)

            if (
                len(self.token_to_id)
                >= self.max_vocabulary_size
            ):
                break

    # ========================================================
    # ENCODE
    # ========================================================

    def encode(
        self,
        text,
        add_special_tokens=True
    ):

        tokens = self.split_text(text)

        ids = []

        if add_special_tokens:

            ids.append(
                self.token_to_id["<BOS>"]
            )

        unk_id = self.token_to_id["<UNK>"]

        for token in tokens:

            token_id = self.token_to_id.get(
                token,
                unk_id
            )

            ids.append(token_id)

        if add_special_tokens:

            ids.append(
                self.token_to_id["<EOS>"]
            )

        return ids

    # ========================================================
    # DECODE
    # ========================================================

    def decode(
        self,
        ids,
        remove_special_tokens=True
    ):

        tokens = []

        for token_id in ids:

            token = self.id_to_token.get(
                int(token_id),
                "<UNK>"
            )

            if remove_special_tokens:

                if token in self.SPECIAL_TOKENS:
                    continue

            tokens.append(token)

        if not tokens:
            return ""

        text = ""

        punctuation_without_space = {
            ".",
            ",",
            "!",
            "?",
            ";",
            ":",
            "%",
            ")",
            "]",
            "}"
        }

        opening_punctuation = {
            "(",
            "[",
            "{"
        }

        for token in tokens:

            if not text:

                text = token

            elif token in punctuation_without_space:

                text += token

            elif token in opening_punctuation:

                text += " " + token

            else:

                text += " " + token

        return text

    # ========================================================
    # SAVE
    # ========================================================

    def save(self, path):

        data = {
            "max_vocabulary_size":
                self.max_vocabulary_size,

            "token_to_id":
                self.token_to_id,

            "id_to_token": {
                str(k): v
                for k, v in self.id_to_token.items()
            }
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self, path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        self.max_vocabulary_size = int(
            data.get(
                "max_vocabulary_size",
                10000
            )
        )

        self.token_to_id = {
            str(k): int(v)
            for k, v in data["token_to_id"].items()
        }

        self.id_to_token = {
            int(k): v
            for k, v in data["id_to_token"].items()
        }

    # ========================================================
    # VOCABULARY SIZE
    # ========================================================

    @property
    def vocabulary_size(self):

        return len(
            self.token_to_id
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    tokenizer = EmittTokenizer(
        max_vocabulary_size=10000
    )

    training_texts = [

        "Merhaba Emitt.",
        "Ben Emitt yapay zekasıyım.",
        "Nasılsın?",
        "Ben iyiyim.",
        "Bugün hava çok güzel.",
        "Yapay zeka öğreniyorum.",

        "Türkiye'nin başkenti Ankara'dır.",
        "Dünya bir gezegendir.",
        "Güneş bir yıldızdır.",
        "Python bir programlama dilidir.",

    ]

    tokenizer.build_vocabulary(
        training_texts
    )

    text = "Merhaba Emitt!"

    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode(
        encoded
    )

    print()
    print("Emitt Tokenizer V2")
    print("------------------")
    print("Metin      :", text)
    print("Token ID   :", encoded)
    print("Geri dönüş :", decoded)
    print(
        "Vocabulary :",
        tokenizer.vocabulary_size
    )
    print(
        "Limit      :",
        tokenizer.max_vocabulary_size
    )