"""Emitt AI Engine.

Tokenizer + model + internet aramasını birleştiren tek giriş noktası.
API (api.py) ve terminal istemcisi (generation.py) cevap üretmek için
bu sınıfı kullanır. Böylece cevap üretme mantığı tek bir yerde durur.
"""

import os
import re

import numpy as np

from tokenizer import EmittTokenizer
from model import EmittModel
from internet import search_web


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

DEFAULT_VOCABULARY_PATH = os.path.join(DATA_DIR, "vocabulary.json")
DEFAULT_MODEL_PATH = os.path.join(DATA_DIR, "emitt_model.json")

MAX_TOKENS = 20
TEMPERATURE = 0.70
TOP_K = 20

BAD_TOKENS = {
    "=", "'", '"', "(", ")", "[", "]", "{", "}", ":", ";", "%",
    "<PAD>", "<BOS>", "<UNK>",
}


class EmittEngine:

    def __init__(self, vocabulary_path=None, model_path=None, use_internet=True):
        self.name = "Emitt AI Engine"
        self.version = "0.2.0"

        self.vocabulary_path = vocabulary_path or DEFAULT_VOCABULARY_PATH
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.use_internet = use_internet

        self.tokenizer = None
        self.model = None
        self.bad_token_ids = set()

        self.ready = False
        self.load_error = None

        self.load()

    # ============================================================
    # YÜKLEME
    # ============================================================

    def load(self):
        """Tokenizer ve modeli diskten yükler.

        Dosyalar eksikse ya da bozuksa exception fırlatmaz; bunun yerine
        motoru 'not_installed' durumunda bırakır, hatayı load_error
        alanında tutar. API bu sayede model henüz eğitilmemişken de
        çökmeden ayağa kalkabilir.
        """

        self.ready = False
        self.load_error = None

        if not os.path.exists(self.vocabulary_path):
            self.load_error = f"Vocabulary bulunamadı: {self.vocabulary_path}"
            return False

        if not os.path.exists(self.model_path):
            self.load_error = f"Model bulunamadı: {self.model_path}"
            return False

        try:
            tokenizer = EmittTokenizer(max_vocabulary_size=10000)
            tokenizer.load(self.vocabulary_path)

            model = EmittModel.load(self.model_path)

        except Exception as error:
            self.load_error = f"Yükleme hatası: {error}"
            return False

        self.tokenizer = tokenizer
        self.model = model

        self.bad_token_ids = {
            token_id
            for token, token_id in tokenizer.token_to_id.items()
            if token in BAD_TOKENS
        }

        self.ready = True
        return True

    def reload(self):
        """training.py yeniden çalıştırıldıktan sonra, sunucuyu
        yeniden başlatmadan yeni ağırlıkları yüklemek için kullanılır."""
        return self.load()

    # ============================================================
    # SAMPLING
    # ============================================================

    def _select_next_token(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        probabilities = np.maximum(probabilities, 0.0)

        for token_id in self.bad_token_ids:
            if token_id < len(probabilities):
                probabilities[token_id] = 0.0

        total = probabilities.sum()
        if total <= 0:
            return None
        probabilities /= total

        logits = np.log(probabilities + 1e-12)
        logits /= TEMPERATURE
        logits -= np.max(logits)

        probabilities = np.exp(logits)
        probabilities /= (probabilities.sum() + 1e-12)

        k = min(TOP_K, len(probabilities))
        top_indices = np.argpartition(probabilities, -k)[-k:]
        top_probabilities = probabilities[top_indices]

        total = top_probabilities.sum()
        if total <= 0:
            return None
        top_probabilities /= total

        return int(np.random.choice(top_indices, p=top_probabilities))

    def _generate_with_model(self, message, web_data):
        question_ids = self.tokenizer.encode(message, add_special_tokens=False)
        if not question_ids:
            return ""

        context = question_ids.copy()

        keyword_ids = []
        for word in web_data.get("keywords", []):
            token_id = self.tokenizer.token_to_id.get(word)
            if token_id is not None:
                keyword_ids.append(token_id)
        context.extend(keyword_ids[:8])

        generated_ids = []
        eos_id = self.tokenizer.token_to_id["<EOS>"]

        for _ in range(MAX_TOKENS):
            probabilities = self.model.probabilities(
                context[-self.model.context_size:]
            )
            next_token = self._select_next_token(probabilities)

            if next_token is None or next_token == eos_id:
                break

            generated_ids.append(next_token)
            context.append(next_token)

        if not generated_ids:
            return ""

        return self.tokenizer.decode(generated_ids).strip()

    @staticmethod
    def _clean_generated_text(text):
        if not text:
            return ""
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _is_bad_response(response, web_data):
        if not response:
            return True

        words = re.findall(r"\w+", response.lower(), flags=re.UNICODE)

        if len(words) < 3 or len(set(words)) < 3:
            return True

        keywords = [word.lower() for word in web_data.get("keywords", [])]
        if keywords and not any(word in words for word in keywords):
            return True

        return False

    @staticmethod
    def _build_web_answer(web_data):
        sentences = web_data.get("sentences", [])
        if not sentences:
            return ""

        answer = sentences[0]
        if len(answer) < 180 and len(sentences) > 1 and sentences[1] != answer:
            answer += " " + sentences[1]

        return answer.strip()

    # ============================================================
    # GENERATE
    # ============================================================

    def generate(self, message, use_internet=None):
        """Bir mesaja cevap üretir.

        Returns:
            dict: {"response": str, "source": "model" | "web" | "fallback"}

        Raises:
            RuntimeError: motor hazır değilse (model eğitilmemiş).
            ValueError: mesaj boşsa.
        """

        if not self.ready:
            raise RuntimeError(self.load_error or "Emitt motoru hazır değil.")

        message = (message or "").strip()
        if not message:
            raise ValueError("Mesaj boş olamaz.")

        should_search = self.use_internet if use_internet is None else use_internet

        web_data = {"keywords": [], "sentences": []}

        if should_search:
            try:
                web_data = search_web(message)
            except Exception:
                web_data = {"keywords": [], "sentences": []}

        model_response = self._clean_generated_text(
            self._generate_with_model(message, web_data)
        )

        if not self._is_bad_response(model_response, web_data):
            return {"response": model_response, "source": "model"}

        web_answer = self._build_web_answer(web_data)
        if web_answer:
            return {"response": web_answer, "source": "web"}

        return {
            "response": "Bu konuda yeterli bilgi bulamadım.",
            "source": "fallback",
        }
