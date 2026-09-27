import os
import sys
import random
import numpy as np


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

if CURRENT_DIR not in sys.path:
    sys.path.insert(
        0,
        CURRENT_DIR
    )


from tokenizer import EmittTokenizer
from model import EmittModel


# ============================================================
# EMITT TEMEL EĞİTİM VERİLERİ
# ============================================================

CONVERSATIONS = [

    (
        "Merhaba",
        "Merhaba! Sana nasıl yardımcı olabilirim?"
    ),

    (
        "Merhaba!",
        "Merhaba! Sana nasıl yardımcı olabilirim?"
    ),

    (
        "Selam",
        "Merhaba! Sana nasıl yardımcı olabilirim?"
    ),

    (
        "Selam!",
        "Merhaba! Sana nasıl yardımcı olabilirim?"
    ),

    (
        "Nasılsın?",
        "Ben iyiyim, teşekkür ederim."
    ),

    (
        "Nasılsın",
        "Ben iyiyim, teşekkür ederim."
    ),

    (
        "Sen nasılsın?",
        "Ben iyiyim, teşekkür ederim."
    ),

    (
        "Senin adın ne?",
        "Benim adım Emitt."
    ),

    (
        "Adın ne?",
        "Benim adım Emitt."
    ),

    (
        "Sen kimsin?",
        "Ben Emitt yapay zekasıyım."
    ),

    (
        "Kimsin?",
        "Ben Emitt yapay zekasıyım."
    ),

    (
        "Senin ismin ne?",
        "Benim adım Emitt."
    ),

    (
        "Benim adım ne?",
        "Senin adını henüz bilmiyorum."
    ),

    (
        "Türkiye'nin başkenti nedir?",
        "Türkiye'nin başkenti Ankara'dır."
    ),

    (
        "Türkiye'nin başkenti neresi?",
        "Türkiye'nin başkenti Ankara'dır."
    ),

    (
        "Ankara hangi ülkenin başkentidir?",
        "Ankara Türkiye'nin başkentidir."
    ),

    (
        "Türkiye bir ülke mi?",
        "Evet, Türkiye bir ülkedir."
    ),

    (
        "Ankara nedir?",
        "Ankara Türkiye'nin başkentidir."
    ),

    (
        "Ankara nerede?",
        "Ankara Türkiye'de bulunan bir şehirdir."
    ),

    (
        "Dünya nedir?",
        "Dünya bir gezegendir."
    ),

    (
        "Dünya bir gezegen mi?",
        "Evet, Dünya bir gezegendir."
    ),

    (
        "Dünya'nın şekli nedir?",
        "Dünya yaklaşık olarak küresel bir şekle sahiptir."
    ),

    (
        "Dünya neyin etrafında döner?",
        "Dünya Güneş'in etrafında döner."
    ),

    (
        "Dünya Güneş'in etrafında döner mi?",
        "Evet, Dünya Güneş'in etrafında döner."
    ),

    (
        "Güneş nedir?",
        "Güneş bir yıldızdır."
    ),

    (
        "Güneş bir yıldız mı?",
        "Evet, Güneş bir yıldızdır."
    ),

    (
        "Python nedir?",
        "Python bir programlama dilidir."
    ),

    (
        "Python ne işe yarar?",
        "Python ile programlar geliştirilebilir."
    ),

    (
        "Python bir programlama dili mi?",
        "Evet, Python bir programlama dilidir."
    ),

    (
        "Emitt hangi dille geliştiriliyor?",
        "Emitt Python ile geliştiriliyor."
    ),

    (
        "Emitt nasıl geliştiriliyor?",
        "Emitt Python ve NumPy kullanılarak geliştiriliyor."
    ),

    (
        "Yapay zeka nedir?",
        "Yapay zeka bilgisayarların öğrenmesini ve görevleri gerçekleştirmesini sağlayan bir alandır."
    ),

    (
        "Yapay zeka ne işe yarar?",
        "Yapay zeka bilgisayarların çeşitli görevleri öğrenmesine yardımcı olur."
    ),

    (
        "Su kaç derecede kaynar?",
        "Su normal şartlarda 100 derecede kaynar."
    ),

    (
        "Su ne zaman kaynar?",
        "Su normal şartlarda 100 derecede kaynar."
    ),

    (
        "Teşekkür ederim.",
        "Rica ederim."
    ),

    (
        "Teşekkürler.",
        "Rica ederim."
    ),

    (
        "Sağ ol.",
        "Rica ederim."
    ),

    (
        "Görüşürüz.",
        "Görüşürüz! İyi günler."
    ),

    (
        "Güle güle.",
        "Görüşürüz! Kendine iyi bak."
    ),

    (
        "İyi günler.",
        "İyi günler!"
    ),

    (
        "İyi geceler.",
        "İyi geceler!"
    ),

    (
        "Bana yardım eder misin?",
        "Elbette, sana yardımcı olmaya çalışabilirim."
    ),

    (
        "Bana yardımcı olabilir misin?",
        "Elbette, sana yardımcı olmaya çalışabilirim."
    ),

    (
        "Ne yapabilirsin?",
        "Sorularını cevaplamaya ve bildiğim konuları açıklamaya çalışabilirim."
    ),

    (
        "Ne biliyorsun?",
        "Eğitimimde bulunan bilgiler hakkında cevap vermeye çalışabilirim."
    ),

    (
        "Bir şey sorabilir miyim?",
        "Elbette, sorunu sorabilirsin."
    ),

    (
        "Sana bir soru soracağım.",
        "Tabii, sorunu dinliyorum."
    ),

    (
        "Hazır mısın?",
        "Evet, hazırım."
    ),

    (
        "Emitt hazır mı?",
        "Evet, Emitt hazır."
    ),

]


# ============================================================
# MODEL AYARLARI
# ============================================================

MAX_VOCABULARY_SIZE = 10000

EMBEDDING_SIZE = 64
HIDDEN_SIZE = 128
CONTEXT_SIZE = 32

LEARNING_RATE = 0.005

# V2: 6 epoch
EPOCHS = 6

MIN_FREQUENCY = 1

RANDOM_SEED = 42

# Corpus'un tamamını her epoch'ta eğitmek
# mevcut NumPy modelinde çok yavaş olabilir.
MAX_CORPUS_EXAMPLES = 5000


# ============================================================
# CORPUS OKU
# ============================================================

def load_corpus():

    if not os.path.exists(
        CORPUS_PATH
    ):

        print(
            "Corpus bulunamadı:"
        )

        print(
            CORPUS_PATH
        )

        print(
            "Sadece temel konuşmalar kullanılacak."
        )

        return []

    try:

        with open(
            CORPUS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

    except Exception as error:

        print(
            "Corpus okuma hatası:",
            error
        )

        return []

    if not text.strip():

        print(
            "Corpus boş."
        )

        return []

    print(
        "Corpus yüklendi."
    )

    print(
        "Corpus karakter:",
        len(text)
    )

    return [
        text
    ]


# ============================================================
# KONUŞMA EĞİTİM ÖRNEKLERİ
# ============================================================

def create_training_examples(
    tokenizer,
    conversations
):

    examples = []

    bos_id = tokenizer.token_to_id[
        "<BOS>"
    ]

    eos_id = tokenizer.token_to_id[
        "<EOS>"
    ]

    for user_text, assistant_text in conversations:

        user_ids = tokenizer.encode(
            user_text,
            add_special_tokens=False
        )

        assistant_ids = tokenizer.encode(
            assistant_text,
            add_special_tokens=False
        )

        if not user_ids:
            continue

        if not assistant_ids:
            continue

        context = [
            bos_id
        ]

        context.extend(
            user_ids
        )

        for target_id in assistant_ids:

            examples.append(
                (
                    context.copy(),
                    target_id
                )
            )

            context.append(
                target_id
            )

        examples.append(
            (
                context.copy(),
                eos_id
            )
        )

    return examples


# ============================================================
# CORPUS EĞİTİM ÖRNEKLERİ
# ============================================================

def create_corpus_examples(
    tokenizer,
    corpus_text
):

    examples = []

    if not corpus_text.strip():
        return examples

    token_ids = tokenizer.encode(
        corpus_text,
        add_special_tokens=False
    )

    if len(token_ids) < 2:
        return examples

    bos_id = tokenizer.token_to_id[
        "<BOS>"
    ]

    eos_id = tokenizer.token_to_id[
        "<EOS>"
    ]

    context = [
        bos_id
    ]

    for token_id in token_ids:

        examples.append(
            (
                context[-CONTEXT_SIZE:].copy(),
                token_id
            )
        )

        context.append(
            token_id
        )

    examples.append(
        (
            context[-CONTEXT_SIZE:].copy(),
            eos_id
        )
    )

    return examples


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=================================")
    print("       EMITT TRAINING V2")
    print("=================================")
    print()

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    # ========================================================
    # CORPUS
    # ========================================================

    corpus_texts = load_corpus()

    # ========================================================
    # TÜM METİNLER
    # ========================================================

    texts = []

    for user_text, assistant_text in CONVERSATIONS:

        texts.append(
            user_text
        )

        texts.append(
            assistant_text
        )

    texts.extend(
        corpus_texts
    )

    print()

    print(
        "Temel konuşma sayısı:",
        len(CONVERSATIONS)
    )

    print(
        "Corpus sayısı:",
        len(corpus_texts)
    )

    print(
        "Toplam metin:",
        len(texts)
    )

    print()

    # ========================================================
    # TOKENIZER
    # ========================================================

    print(
        "Vocabulary oluşturuluyor..."
    )

    tokenizer = EmittTokenizer(
        max_vocabulary_size=
            MAX_VOCABULARY_SIZE
    )

    tokenizer.build_vocabulary(
        texts,
        min_frequency=
            MIN_FREQUENCY
    )

    print(
        "Vocabulary size:",
        tokenizer.vocabulary_size
    )

    print(
        "Vocabulary limit:",
        MAX_VOCABULARY_SIZE
    )

    # ========================================================
    # KONUŞMA EĞİTİM ÖRNEKLERİ
    # ========================================================

    print()
    print(
        "Eğitim örnekleri oluşturuluyor..."
    )

    training_examples = create_training_examples(
        tokenizer,
        CONVERSATIONS
    )

    # ========================================================
    # CORPUS EĞİTİM ÖRNEKLERİ
    # ========================================================

    for corpus_text in corpus_texts:

        corpus_examples = create_corpus_examples(
            tokenizer,
            corpus_text
        )

        original_count = len(
            corpus_examples
        )

        if (
            len(corpus_examples)
            > MAX_CORPUS_EXAMPLES
        ):

            corpus_examples = corpus_examples[
                :MAX_CORPUS_EXAMPLES
            ]

        print(
            "Corpus örnekleri:",
            original_count
        )

        print(
            "Kullanılan corpus örnekleri:",
            len(corpus_examples)
        )

        training_examples.extend(
            corpus_examples
        )

    print(
        "Toplam eğitim örneği:",
        len(training_examples)
    )

    if not training_examples:

        print(
            "HATA: Eğitim örneği oluşturulamadı."
        )

        return

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Model oluşturuluyor..."
    )

    model = EmittModel(
        vocabulary_size=
            tokenizer.vocabulary_size,

        embedding_size=
            EMBEDDING_SIZE,

        hidden_size=
            HIDDEN_SIZE,

        context_size=
            CONTEXT_SIZE
    )

    print(
        "Vocabulary:",
        tokenizer.vocabulary_size
    )

    print(
        "Embedding:",
        EMBEDDING_SIZE
    )

    print(
        "Hidden:",
        HIDDEN_SIZE
    )

    print(
        "Context:",
        CONTEXT_SIZE
    )

    # ========================================================
    # EĞİTİM
    # ========================================================

    print()
    print(
        "Eğitim başlıyor..."
    )

    print(
        "Epoch sayısı:",
        EPOCHS
    )

    print(
        "Maksimum corpus örneği:",
        MAX_CORPUS_EXAMPLES
    )

    print()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        random.shuffle(
            training_examples
        )

        total_loss = 0.0

        example_count = 0

        for input_ids, target_id in training_examples:

            loss = model.train_step(
                input_ids=
                    input_ids,

                target_id=
                    target_id,

                learning_rate=
                    LEARNING_RATE
            )

            total_loss += loss

            example_count += 1

        average_loss = (
            total_loss /
            max(
                example_count,
                1
            )
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Loss: {average_loss:.6f}"
        )

    # ========================================================
    # DOSYA YOLLARI
    # ========================================================

    vocabulary_path = os.path.join(
        DATA_DIR,
        "vocabulary.json"
    )

    model_path = os.path.join(
        DATA_DIR,
        "emitt_model.json"
    )

    # ========================================================
    # KAYDET
    # ========================================================

    print()
    print(
        "Dosyalar kaydediliyor..."
    )

    tokenizer.save(
        vocabulary_path
    )

    model.save(
        model_path
    )

    # ========================================================
    # SONUÇ
    # ========================================================

    print()
    print("=================================")
    print("       EĞİTİM TAMAMLANDI")
    print("=================================")
    print()

    print(
        "Konuşma sayısı:",
        len(CONVERSATIONS)
    )

    print(
        "Corpus:",
        len(corpus_texts)
    )

    print(
        "Vocabulary:",
        tokenizer.vocabulary_size
    )

    print(
        "Eğitim örneği:",
        len(training_examples)
    )

    print(
        "Epoch:",
        EPOCHS
    )

    print(
        "Vocabulary:",
        vocabulary_path
    )

    print(
        "Model:",
        model_path
    )

    print()

    print(
        "Emitt V2 modeli kaydedildi."
    )

    print()


if __name__ == "__main__":

    main()