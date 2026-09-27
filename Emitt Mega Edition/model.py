import json
import numpy as np


class EmittModel:

    def __init__(
        self,
        vocabulary_size,
        embedding_size=64,
        hidden_size=128,
        context_size=32
    ):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size

        # Token embeddingleri
        self.embeddings = (
            np.random.randn(
                vocabulary_size,
                embedding_size
            ) * 0.02
        )

        # Her pozisyon için ayrı ağırlık.
        # Böylece token sırası korunur.
        self.position_weights = (
            np.random.randn(
                context_size,
                embedding_size
            ) * 0.02
        )

        # Gizli katman
        self.W1 = (
            np.random.randn(
                embedding_size,
                hidden_size
            ) * np.sqrt(
                2.0 / embedding_size
            )
        )

        self.b1 = np.zeros(
            hidden_size,
            dtype=np.float64
        )

        # Çıkış katmanı
        self.W2 = (
            np.random.randn(
                hidden_size,
                vocabulary_size
            ) * np.sqrt(
                2.0 / hidden_size
            )
        )

        self.b2 = np.zeros(
            vocabulary_size,
            dtype=np.float64
        )

    # ========================================================
    # AKTİVASYON
    # ========================================================

    def relu(self, x):
        return np.maximum(
            0.0,
            x
        )

    def relu_derivative(self, x):
        return (
            x > 0
        ).astype(
            np.float64
        )

    # ========================================================
    # SOFTMAX
    # ========================================================

    def softmax(self, logits):

        logits = (
            logits -
            np.max(logits)
        )

        exp_values = np.exp(
            logits
        )

        total = np.sum(
            exp_values
        )

        if total <= 0:
            return (
                np.ones_like(
                    logits
                ) /
                len(logits)
            )

        return (
            exp_values /
            total
        )

    # ========================================================
    # CONTEXT HAZIRLAMA
    # ========================================================

    def prepare_context(self, token_ids):

        if not token_ids:
            return (
                np.zeros(
                    self.embedding_size,
                    dtype=np.float64
                ),
                [],
                []
            )

        token_ids = np.asarray(
            token_ids,
            dtype=np.int64
        )

        token_ids = token_ids[
            -self.context_size:
        ]

        vectors = self.embeddings[
            token_ids
        ].copy()

        actual_length = len(
            token_ids
        )

        start_position = (
            self.context_size -
            actual_length
        )

        combined = np.zeros(
            self.embedding_size,
            dtype=np.float64
        )

        for i in range(
            actual_length
        ):

            position = (
                start_position +
                i
            )

            combined += (
                vectors[i] *
                self.position_weights[position]
            )

        combined /= max(
            actual_length,
            1
        )

        return (
            combined,
            token_ids,
            list(
                range(
                    start_position,
                    self.context_size
                )
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, token_ids):

        x, _, _ = self.prepare_context(
            token_ids
        )

        hidden_pre = (
            x @ self.W1
        ) + self.b1

        hidden = self.relu(
            hidden_pre
        )

        logits = (
            hidden @ self.W2
        ) + self.b2

        return logits

    # ========================================================
    # PROBABILITIES
    # ========================================================

    def probabilities(self, token_ids):

        logits = self.forward(
            token_ids
        )

        return self.softmax(
            logits
        )

    # ========================================================
    # TAHMİN
    # ========================================================

    def predict_next(self, token_ids):

        probabilities = self.probabilities(
            token_ids
        )

        return int(
            np.argmax(
                probabilities
            )
        )

    # ========================================================
    # EĞİTİM
    # ========================================================

    def train_step(
        self,
        input_ids,
        target_id,
        learning_rate=0.005
    ):

        if not input_ids:
            return 0.0

        (
            x,
            token_ids,
            positions
        ) = self.prepare_context(
            input_ids
        )

        hidden_pre = (
            x @ self.W1
        ) + self.b1

        hidden = self.relu(
            hidden_pre
        )

        logits = (
            hidden @ self.W2
        ) + self.b2

        probabilities = self.softmax(
            logits
        )

        target_probability = max(
            probabilities[target_id],
            1e-12
        )

        loss = -np.log(
            target_probability
        )

        # ----------------------------------------------------
        # Çıkış gradyanı
        # ----------------------------------------------------

        d_logits = probabilities.copy()

        d_logits[target_id] -= 1.0

        # ----------------------------------------------------
        # W2 / b2
        # ----------------------------------------------------

        d_W2 = np.outer(
            hidden,
            d_logits
        )

        d_b2 = d_logits.copy()

        # ----------------------------------------------------
        # Gizli katman
        # ----------------------------------------------------

        d_hidden = (
            d_logits @
            self.W2.T
        )

        d_hidden_pre = (
            d_hidden *
            self.relu_derivative(
                hidden_pre
            )
        )

        # ----------------------------------------------------
        # W1 / b1
        # ----------------------------------------------------

        d_W1 = np.outer(
            x,
            d_hidden_pre
        )

        d_b1 = d_hidden_pre.copy()

        # ----------------------------------------------------
        # x gradyanı
        # ----------------------------------------------------

        d_x = (
            d_hidden_pre @
            self.W1.T
        )

        actual_length = len(
            token_ids
        )

        if actual_length > 0:

            d_x /= (
                actual_length
            )

        # ----------------------------------------------------
        # Embedding + position gradyanları
        # ----------------------------------------------------

        d_embedding = (
            d_x
        )

        gradient_limit = 1.0

        d_W1 = np.clip(
            d_W1,
            -gradient_limit,
            gradient_limit
        )

        d_W2 = np.clip(
            d_W2,
            -gradient_limit,
            gradient_limit
        )

        d_b1 = np.clip(
            d_b1,
            -gradient_limit,
            gradient_limit
        )

        d_b2 = np.clip(
            d_b2,
            -gradient_limit,
            gradient_limit
        )

        d_embedding = np.clip(
            d_embedding,
            -gradient_limit,
            gradient_limit
        )

        # ----------------------------------------------------
        # Parametre güncelleme
        # ----------------------------------------------------

        self.W2 -= (
            learning_rate *
            d_W2
        )

        self.b2 -= (
            learning_rate *
            d_b2
        )

        self.W1 -= (
            learning_rate *
            d_W1
        )

        self.b1 -= (
            learning_rate *
            d_b1
        )

        # ----------------------------------------------------
        # Embedding güncelle
        # ----------------------------------------------------

        for index in range(
            actual_length
        ):

            token_id = int(
                token_ids[index]
            )

            position = int(
                positions[index]
            )

            embedding_gradient = (
                d_embedding *
                self.position_weights[position]
            )

            embedding_gradient = np.clip(
                embedding_gradient,
                -gradient_limit,
                gradient_limit
            )

            self.embeddings[
                token_id
            ] -= (
                learning_rate *
                embedding_gradient
            )

            # ------------------------------------------------
            # Position weight güncelle
            # ------------------------------------------------

            position_gradient = (
                d_x *
                self.embeddings[token_id]
            )

            position_gradient = np.clip(
                position_gradient,
                -gradient_limit,
                gradient_limit
            )

            self.position_weights[
                position
            ] -= (
                learning_rate *
                position_gradient
            )

        return float(
            loss
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(self, path):

        data = {
            "vocabulary_size":
                self.vocabulary_size,

            "embedding_size":
                self.embedding_size,

            "hidden_size":
                self.hidden_size,

            "context_size":
                self.context_size,

            "embeddings":
                self.embeddings.tolist(),

            "position_weights":
                self.position_weights.tolist(),

            "W1":
                self.W1.tolist(),

            "b1":
                self.b1.tolist(),

            "W2":
                self.W2.tolist(),

            "b2":
                self.b2.tolist()
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file
            )

    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        path
    ):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        model = cls(
            vocabulary_size=
                data["vocabulary_size"],

            embedding_size=
                data["embedding_size"],

            hidden_size=
                data["hidden_size"],

            context_size=
                data["context_size"]
        )

        model.embeddings = np.array(
            data["embeddings"],
            dtype=np.float64
        )

        # Yeni modellerde bulunacak.
        # Eski model dosyaları için güvenli
        # bir varsayılan oluşturuyoruz.
        if "position_weights" in data:

            model.position_weights = np.array(
                data["position_weights"],
                dtype=np.float64
            )

        model.W1 = np.array(
            data["W1"],
            dtype=np.float64
        )

        model.b1 = np.array(
            data["b1"],
            dtype=np.float64
        )

        model.W2 = np.array(
            data["W2"],
            dtype=np.float64
        )

        model.b2 = np.array(
            data["b2"],
            dtype=np.float64
        )

        return model


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=================================")
    print("       EMITT NUMPY MODEL")
    print("=================================")
    print()

    model = EmittModel(
        vocabulary_size=50,
        embedding_size=32,
        hidden_size=64,
        context_size=32
    )

    test_tokens = [
        2,
        5,
        10
    ]

    probabilities = model.probabilities(
        test_tokens
    )

    print(
        "Vocabulary:",
        model.vocabulary_size
    )

    print(
        "Embedding:",
        model.embedding_size
    )

    print(
        "Hidden:",
        model.hidden_size
    )

    print(
        "Context:",
        model.context_size
    )

    print(
        "Olasılık sayısı:",
        len(probabilities)
    )

    prediction = model.predict_next(
        test_tokens
    )

    print(
        "Tahmin edilen token:",
        prediction
    )

    print()
    print(
        "Model başarıyla çalışıyor."
    )