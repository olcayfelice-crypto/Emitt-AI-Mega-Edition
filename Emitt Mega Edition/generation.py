import os

from tokenizer import EmittTokenizer
from model import EmittModel


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VOCABULARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "vocabulary.json"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "emitt_model.json"
)

BRAINFUCK_FILE = os.path.join(
    BASE_DIR,
    "oneadd.bf"
)


# ============================================================
# BRAINFUCK
# ============================================================

def run_brainfuck(code, input_text=""):
    tape = [0] * 30000

    pointer = 0
    code_pointer = 0
    input_pointer = 0

    output = []

    brackets = {}
    stack = []

    for i, char in enumerate(code):
        if char == "[":
            stack.append(i)

        elif char == "]":
            if not stack:
                raise ValueError(
                    "Brainfuck: eşleşmeyen ]"
                )

            start = stack.pop()

            brackets[start] = i
            brackets[i] = start

    if stack:
        raise ValueError(
            "Brainfuck: eşleşmeyen ["
        )

    while code_pointer < len(code):

        command = code[code_pointer]

        if command == ">":
            pointer += 1

            if pointer >= len(tape):
                pointer = 0

        elif command == "<":
            pointer -= 1

            if pointer < 0:
                pointer = len(tape) - 1

        elif command == "+":
            tape[pointer] = (
                tape[pointer] + 1
            ) % 256

        elif command == "-":
            tape[pointer] = (
                tape[pointer] - 1
            ) % 256

        elif command == ".":
            output.append(
                chr(tape[pointer])
            )

        elif command == ",":
            if input_pointer < len(input_text):
                tape[pointer] = (
                    ord(input_text[input_pointer])
                    % 256
                )
                input_pointer += 1
            else:
                tape[pointer] = 0

        elif command == "[":
            if tape[pointer] == 0:
                code_pointer = brackets[
                    code_pointer
                ]

        elif command == "]":
            if tape[pointer] != 0:
                code_pointer = brackets[
                    code_pointer
                ]

        code_pointer += 1

    return "".join(output)


def load_brainfuck():
    if not os.path.exists(BRAINFUCK_FILE):
        return None

    with open(
        BRAINFUCK_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()


# ============================================================
# EMITT
# ============================================================

def load_emitt():

    print("Tokenizer yükleniyor...")

    tokenizer = EmittTokenizer()

    tokenizer.load(
        VOCABULARY_FILE
    )

    print("Model yükleniyor...")

    model = EmittModel.load(
        MODEL_FILE
    )

    return tokenizer, model


def generate(
    tokenizer,
    model,
    text,
    length=12
):

    tokens = tokenizer.encode(text)

    if not tokens:
        tokens = [
            tokenizer.token_to_id["<UNK>"]
        ]

    generated = tokens.copy()

    for _ in range(length):

        next_token = (
            model.predict_next(
                generated
            )
        )

        generated.append(
            next_token
        )

    new_tokens = generated[
        len(tokens):
    ]

    return tokenizer.decode(
        new_tokens
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("================================")
    print("        EMITT GENERATION")
    print("================================")

    if not os.path.exists(
        VOCABULARY_FILE
    ):
        print(
            "Vocabulary bulunamadı."
        )
        print(
            "Önce training.py çalıştır."
        )
        return

    if not os.path.exists(
        MODEL_FILE
    ):
        print(
            "Model bulunamadı."
        )
        print(
            "Önce training.py çalıştır."
        )
        return

    tokenizer, model = load_emitt()

    print()

    brainfuck_code = load_brainfuck()

    if brainfuck_code is not None:

        print(
            "Brainfuck modülü yüklendi."
        )

        try:
            brainfuck_output = (
                run_brainfuck(
                    brainfuck_code
                )
            )

            if brainfuck_output:
                print(
                    "Brainfuck:",
                    brainfuck_output
                )

        except Exception as error:

            print(
                "Brainfuck hatası:",
                error
            )

    else:

        print(
            "Brainfuck modülü bulunamadı."
        )

    print()
    print("Emitt hazır.")
    print("Çıkmak için: exit")
    print()

    while True:

        text = input("Sen: ")

        if text.lower() == "exit":
            print(
                "Emitt kapatılıyor."
            )
            break

        if not text.strip():
            continue

        answer = generate(
            tokenizer,
            model,
            text
        )

        print(
            "Emitt:",
            answer
        )


if __name__ == "__main__":
    main()