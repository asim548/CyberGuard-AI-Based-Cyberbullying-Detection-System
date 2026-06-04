"""Generate sample training data when Kaggle CSV is not available."""

import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, DATASET_FILENAME  # noqa: E402

SAFE = [
    "Had a great day at school today!",
    "Love this new song, so uplifting.",
    "Good luck on your exam tomorrow!",
    "Thanks for helping me with homework.",
    "Beautiful weather for a walk in the park.",
    "Proud of our team for winning the match.",
    "Happy birthday! Hope you have an amazing day.",
    "That movie was really entertaining.",
    "Can't wait for the weekend trip.",
    "You did an awesome job on the presentation.",
    "Learning Python is fun and useful.",
    "Grateful for supportive friends and family.",
    "The cafeteria food was surprisingly good today.",
    "Just finished reading an interesting book.",
    "Congrats on your graduation!",
]

TOXIC = [
    "You're so annoying, nobody likes you.",
    "Shut up, nobody cares what you think.",
    "You're pathetic and always mess things up.",
    "Go away, loser.",
    "Stop posting, you're embarrassing yourself.",
    "Nobody wants you here, just leave.",
    "You're weird and creepy.",
    "Why are you even in this group?",
    "You always ruin everything for everyone.",
    "Don't talk to me again, idiot.",
    "You're a joke and everyone laughs at you.",
    "Get a life, seriously.",
    "You're so dumb it's painful.",
    "I can't stand you, you're the worst.",
    "Nobody respects you at all.",
]

SEVERE = [
    "I hope you die, nobody would miss you.",
    "You should kill yourself, the world is better without you.",
    "I'm going to hurt you if I see you at school.",
    "People like you deserve to suffer.",
    "I'll find where you live and make you pay.",
    "You're disgusting because of your religion.",
    "Hate people like you, you don't belong here.",
    "I'll beat you up after class, watch your back.",
    "You're trash because of your race.",
    "I will ruin your life, you worthless piece of garbage.",
    "Girls like you deserve harassment.",
    "I'll spread lies until everyone hates you.",
    "You should disappear forever.",
    "I'm going to make sure you regret being born.",
    "Violence is what you deserve, freak.",
]


def augment(text: str) -> str:
    variants = [
        text,
        text + " lol",
        text + "!!!",
        text.lower(),
        text.upper() if random.random() > 0.7 else text,
        text.replace("you", "u") if random.random() > 0.5 else text,
    ]
    return random.choice(variants)


def main(rows_per_class: int = 400) -> None:
    random.seed(42)
    records = []
    for template in SAFE:
        for _ in range(rows_per_class // len(SAFE) + 1):
            records.append({"text": augment(template), "label": "not_cyberbullying"})
    for template in TOXIC:
        for _ in range(rows_per_class // len(TOXIC) + 1):
            records.append({"text": augment(template), "label": "other_cyberbullying"})
    for i, template in enumerate(SEVERE):
        kaggle_label = ["gender", "religion", "ethnicity", "age"][i % 4]
        for _ in range(rows_per_class // len(SEVERE) + 1):
            records.append({"text": augment(template), "label": kaggle_label})

    random.shuffle(records)
    df = pd.DataFrame(records)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / DATASET_FILENAME
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()
