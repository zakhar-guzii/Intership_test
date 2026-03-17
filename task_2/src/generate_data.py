import json
import os
import random
import re

from faker import Faker

fake = Faker("en_US")


ANIMALS = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel",
]

ANIMAL_PLURALS = {
    "butterflies": "butterfly",
    "cats":        "cat",
    "chickens":    "chicken",
    "cows":        "cow",
    "dogs":        "dog",
    "elephants":   "elephant",
    "horses":      "horse",
    "sheep":       "sheep", 
    "spiders":     "spider",
    "squirrels":   "squirrel",
}

ALL_ANIMAL_FORMS: set[str] = set(ANIMALS) | set(ANIMAL_PLURALS.keys())

OTHER_ANIMALS = [
    "lion", "tiger", "bear", "wolf", "fox",
    "eagle", "shark", "dolphin", "snake", "frog",
    "parrot", "crocodile", "penguin", "zebra", "giraffe",
]

def make_subjects(a: str) -> list:
    plural = next(
        (p for p, s in ANIMAL_PLURALS.items() if s == a), a + "s"
    )
    return [
        f"The {a}",
        f"A {a}",
        f"An injured {a}",
        f"A wild {a}",
        f"A large {a}",
        f"A tiny {a}",
        f"A young {a}",
        f"That {a}",
        f"The stray {a}",
        f"One {a}",
        # Plural forms
        f"The {plural}",
        f"Several {plural}",
        f"A group of {plural}",
        f"Two {plural}",
        f"These {plural}",
        f"Wild {plural}",
    ]

LOCATIONS = [
    lambda: f"in {fake.city()}",
    lambda: f"near the {random.choice(['river', 'lake', 'forest', 'farm', 'park', 'barn', 'garden', 'field', 'road', 'market'])}",
    lambda: f"on {fake.street_name()}",
    lambda: f"at the {random.choice(['zoo', 'sanctuary', 'shelter', 'farm', 'reserve', 'rescue center', 'vet clinic'])}",
    lambda: f"somewhere in {fake.country()}",
    lambda: f"outside {fake.city()} city limits",
    lambda: f"behind {fake.last_name()}'s property",
    lambda: "",
]

VERBS = [
    "was spotted", "appeared", "escaped", "wandered",
    "was found", "caused a scene", "attracted a crowd",
    "was rescued", "was photographed", "made headlines",
    "surprised locals", "was seen roaming",
    "had been missing for days before being found",
    "was caught on camera", "disrupted traffic",
    "were spotted",        
    "were found roaming",
    "have been reported",
    "are becoming more common",
]

ENDINGS = [
    lambda: f"by {fake.name()} on {fake.date_this_year()}.",
    lambda: f"early {random.choice(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])} morning.",
    lambda: f"according to local authorities.",
    lambda: f"near a residential area in {fake.city()}.",
    lambda: f"which left residents shocked.",
    lambda: f"and was later safely relocated.",
    lambda: f"with no injuries reported.",
    lambda: f"during the annual {fake.bs()} festival.",
    lambda: f"prompting an emergency response.",
    lambda: "",
]

OPENERS = [
    lambda: f"Residents of {fake.city()} were alarmed when",
    lambda: f"Wildlife officials confirmed that",
    lambda: f"According to {fake.name()}, a local farmer,",
    lambda: f"Surveillance footage from {fake.street_name()} shows that",
    lambda: f"In a surprising turn of events,",
    lambda: f"For the third time this month,",
    lambda: f"Nobody expected it, but",
    lambda: f"During a routine inspection,",
    lambda: f"Late last night,",
    lambda: "",
]

INTERNET_CLOSERS = [
    f"@{fake.user_name()} come look at this",
    "someone call the authorities lmao",
    "this is NOT what i expected today",
    "nature is wild",
    "only in this neighborhood",
    "i need to move",
    f"replying to @{fake.user_name()}: yes this is real",
]

NEWS_VERBS = [
    "escapes enclosure",
    "found in suburban backyard",
    "disrupts morning commute",
    "spotted near school",
    "rescued after flood",
    "causes stir at local market",
    "goes viral after being filmed downtown",
]



def gen_news_headline(animal: str) -> str:
    plural = next((p for p, s in ANIMAL_PLURALS.items() if s == animal), animal + "s")
    name   = random.choice([animal, plural])
    casing = random.choice(["normal", "upper", "title"])
    if casing == "upper":
        name = name.upper()
    elif casing == "title":
        name = name.capitalize()
    return f"BREAKING: {name} {random.choice(NEWS_VERBS)} in {fake.city()}."


def gen_structured_sentence(animal: str) -> str:
    plural   = next((p for p, s in ANIMAL_PLURALS.items() if s == animal), animal + "s")
    subjects = make_subjects(animal)
    opener   = random.choice(OPENERS)()
    subject  = random.choice(subjects)
    verb     = random.choice(VERBS)
    loc      = random.choice(LOCATIONS)()
    ending   = random.choice(ENDINGS)()

    casing = random.choice(["normal", "upper", "title", "normal", "normal"])
    if casing == "upper":
        subject = subject.replace(animal, animal.upper()).replace(plural, plural.upper())
    elif casing == "title":
        subject = subject.replace(animal, animal.capitalize()).replace(plural, plural.capitalize())

    parts    = [p for p in [opener, subject, verb, loc, ending] if p]
    sentence = " ".join(parts)
    return re.sub(r" {2,}", " ", sentence).strip()


def gen_internet_post(animal: str) -> str:
    plural  = next((p for p, s in ANIMAL_PLURALS.items() if s == animal), animal + "s")
    name    = random.choice([
        animal.upper(),
        animal,
        plural.upper(),
        plural,
        animal.capitalize(),
    ])
    closer  = random.choice(INTERNET_CLOSERS)
    openers = [
        "omg", "wait WHAT", "i can't believe this",
        "you guys", "absolutely unhinged:", "daily reminder that",
    ]
    mids = [
        f"a {name} just walked past my house",
        f"there's a {name} in the parking lot",
        f"someone left a {name} in the elevator",
        f"i found a {name} in my garden",
        f"a {name} ate my lunch",
        f"my neighbor owns a {name} apparently",
        f"a {name} followed me home",
        f"there are literally {name} outside rn",
        f"TWO {name.upper()} showed up at work today",
    ]
    return f"{random.choice(openers)} {random.choice(mids)}... {closer}"


def gen_dialogue(animal: str) -> str:
    plural   = next((p for p, s in ANIMAL_PLURALS.items() if s == animal), animal + "s")
    name     = random.choice([animal, plural, animal.upper(), animal.capitalize()])
    speaker1 = fake.first_name()
    speaker2 = fake.first_name()
    styles   = [
        f'"{speaker1}, is that a {name}?!" — "{speaker2}: yes, definitely a {name}, call someone!"',
        f'"I swear I saw a {name} near the {random.choice(["school","office","hospital","mall"])}" said {speaker1}.',
        f'"{speaker2} told me the {name} has been there since {fake.time()}," {speaker1} wrote online.',
        f'"{speaker1}, did you seriously let a {name} into the house?" {speaker2} asked.',
        f'"There were at least three {plural} out there," {speaker1} insisted.',
    ]
    return random.choice(styles)


def gen_scientific(animal: str) -> str:
    plural = next((p for p, s in ANIMAL_PLURALS.items() if s == animal), animal + "s")
    templates = [
        f"A study published by {fake.company()} observed that {animal} populations in {fake.country()} declined by {random.randint(5,60)}% over the past decade.",
        f"Dr. {fake.last_name()} confirmed that the {animal} exhibits unusual behavior during winter months.",
        f"Field researchers documented a {animal} traveling over {random.randint(10,300)} km in search of food.",
        f"The {animal}, classified under genus {fake.last_name()}us, remains one of the most studied species in {fake.country()}.",
        f"Populations of {plural} in {fake.country()} have grown significantly since conservation efforts began.",
        f"It was noted that {plural} tend to avoid urban areas, though exceptions have been recorded.",
    ]
    return random.choice(templates)


def gen_list_sentence(animal: str) -> str:
    plural = next((p for p, s in ANIMAL_PLURALS.items() if s == animal), animal + "s")
    name   = random.choice([animal, plural, animal.upper()])
    items  = [name, fake.word(), fake.word(), fake.word()]
    random.shuffle(items)
    return f"Things I saw today: {', '.join(items)}. The {random.choice([animal, plural])} was by far the strangest."


def gen_multi_animal_sentence(animal: str) -> str:
    plural = next((p for p, s in ANIMAL_PLURALS.items() if s == animal), animal + "s")
    other  = random.choice(OTHER_ANIMALS)
    name   = random.choice([animal, plural])
    templates = [
        f"While the {other} stayed hidden, the {name} moved boldly into the open.",
        f"We expected to find a {other}, but it turned out to be a {name}.",
        f"The {name} and the {other} were both spotted near {fake.city()} — experts are baffled.",
        f"Unlike the aggressive {other}, the {name} appeared calm and unbothered.",
        f"The ranger had dealt with {other}s before, but {name} were a first.",
    ]
    return random.choice(templates)


def gen_negation_sentence(animal: str) -> str:
    plural = next((p for p, s in ANIMAL_PLURALS.items() if s == animal), animal + "s")
    name   = random.choice([animal, plural])
    templates = [
        f"It wasn't a {name}, as some initially claimed — just a large {fake.word()}.",
        f"Despite rumors, no {name} was ever found on the premises.",
        f"The footage showed what appeared to be a {name}, but experts disagreed.",
        f"Locals described it as a {name}-like creature, though this remains unconfirmed.",
    ]
    return random.choice(templates)


def gen_negative_sentence() -> str:
    other = random.choice(OTHER_ANIMALS)
    styles = [
        lambda: f"A {other} was spotted near {fake.city()} — wildlife officials are monitoring.",
        lambda: f"{fake.name()} filed a report after finding a {other} in their backyard.",
        lambda: f"The {fake.company()} announced record profits for Q{random.randint(1,4)} {fake.year()}.",
        lambda: f"Traffic on {fake.street_name()} was disrupted due to road construction.",
        lambda: f'"{fake.first_name()}, did you pick up the {fake.word()}?" {fake.name()} texted.',
        lambda: f"omg there was a {other} outside my window... {random.choice(INTERNET_CLOSERS)}",
        lambda: f"Scientists discovered a new species of {other} in {fake.country()}.",
        lambda: f"The weather in {fake.city()} reached {random.randint(-10,45)}°C today.",
        lambda: f"A {fake.color_name()} car was reported stolen on {fake.street_name()} last night.",
        lambda: f"{fake.name()} won the {fake.year()} award for contributions to {fake.bs()}.",
    ]
    return random.choice(styles)()


GENERATORS = [
    gen_news_headline,
    gen_structured_sentence,
    gen_internet_post,
    gen_dialogue,
    gen_scientific,
    gen_list_sentence,
    gen_multi_animal_sentence,
    gen_negation_sentence,
]


def tokenise_and_tag(sentence: str) -> dict:
    """
    Tokenise sentence and assign BIO tags.
    Matching is case-insensitive and covers both singular and plural forms.
    """
    tokens   = re.findall(r"[\w']+|[^\w\s]", sentence)
    ner_tags = []

    for token in tokens:
        token_lower = token.lower()
        if token_lower in ALL_ANIMAL_FORMS:
            ner_tags.append(1)  # B-ANIMAL
        else:
            ner_tags.append(0)  # O

    return {"tokens": tokens, "ner_tags": ner_tags}

def generate_dataset(
    samples_per_class: int = 120,
    negative_samples: int  = 400,
    output_path: str       = "task_2/data/ner_dataset.json",
) -> None:
    dataset = []

    for animal in ANIMALS:
        for _ in range(samples_per_class):
            generator = random.choice(GENERATORS)
            sentence  = generator(animal)
            dataset.append(tokenise_and_tag(sentence))

    for _ in range(negative_samples):
        sentence = gen_negative_sentence()
        tokens   = re.findall(r"[\w']+|[^\w\s]", sentence)
        dataset.append({"tokens": tokens, "ner_tags": [0] * len(tokens)})

    random.shuffle(dataset)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    total     = len(dataset)
    positives = sum(1 for d in dataset if any(t != 0 for t in d["ner_tags"]))
    negatives = total - positives

    print(f"Dataset generated: {total} sentences")
    print(f"  Positive (contains animal): {positives}")
    print(f"  Negative (no target animal): {negatives}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    task_2_dir = os.path.dirname(script_dir)
    final_output_path = os.path.join(task_2_dir, "data", "ner_dataset.json")
    generate_dataset(
        samples_per_class=120,
        negative_samples=400,
        output_path=final_output_path,
    )