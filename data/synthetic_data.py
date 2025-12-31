import random

def generate_dataset(size=100000):
    # CLEAN — neutral
    clean_neutral = [
        "have a nice day",
        "just chilling",
        "weekend mood",
        "working hard",
        "daily routine",
        "life goes on"
    ]

    # CLEAN — positive emotion (VERY IMPORTANT)
    clean_positive = [
        "happy ready friend",
        "great vibes only",
        "life is beautiful",
        "feeling amazing today",
        "good times with friends",
        "love this journey",
        "peaceful calm vibes",
        "proud of myself",
        "positive mindset",
        "smiling all day"
    ]

    # CLEAN — profanity but positive (CRITICAL FIX)
    clean_positive_profanity = [
        "fucking awesome bro",
        "damn good job",
        "kick ass performance",
        "badass vibes today",
        "you are fucking amazing"
    ]

    # ABUSIVE — direct
    abusive_direct = [
        "you are an idiot",
        "fucking idiot",
        "you are stupid",
        "hate you so much",
        "you are disgusting",
        "bloody asshole",
        "shut up idiot",
        "fuck you"
    ]

    # ABUSIVE — strong tone
    abusive_strong = [
        "go to hell",
        "what a dumb person",
        "you are a loser",
        "nobody likes you",
        "you are trash"
    ]
# ABUSIVE — high density (MISSING RIGHT NOW)
    abusive_dense = [
        "fed up with stupid assholes everywhere",
        "so many fucking idiots around",
        "this place is full of dumb assholes",
        "tired of stupid morons all the time",
        "nothing but idiots and assholes here",
        "hate these dumb fucking people",
        "everywhere stupid idiots and trash",
    ]
    # ABUSIVE — indirect / generalized (CRITICAL)
    abusive_indirect = [
        "fed up with stupid assholes everywhere",
        "this place is full of idiots",
        "so many dumb people around",
        "tired of these stupid morons",
        "nothing but trash people everywhere",
        "hate dealing with idiots all day",
        "sick of dumb assholes",
    ]

    dataset = []

    for _ in range(size):
        r = random.random()

        if r < 0.35:
            caption = random.choice(clean_neutral)
            label = 0
        elif r < 0.65:
            caption = random.choice(clean_positive)
            label = 0
        elif r < 0.75:
            caption = random.choice(clean_positive_profanity)
            label = 0
        elif r < 0.85:
            caption = random.choice(abusive_direct + abusive_strong)
            label = 1
        else:
            caption = random.choice(abusive_indirect)
            label = 1

        dataset.append((caption, label))

    return dataset
import pandas as pd

df = pd.read_csv(r"C:\Users\Dioneapps\Desktop\hit\embeddings\data\synthetic_abuse_dataset2.csv")
print(df.columns.tolist())
print(df.head(3))
