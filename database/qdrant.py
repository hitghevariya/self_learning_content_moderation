from gensim.models import Word2Vec
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance , PointStruct
from config import Qdrant_api_key,Qdrant_url


custom_sentences = [
    "abuse","abusive","ass","asshole","asswipe","asslicker","assface","asshat","assclown",
    "bastard","bitch","bitchy","bloody","bullshit","bully","bigot","bonehead",
    "cock","cockhead","cockroach","crap","crappy","creep","creepy","coward","corrupt",
    "damn","damned","damnation","dick","dickhead","dickface","dirtbag","dimwit","degenerate",
    "evil","exploit","exploiter","extremist",
    "fake","filthy","fool","freak","fraud","fraudster","foul",
    "garbage","gaylord","goddamn","greedy","gross","goon","grubby",
    "hack","hacker","hateful","hate","hater","hell","hellish","hogwash","hypocrite","hooligan",
    "idiot","idiotic","imbecile","immoral","incompetent","insane","ignorant","inferior",
    "jackass","jerk","jerkoff","junkie","joke",
    "kill","killer","killing","klutz",
    "liar","lousy","loser","lunatic","leech","lowlife",
    "mad","maniac","mess","moron","motherfucker","malicious","menace","mongrel",
    "nasty","negligent","nonsense","nutcase","narcissist","nitwit",
    "obscene","offender","offensive","outlaw","oppressor",
    "pathetic","pervert","pig","pimp","piss","pissed","poor","psycho","parasite","perv",
    "racist","rascal","reject","retard","rogue","rude","rat","rotten","ruthless",
    "scam","scammer","scoundrel","screw","scum","scumbag","selfish","shit","shithead",
    "shitty","sick","sleazy","slime","slut","snake","stupid","sucker","swine","savage",
    "terrible","thief","thug","toxic","trash","trashy","troll","traitor","twisted",
    "ugly","unclean","unethical","uncivilized","unreliable","useless",
    "vandal","violent","villain","vulgar",
    "waste","weak","weirdo","wicked","worthless","wreck",
    "yahoo","yob","zombie","zero"

]
from gensim.utils import simple_preprocess

tokenized_sentences = [simple_preprocess(s) for s in custom_sentences]


model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)
print("idiot" in model.wv)        
print("disgusting" in model.wv)  

model.save("custom_word2vec.model")
bad_words = [
    "abuse","abusive","ass","asshole","asswipe","asslicker","assface","asshat","assclown",
    "bastard","bitch","bitchy","bloody","bullshit","bully","bigot","bonehead",
    "cock","cockhead","cockroach","crap","crappy","creep","creepy","coward","corrupt",
    "damn","damned","damnation","dick","dickhead","dickface","dirtbag","dimwit","degenerate",
    "evil","exploit","exploiter","extremist",
    "fake","filthy","fool","freak","fraud","fraudster","foul",
    "garbage","gaylord","goddamn","greedy","gross","goon","grubby",
    "hack","hacker","hateful","hate","hater","hell","hellish","hogwash","hypocrite","hooligan",
    "idiot","idiotic","imbecile","immoral","incompetent","insane","ignorant","inferior",
    "jackass","jerk","jerkoff","junkie","joke",
    "kill","killer","killing","klutz",
    "liar","lousy","loser","lunatic","leech","lowlife",
    "mad","maniac","mess","moron","motherfucker","malicious","menace","mongrel",
    "nasty","negligent","nonsense","nutcase","narcissist","nitwit",
    "obscene","offender","offensive","outlaw","oppressor",
    "pathetic","pervert","pig","pimp","piss","pissed","poor","psycho","parasite","perv",
    "racist","rascal","reject","retard","rogue","rude","rat","rotten","ruthless",
    "scam","scammer","scoundrel","screw","scum","scumbag","selfish","shit","shithead",
    "shitty","sick","sleazy","slime","slut","snake","stupid","sucker","swine","savage",
    "terrible","thief","thug","toxic","trash","trashy","troll","traitor","twisted",
    "ugly","unclean","unethical","uncivilized","unreliable","useless",
    "vandal","violent","villain","vulgar",
    "waste","weak","weirdo","wicked","worthless","wreck",
    "yahoo","yob","zombie","zero"
]


vectors = []

for word in bad_words:
    if word in model.wv:
        vectors.append((word, model.wv[word]))
    else:
        print(f"Missing word: {word}")


qdrant_client = QdrantClient(
    url=Qdrant_url, 
    api_key=Qdrant_api_key,
)

print(qdrant_client.get_collections())

points = []
start_id = 0

for i, (word, vector) in enumerate(vectors):
    points.append(
        PointStruct(
            id=start_id + i,
            vector=vector.tolist(),
            payload={
                "word": word,
                "source": "custom_word2vec"
            }
        )
    )

if not points:
    raise ValueError(" No vectors to upsert")

qdrant_client.upsert(
    collection_name="bad_words",
    points=points
)

print(" Word2Vec vectors stored in Qdrant")


