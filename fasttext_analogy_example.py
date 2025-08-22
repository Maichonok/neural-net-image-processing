from gensim.models import KeyedVectors

model_ft = KeyedVectors.load_word2vec_format(
    "./models/cc.en.300.vec.gz",
    binary=False,
    limit=50000
)

# Compute the analogy "prince" - "man" + "woman"
res = model_ft.most_similar(
    positive=["woman", "prince"],
    negative=["man"]
)

print("prince - man + woman →", res[0])
