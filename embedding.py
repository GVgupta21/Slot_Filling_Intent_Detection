from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer('bert-base-nli-mean-tokens')
def get_embeddings(sentence):
       return sbert.encode(sentence)

# x = get_embeddings("Hello")
# print(x)
# print(len(x))
# print(type(x))