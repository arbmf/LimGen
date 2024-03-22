

from transformers import BertTokenizer
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# Load the spaCy model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlp = spacy.load("en_core_web_sm")



tokens_for_sentence = {}
def tokenize(text):
    """
    Tokenize the text using spaCy and return a list of tokens.
    """
    if text not in tokens_for_sentence:
        tokens_for_sentence[text] = tokenizer.tokenize(text)

    return tokens_for_sentence[text]

def split_into_sentences(text):
    """
    Splits the text into sentences using spaCy's sentence boundary detection.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def estimate_parts(tokens, max_length=600):
    """
    Estimate the number of parts needed based on the total token count.
    """
    return max(1, -(-len(tokens) // max_length))

def get_sentence_from_token(sentence_tokens):
    token_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    decoded_text = tokenizer.decode(token_ids)
    return decoded_text


def split_paragraph(paragraph, max_length=600, last=False):
    """
    Split a paragraph into multiple parts, each as close to equal length as possible,
    without exceeding max_length tokens, and breaking at sentence ends.
    """
    sentences = split_into_sentences(paragraph)
    all_tokens = [token for sentence in sentences for token in tokenize(sentence)]
    if(len(all_tokens)<256 and not last):
        return None
    num_parts = estimate_parts(all_tokens, max_length)

    parts = []
    current_part_tokens = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = tokenize(sentence)

        if current_token_count + len(sentence_tokens) > (len(all_tokens) // num_parts):
            parts.append(current_part_tokens)
            current_part_tokens = []
            current_part_tokens.append(get_sentence_from_token(sentence_tokens))
            current_token_count = len(sentence_tokens)
        else:
            current_part_tokens.append(get_sentence_from_token(sentence_tokens))
            current_token_count += len(sentence_tokens)
    # Add the last part if it's not empty
    if current_part_tokens and (not parts or parts[-1] != current_part_tokens):
        parts.append(current_part_tokens)
    
    return parts

    
def process_text(text):
    """
    Process the entire text, splitting it into paragraphs and further splitting each paragraph.
    """
    paragraphs = text.split('\n')
    processed_paragraphs = []
    last = False
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
        if index == len(paragraphs) - 1:
            last = True
        processed_paragraph = split_paragraph(paragraph,600,last)
        if processed_paragraph is None:
            paragraphs[index+1] = paragraph + "\n" + paragraphs[index+1]
            continue
        processed_paragraphs.extend(processed_paragraph)
    return processed_paragraphs




import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)


def encode_texts(model, texts):
    """
    Encode a list of texts using the provided model.

    :param model: Loaded model.
    :param texts: List of texts (sentences or passages) to encode.
    :return: List of encoded embeddings.
    """
    return model.encode(texts)

def find_most_relevant_passage_sentence_level(model, input_sentence, passages):
    """
    Find the most relevant passage for the given sentence, comparing at the sentence level.

    :param model: Loaded model.
    :param input_sentence: Input sentence for which to find relevant passage.
    :param passages: List of passages, each being a list of sentences.
    :return: Most relevant passage.
    """
    sentence_embedding = encode_texts(model, [input_sentence])

    highest_similarity = -1
    second_highest_similarity = -1  # Initialize to a low value
    third_highest_similarity = -1
    
    
    most_relevant_passage_index = -1
    second_most_relevant_passage_index = -1
    third_most_relevant_passage_index = -1
    
    all_passages = []
    
    # Iterate over each passage
    for i, passage in enumerate(passages):
        if passage in all_passages:
            print("------------------duplicate-----------------")
            continue
        all_passages.append(passage)
        
        # Encode each sentence in the passage
        passage_embeddings = encode_texts(model, passage)

        # Calculate similarities for each sentence in the passage
        similarities = cosine_similarity(sentence_embedding, passage_embeddings)

        # Find the highest similarity score in this passage
        max_similarity_in_passage = similarities.max()
        # Check if this passage contains the most similar sentence so far
        if max_similarity_in_passage > highest_similarity:
            third_highest_similarity = second_highest_similarity
            second_highest_similarity = highest_similarity
            highest_similarity = max_similarity_in_passage
    
            third_most_relevant_passage_index = second_most_relevant_passage_index
            second_most_relevant_passage_index = most_relevant_passage_index
            most_relevant_passage_index = i
            
        elif max_similarity_in_passage > second_highest_similarity:
            third_highest_similarity = second_highest_similarity
            second_highest_similarity = max_similarity_in_passage
            third_most_relevant_passage_index = second_most_relevant_passage_index
            second_most_relevant_passage_index = i
            
        elif max_similarity_in_passage > third_highest_similarity:
            third_highest_similarity = max_similarity_in_passage
            third_most_relevant_passage_index = i
    similarity_dict = {
        'highest_similarity': highest_similarity,
        'second_highest_similarity': second_highest_similarity,
        'third_highest_similarity': third_highest_similarity,
        'most_relevant_passage': " ".join(passages[most_relevant_passage_index]),
        'second_most_relevant_passage': " ".join(passages[second_most_relevant_passage_index]),
        'third_most_relevant_passage': " ".join(passages[third_most_relevant_passage_index]),
    }
    return similarity_dict



import jsonlines
from tqdm import tqdm

data_dict = []
with jsonlines.open("./train.jsonl", mode='r') as reader:
    for row in tqdm(reader):
        data_dict.append(row)



import csv
from tqdm import tqdm
import sys
csv.field_size_limit(sys.maxsize)
output_file = '/dpr_top_3.jsonl'

for row in tqdm(data_dict):
    processed_text = process_text(row['content'])
    processed_summary = split_into_sentences(row['limitations'])
    for sentence in processed_summary:
        similarities_dict = find_most_relevant_passage_sentence_level(model, sentence, processed_text)
        
        tosave = {"paper_id":row['id'], "limitation_sentence": sentence}
        tosave.update(similarities_dict)
        #append_to_file(output_file)





