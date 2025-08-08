import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import time

from load_data import Load_Data
from seq_embedding import SequenceEmbedder
from word_embedding import WordEmbedder
from aligning import Alignment





BATCH_SIZE = 1
EPOCHS = 20

SEQ_INPUT_DIM = 6
WORD_INPUT_DIM = 30
HIDDEN_DIM = 64
EMBEDDING_DIM = 128
NUM_LAYERS = 2
NUM_DIRECTIONS = 2
DROPOUT = 0.3

LEARNING_RATE = 0.001
SIMILARITY_WEIGHT = 0.7
TEMPERATURE = 0.1
MARGIN = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCABULARY_PATH = '3500.txt'
TEST_PATH =  '3500_test.txt'

ALIGNMENT_PATH = 'Alignment_v2_3500_20250804.pth'

IDX2CHAR = {
    0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 
    3: 'a', 4: 'b', 5: 'c', 6: 'd', 7: 'e', 8: 'f', 9: 'g',
    10: 'h', 11: 'i', 12: 'j', 13: 'k', 14: 'l', 15: 'm', 16: 'n',
    17: 'o', 18: 'p', 19: 'q', 20: 'r', 21: 's', 22: 't', 23: 'u',
    24: 'v', 25: 'w', 26: 'x', 27: 'y', 28: 'z', 29: '\n'
}
CHAR2IDX = {v: k for k, v in IDX2CHAR.items()}





class Predictor():

    def __init__(self, vocabulary_path, alignment_path, device, converter, 
                 seq_input_dim, word_input_dim, hidden_dim, embedding_dim, 
                 num_layers=2, num_directions=2, dropout=0.3, 
                 similarity_weight = 0.7, temperature = 0.1):

        self.vocabulary_path = vocabulary_path
        self.device = device
        self.converter = converter

        self.seq_embedder = SequenceEmbedder(
            seq_input_dim, 
            hidden_dim, 
            embedding_dim, 
            num_layers, 
            num_directions, 
            dropout
        ).to(self.device)

        self.word_embedder = WordEmbedder(
            word_input_dim, 
            hidden_dim, 
            embedding_dim, 
            num_directions, 
            num_directions, 
            dropout
        ).to(self.device)

        self.alignment = Alignment(
            self.seq_embedder, 
            self.word_embedder, 
            embedding_dim, 
            similarity_weight, 
            temperature
        ).to(self.device)

        checkpoint = torch.load(alignment_path, map_location=device)
        self.alignment.load_state_dict(checkpoint)
        self.alignment.eval()
            
        for param in self.alignment.parameters():
            param.requires_grad = False
        
        self.vocabulary_embeddings, self.vocabulary_words = self.precompute_vocabulary()
        # array = torch.tensor(self.vocabulary_embeddings)
        # long_word_indices = [idx for idx, word in enumerate(self.vocabulary_words) if len(word) > 7]
        # array_long = array[long_word_indices] if long_word_indices else torch.tensor([])
        # sum_long = torch.sum(array_long, dim=0)
        # short_word_indices = [idx for idx, word in enumerate(self.vocabulary_words) if len(word) < 6]
        # array_short = array[short_word_indices] if short_word_indices else torch.tensor([])
        # sum_short = torch.sum(array_short, dim=0)
        # print(sum_long, sum_short)
        # print(torch.linalg.norm(sum_long / array_long.size(0)), torch.linalg.norm(sum_short / array_short.size(0)))


    def precompute_vocabulary(self):

        with open(self.vocabulary_path, 'r') as f:
            vocabulary = [line.strip() for line in f.readlines()]
        
        embeddings = []
        words = []

        for word in vocabulary:

            char_indice = [self.converter[char] for char in word]
            
            word_tensor = torch.tensor([char_indice], dtype=torch.long).to(self.device)
            word_len = torch.tensor([len(char_indice)]).to(self.device)
            
            with torch.no_grad():
                word_emb = self.word_embedder(word_tensor, word_len)
                word_proj = self.alignment.word_projector(word_emb)
                
            embeddings.append(word_proj.squeeze().cpu().numpy())
            words.append(word)

        return np.array(embeddings), words
    

    def predict(self, seq, seq_len):
        
        seq = seq.to(self.device)
        seq_len = seq_len.to(self.device)

        with torch.no_grad():
            seq_emb = self.seq_embedder(seq, seq_len)
            seq_proj = self.alignment.seq_projector(seq_emb)
            seq_proj_np = seq_proj.cpu().numpy().squeeze()
         
        similarities = []
        for word_emb in self.vocabulary_embeddings:
            cos_similarity = F.cosine_similarity(
                torch.tensor(seq_proj_np).unsqueeze(0).float(),
                torch.tensor(word_emb).unsqueeze(0).float(),
                dim=1
            )
            similarities.append(cos_similarity.item())
        
        sorted_indices = np.argsort(similarities)[::-1]
        top_idx = sorted_indices[0]

        predicted_word = self.vocabulary_words[top_idx]
        confidence = similarities[top_idx]
        
        top_candidates = []
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            top_candidates.append((self.vocabulary_words[idx], similarities[idx], self.vocabulary_embeddings[idx]))
        
        return predicted_word, confidence, top_candidates



class SequenceDataset(Dataset):

    def __init__(self, file_path):
        
        self.load_data = Load_Data(file_path)
        self.data, self.size, _ = self.load_data.v1()
    

    def __len__(self):

        return self.size
    

    def __getitem__(self, idx):
        
        return self.data[idx]



def collate_fn(batch):

    seqs, words = zip(*batch)

    seq_lens = [len(seq) for seq in seqs]
    max_len = max(seq_lens)
            
    padded_seqs = torch.zeros(len(seqs), max_len, SEQ_INPUT_DIM, dtype=torch.float32)
    for i, seq in enumerate(seqs):
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        padded_seqs[i, :len(seq)] = seq_tensor
    
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long)

    return padded_seqs, seq_lens_tensor, words



if __name__ == '__main__':

    test_dataset = SequenceDataset(TEST_PATH)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    predictor = Predictor(
        VOCABULARY_PATH, 
        ALIGNMENT_PATH, 
        device, 
        CHAR2IDX, 
        SEQ_INPUT_DIM,
        WORD_INPUT_DIM,
        HIDDEN_DIM, 
        EMBEDDING_DIM, 
        NUM_LAYERS, 
        NUM_DIRECTIONS, 
        DROPOUT, 
        SIMILARITY_WEIGHT, 
        TEMPERATURE
    )

    correct = 0
    time_start = time.time()

    for idx, (seq, seq_len, word) in enumerate(test_loader):

        predicted_word, confidence, candidates = predictor.predict(seq, seq_len)
        actual_word = word[0]
        if predicted_word == actual_word:
            correct += 1

        print(f"  Sample {idx+1}:")
        print(f"  Actual: {actual_word}")
        print(f"  Predicted: {predicted_word} --Confidence: {confidence}")
        for i, (candidate, confidence, embedding) in enumerate(candidates):
            print(f"    {i+1}. {candidate} (confidence: {confidence:.4f}) (norm: {np.linalg.norm(embedding)})")

        print("-" * 50)
    
    time_end = time.time()
    time_duration = time_end - time_start
    print(time_duration)
    accuracy = correct / len(test_loader)
    print(f'Accuracy: {accuracy}')