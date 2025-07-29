import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

import time
from datetime import datetime

from load_data import Load_Data



class WordEmbedder(nn.Module):

    '''
    Boidirectional GRU

    INPUT  
    src: (batch_size, max_seq_len)  
    src_lens: (batch_size, )

    OUTPUT  
    embedding: (batch_size, embedding_dim)
    '''
    
    def __init__(self, input_dim, hidden_dim, embedding_dim, 
                 num_layers=2, num_directions=2, dropout=0.3):

        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)
    

    def forward(self, src, src_len):

        # self.rnn
        # INPUT
        # src: (batch_size, max_seq_len, hidden_dim)
        # OUTPUT
        # output: (batch_size, max_seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)

        # self.attention
        # hidden_dim * num_directions -> 1

        # self.fc
        # hidden_dim * num_directions -> embedding_dim

        # num_layers = 2
        # num_directions = 1 if 单向 else 2

        embedded = self.embedding(src)
        # (batch_size, max_length, hidden_dim)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        # PackedSequence class
        output, _ = self.rnn(packed)
        # PackedSequence class
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # (batch_size, max_length, hidden_dim * num_directions)
        
        attn_weights = F.softmax(self.attention(output), dim=1)
        # (batch_size, max_length, 1)
        weighted = torch.sum(attn_weights * output, dim=1)
        # (batch_size, hidden_dim * num_directions)
        
        embedding = self.fc(weighted)
        # (batch_size, embedding_dim)

        return embedding # (batch_size, embedding_dim)



class EditDistanceLoss(nn.Module):

    def __init__(self, similarity_weight=0.7, temperature=0.1, margin=3.0):

        """
        EditDistanceLoss

        PARAMETERS    
        temperature (float): 温度参数控制分布锐度  
        margin (float): 边界值用于正负样本分离

        """

        super().__init__()

        self.similarity_weight = similarity_weight
        self.temperature = temperature
        self.margin = margin
    

    def forward(self, embedding, edit_distance):

        """

        INPUT
        
        embedding: (batch_size, embedding_dim) --单词嵌入向量  
        edit_distances: (batch_size, batch_size) --编辑距离矩阵 
        
        OUTPUT
        loss: (batch_size, )

        """
        
        batch_size = embedding.size(0)
        
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        # (batch_size, embedding_dim)
        cos_similarity = torch.mm(norm_embedding, norm_embedding.t())
        # (batch_size, batch_size)
        max_dist = edit_distance.max()
        if max_dist == 0:
            max_dist = torch.tensor(1.0, device=device)
        target_similarity = 1.0 - (edit_distance / max_dist)
        # (batch_size, batch_size)
        similarity_loss = F.mse_loss(cos_similarity, target_similarity)
        
        logits = cos_similarity / self.temperature
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=embedding.device)
        positive_mask = (edit_distance < 3) & ~eye_mask
        negative_mask = (edit_distance >= 5) & ~eye_mask
        
        eps = 1e-8
        positive_loss = 0.0
        if positive_mask.any():
            positive_loss = -torch.log(torch.sigmoid(logits[positive_mask]) + eps).mean()
        negative_loss = 0.0
        if negative_mask.any():
            negative_loss = -torch.log(1 - torch.sigmoid(logits[negative_mask]) + eps).mean()
        contrastive_loss = positive_loss + negative_loss
        
        total_loss =  contrastive_loss + self.similarity_weight * similarity_loss

        return total_loss
    


class WordDataset(Dataset):

    def __init__(self, file_path, converter):

        self.load_data = Load_Data(file_path)
        self.converter = converter
        self.word = self.load_data.load_words(file_path)
    

    def __len__(self):

        return self.load_data.data_size
    

    def __getitem__(self, idx):

        word = self.word[idx]
        char_indices = [self.converter[c] for c in word]

        return char_indices, word



def levenshtein_distance(s1, s2):
    
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]



def collate_fn(batch):
    
    char_indices, words = zip(*batch)
    
    lengths = torch.tensor([len(seq) for seq in char_indices])
    
    max_len = max(lengths)
    padded = torch.zeros(len(char_indices), max_len, dtype=torch.long)
    for i, seq in enumerate(char_indices):
        padded[i, :len(seq)] = torch.tensor(seq)
    
    return padded, lengths, words



def train(model, dataloader, optimizer, criterion, device, epochs=20):
    
    model.train()

    for epoch in range(epochs):

        epoch_loss = 0
        batch_num = len(dataloader)

        for batch_idx, (padded, lengths, words) in enumerate(dataloader):

            padded = padded.to(device)
            lengths = lengths.to(device)

            batch_size = len(words)
            edit_matrix = torch.zeros(batch_size, batch_size)
            for i in range(batch_size):
                for j in range(batch_size):
                    edit_matrix[i, j] = levenshtein_distance(words[i], words[j])

            optimizer.zero_grad()
            embeddings = model(padded, lengths)
            
            loss = criterion(embeddings, edit_matrix)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f'Eopch[{epoch+1}/{epochs}] Batch[{batch_idx+1}/{batch_num}] Loss: {loss}')
        
        epoch_avg_loss = epoch_loss / batch_num
        print(f'Eopch[{epoch+1}/{epochs}] Average_Loss: {epoch_avg_loss}')
    
    return model





BATCH_SIZE = 256
EPOCHS = 20

INPUT_DIM = 30
HIDDEN_DIM = 64
EMBEDDING_DIM = 128
NUM_LAYERS = 2
NUM_DIRECTIONS = 2
DROPOUT = 0.3

LEARNING_RATE = 0.001
SIMILARITY_WEIGHT = 0.7
TEMPERATURE = 0.1
MARGIN = 3.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE_PATH = 'puma.txt'
file_name = FILE_PATH.split('.')[0]
current_date = datetime.now().strftime('%Y%m%d')
SAVE_PATH = f'Word_Embedder_v2_{file_name}_{current_date}.pth'

IDX2CHAR = {
    0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 
    3: 'a', 4: 'b', 5: 'c', 6: 'd', 7: 'e', 8: 'f', 9: 'g',
    10: 'h', 11: 'i', 12: 'j', 13: 'k', 14: 'l', 15: 'm', 16: 'n',
    17: 'o', 18: 'p', 19: 'q', 20: 'r', 21: 's', 22: 't', 23: 'u',
    24: 'v', 25: 'w', 26: 'x', 27: 'y', 28: 'z', 29: '\n'
}
CHAR2IDX = {v: k for k, v in IDX2CHAR.items()}





if __name__ == "__main__":

    print(f'device: {device}')

    dataset = WordDataset(FILE_PATH, CHAR2IDX)
    dataloader = DataLoader(
        dataset = dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    print(f'Load Data Done! Size: {len(dataset)}')

    model = WordEmbedder(
        input_dim=INPUT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        embedding_dim=EMBEDDING_DIM, 
        num_layers=NUM_LAYERS, 
        num_directions=NUM_DIRECTIONS, 
        dropout=DROPOUT
    ).to(device)
    print('Model Initialized! ')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = EditDistanceLoss(
        similarity_weight=SIMILARITY_WEIGHT, 
        temperature=TEMPERATURE,
        margin=MARGIN
    )
    print('Optimizer and Criterion Initialized! ')

    print('Starting Training... ')
    start_time = time.time()
    train(model, dataloader, optimizer, criterion, device, EPOCHS)
    training_time = time.time() - start_time
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Training completed in {training_time//60:.0f}m {training_time%60:.0f}s")
    print(f"model saved to: {SAVE_PATH}")