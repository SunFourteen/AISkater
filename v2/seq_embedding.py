import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from scipy.spatial.distance import cdist

import time
from tqdm import tqdm
from datetime import datetime

from load_data import Load_Data



class SequenceEmbedder(nn.Module):
    
    '''
    Boidirectional GRU

    INPUT  
    src: (batch_size, max_seq_len, input_dim)  
    src_lens: (batch_size, )

    OUTPUT  
    embedding: (batch_size, embedding_dim)
    '''
    
    def __init__(self, input_dim, hidden_dim, embedding_dim, 
                 num_layers=2, num_directions=2, dropout=0.3):
        
        super().__init__()
        
        self.rnn = nn.GRU(
            input_size = input_dim, 
            hidden_size = hidden_dim, 
            num_layers = num_layers, 
            bidirectional = (num_directions == 2), 
            batch_first = True, 
            dropout = dropout if num_layers > 1 else 0
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim * num_layers * num_directions, hidden_dim * num_layers), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim * num_layers, embedding_dim)
        )
        
    
    def forward(self, src, src_lens):
        
        # self.rnn
        # INPUT
        # src: (batch_size, max_seq_len, input_dim)
        # OUTPUT
        # output: (batch_size, max_seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)

        # self.projection
        # INPUT   hidden: (batch_size, hidden_dim * num_layers * num_directions, )
        # OUTPUT  embedding:(batch_size, embedding_dim)

        # num_layers = 2
        # num_directions = 1 if 单向 else 2

        packed_src = rnn_utils.pack_padded_sequence(
            src, src_lens.cpu(), batch_first=True, enforce_sorted=False
            )
        # PackedSequence class
        packed_output, hidden = self.rnn(packed_src)
        # PackedSequence class, (nun_layers * num_directions, batch_size, hidden_dim)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        # (batch_size, max_seq_len, hidden_dim * num_directions)

        pooled = self.adaptive_pool(output.permute(0, 2, 1)).squeeze(-1)
        # (batch_size, hidden_dim * num_directions)

        hidden_forward = hidden[-2, :, :]
        # (batch_size, hidden_dim)
        hidden_backward = hidden[-1, :, :]
        # (batch_size, hidden_dim)

        combined = torch.cat((pooled, hidden_forward, hidden_backward), dim=-1)
        # (batch_size, hidden_dim * nun_layers * num_directions)

        embedding = self.embedding(combined)
        # (batch_size, embedding_dim)
        
        return embedding # (batch_size, embedding_dim)



class SequenceSimilarity():

    def __init__(self, metric='DTW'):
        self.metric = metric

    
    def compute(self, seq1, seq2):

        seq1, seq2 = self.feature(seq1, seq2)
        if self.metric == 'euclidean':
            return self.euclidian(seq1, seq2)
        elif self.metric == 'cosine':
            return self.cosine(seq1, seq2)
        elif self.metric == 'DTW':
            return self.dynamic_time_warping(seq1, seq2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")


    def feature(self, seq1, seq2):

        return seq1, seq2


    def euclidian(self,seq1, seq2):

        return np.exp(-cdist(seq1, seq2).mean())


    def cosine(self, seq1, seq2):

        flat1 = seq1.flatten()
        flat2 = seq2.flatten()
        min_len = min(len(flat1), len(flat2))
        vec1 = flat1[:min_len]
        vec2 = flat2[:min_len]
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        cosine = dot / (norm1 * norm2)

        return cosine


    def dynamic_time_warping(self, seq1, seq2):

        n = len(seq1)
        m = len(seq2)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[0, 1:] = np.inf
        dtw_matrix[1:, 0] = np.inf
        
        dist_matrix = cdist(seq1, seq2, 'euclidean')
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = dist_matrix[i-1, j-1]
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j], 
                    dtw_matrix[i, j-1], 
                    dtw_matrix[i-1, j-1]
                )
        
        return np.exp(-dtw_matrix[n, m] / max(n, m))



class ContractiveLoss(nn.Module):

    def __init__(self, margin, similarity_weight):

        super().__init__()
        self.margin = margin
        self.similarity_weight = similarity_weight

    
    def forward(self, anchor, positive, negative, similarity_matrix):

        distance_pos = F.pairwise_distance(anchor, positive, 2) # (batch_size, )
        distance_neg = F.pairwise_distance(anchor, negative, 2) # (batch_size, )
        triplet_loss = F.relu(distance_pos - distance_neg + self.margin, 0) # (batch_size, )

        batch_size = anchor.size(0)
        similarity_loss = 0

        for i in range(batch_size):
            for j in range(batch_size):

                emb_similarity = F.cosine_similarity(anchor[i].unsqueeze(0), anchor[j].unsqueeze(0))
                similarity_loss += F.mse_loss(emb_similarity, similarity_matrix[i, j])
        
        similarity_loss /= (batch_size * batch_size)
        
        total_loss = triplet_loss.mean() + self.similarity_weight * similarity_loss

        return total_loss



class SequenceDataset(Dataset):

    def __init__(self, file_path):
        
        self.load_data = Load_Data(file_path)
        self.anchor, _, self.positive, _, self.negative, _ = self.load_data.v2()
        self.size = self.load_data.data_size
    

    def __len__(self):

        return self.size
    

    def __getitem__(self, idx):

        anchor_data = self.anchor[idx]
        positive_data = self.positive[idx]
        negative_data = self.negative[idx]

        anchor_seq = anchor_data[0]
        positive_seq = positive_data[0]
        negative_seq = negative_data[0]
        
        return (
            torch.tensor(anchor_seq, dtype=torch.float32),
            torch.tensor(positive_seq, dtype=torch.float32),
            torch.tensor(negative_seq, dtype=torch.float32)
        )



def collate_fn(batch):

    anchor_seqs, positive_seqs, negative_seqs = zip(*batch)
    
    anchor_lengths = torch.tensor([len(seq) for seq in anchor_seqs])
    positive_lengths = torch.tensor([len(seq) for seq in positive_seqs])
    negative_lengths = torch.tensor([len(seq) for seq in negative_seqs])
    
    padded_anchor = rnn_utils.pad_sequence(anchor_seqs, batch_first=True)
    padded_positive = rnn_utils.pad_sequence(positive_seqs, batch_first=True)
    padded_negative = rnn_utils.pad_sequence(negative_seqs, batch_first=True)
    
    return (
        padded_anchor, anchor_lengths, 
        padded_positive, positive_lengths, 
        padded_negative, negative_lengths
    )


def compute_similarity_matrix(sequences, metric='DTW'):

    n = len(sequences)
    similarity_matrix = np.zeros((n, n))
    calculator = SequenceSimilarity(metric=metric)
    
    print(f'Calculate Similarity Matrix --Type: {metric}')
    for i in tqdm(range(n)):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 0.0
            else:
                similarity = calculator.compute(sequences[i], sequences[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
    
    return torch.tensor(similarity_matrix, dtype=torch.float32)



def train(model, dataloader, optimizer, criterion, device, epochs=20):
    
    model.train()

    for epoch in range(epochs):

        epoch_loss = 0
        batch_num = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):

            padded_anchors, anchor_lengths, \
            padded_positives, positive_lengths, \
            padded_negatives, negative_lengths \
            = batch

            padded_anchors = padded_anchors.to(device)
            anchor_lengths = anchor_lengths.to(device)
            padded_positives = padded_positives.to(device)
            positive_lengths = positive_lengths.to(device)
            padded_negatives = padded_negatives.to(device)
            negative_lengths = negative_lengths.to(device)

            anchors = []
            for i, anchor in enumerate(padded_anchors):
                
                length = anchor_lengths[i]
                anchors.append(anchor[:length].cpu().numpy())

            similarity_matrix = compute_similarity_matrix(anchors, metric='DTW')
            similarity_matrix = similarity_matrix.to(device)

            optimizer.zero_grad()
            anchor_emb = model(padded_anchors, anchor_lengths)
            positive_emb = model(padded_positives, positive_lengths)
            negative_emb = model(padded_negatives, negative_lengths)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb, similarity_matrix)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f'Eopch[{epoch+1}/{epochs}] Batch[{batch_idx+1}/{batch_num}] Loss: {loss}')
        
        epoch_avg_loss = epoch_loss / batch_num
        print(f'Eopch[{epoch+1}/{epochs}] Average_Loss: {epoch_avg_loss}')







BATCH_SIZE = 256
EPOCHS = 20

INPUT_DIM = 6
HIDDEN_DIM = 64
EMBEDDING_DIM = 128
NUM_LAYERS = 2
NUM_DIRECTIONS = 2
DROPOUT = 0.3

LEARNING_RATE = 0.001
MARGIN = 0.5
SIMILARITY_WEIGHT = 0.7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE_PATH = '3500.txt'
file_name = FILE_PATH.split('.')[0]
current_date = datetime.now().strftime('%Y%m%d')
SAVE_PATH = f'Seq_Embedder_v2_{file_name}_{current_date}.pth'





if __name__ == '__main__':
    
    print(f'device: {device}')

    dataset = SequenceDataset(FILE_PATH)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        collate_fn=collate_fn
    )
    print(f'Load Data Done! Size: {len(dataset)}')

    model = SequenceEmbedder(
        INPUT_DIM, 
        HIDDEN_DIM, 
        EMBEDDING_DIM, 
        NUM_LAYERS, 
        NUM_DIRECTIONS, 
        DROPOUT
    ).to(device)
    print('Model Initialized! ')

    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    criterion = ContractiveLoss(MARGIN, SIMILARITY_WEIGHT)
    print('Optimizer and Criterion Initialized! ')

    print('Starting Training... ')
    start_time = time.time()
    train(model, dataloader, optimizer, criterion, device, EPOCHS)
    training_time = time.time() - start_time
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Training completed in {training_time//60:.0f}m {training_time%60:.0f}s")
    print(f"model saved to: {SAVE_PATH}")