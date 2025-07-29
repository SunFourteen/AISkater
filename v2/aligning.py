import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import time
from tqdm import tqdm
from datetime import datetime

from load_data import Load_Data
from seq_embedding import SequenceEmbedder
from word_embedding import WordEmbedder


class Alignment(nn.Module):

    def __init__(self, seq_embedder, word_embedder, embedding_dim, 
                 similarity_weight=0.7, temperature=0.1):

        super().__init__()

        self.seq_embedder = seq_embedder
        self.word_embedder = word_embedder
        
        for param in self.seq_embedder.parameters():
            param.requires_grad = False
        for param in self.word_embedder.parameters():
            param.requires_grad = False

        self.seq_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), 
            nn.ReLU(), 
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.word_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), 
            nn.ReLU(), 
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.similarity_weight = similarity_weight
        self.temperature = temperature


    def forward(self, seq, seq_len, word, word_len):

        seq_emb = self.seq_embedder(seq, seq_len)
        word_emb = self.word_embedder(word, word_len)

        seq_proj = self.seq_projector(seq_emb)
        word_proj = self.word_projector(word_emb)
        
        return seq_proj, word_proj # both (bacth_size, embedding_dim)
    
    
    def loss(self, seq_proj, word_proj):

        logits = torch.mm(seq_proj, word_proj.t()) / self.temperature
        # (batch_size, batch_size)
        targets = torch.arange(seq_proj.size(0)).to(seq_proj.device)
        # (batch_size, )
        contrastive_loss = F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
        
        seq_similarity = F.cosine_similarity(seq_proj.unsqueeze(1), seq_proj.unsqueeze(0), dim=-1)
        # (batch_size, batch_size)
        word_similarity = F.cosine_similarity(word_proj.unsqueeze(1), word_proj.unsqueeze(0), dim=-1)
        # (batch_size, batch_size
        similarity_loss = F.mse_loss(seq_similarity, word_similarity)

        total_loss = contrastive_loss + self.similarity_weight * similarity_loss

        return total_loss
    


class AlignmentDataset(Dataset):

    def __init__(self, file_path, converter):

        self.load_data = Load_Data(file_path)
        self.data, self.data_size, _ = self.load_data.v1()
        self.converter = converter

    
    def __len__(self):

        return self.data_size
    

    def __getitem__(self, idx):

        data = self.data[idx]
        seq, word = data
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        char_indice = [self.converter[c] for c in word]

        return seq, char_indice
    


def collate_fn(batch):

    seqs, char_indices = zip(*batch)

    seq_lengths = torch.tensor([len(seq) for seq in seqs])
    char_indices_lengths = torch.tensor([len(seq) for seq in char_indices])

    seqs = [torch.tensor(seqs[idx], dtype=torch.float32) for idx in range(len(batch))]
    padded_seqs = rnn_utils.pad_sequence(seqs, batch_first=True)

    char_indices = [torch.tensor(char_indices[idx], dtype=torch.long) for idx in range(len(batch))]
    padded_char_indices = rnn_utils.pad_sequence(char_indices, batch_first=True)
    
    return (
        padded_seqs, 
        seq_lengths, 
        padded_char_indices, 
        char_indices_lengths
    )



def train(model, dataloader, optimizer, device, epochs=20):
    
    model.train()

    for epoch in range(epochs):

        epoch_loss = 0
        batch_num = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):

            seqs, seq_lens, \
            chars, char_lens \
            = batch

            seqs= seqs.to(device)
            seq_lens = seq_lens.to(device)
            chars = chars.to(device)
            char_lens = char_lens.to(device)

            optimizer.zero_grad()
            seq_proj, word_proj = model(seqs, seq_lens, chars, char_lens)
            loss = model.loss(seq_proj, word_proj)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f'Eopch[{epoch+1}/{epochs}] Batch[{batch_idx+1}/{batch_num}] Loss: {loss}')
        
        epoch_avg_loss = epoch_loss / batch_num
        print(f'Eopch[{epoch+1}/{epochs}] Average_Loss: {epoch_avg_loss}')





BATCH_SIZE = 256
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

FILE_PATH = 'puma.txt'
file_name = FILE_PATH.split('.')[0]
current_date = datetime.now().strftime('%Y%m%d')
SAVE_PATH = f'Align_2_{file_name}_{current_date}.pth'

SEQ_EMBEDDER_PATH = 'Seq_Embedder_v2_puma_20250728.pth'
WORD_EMBEDDER_PATH = 'Word_Embedder_v2_puma_20250728.pth'

IDX2CHAR = {
    0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 
    3: 'a', 4: 'b', 5: 'c', 6: 'd', 7: 'e', 8: 'f', 9: 'g',
    10: 'h', 11: 'i', 12: 'j', 13: 'k', 14: 'l', 15: 'm', 16: 'n',
    17: 'o', 18: 'p', 19: 'q', 20: 'r', 21: 's', 22: 't', 23: 'u',
    24: 'v', 25: 'w', 26: 'x', 27: 'y', 28: 'z', 29: '\n'
}
CHAR2IDX = {v: k for k, v in IDX2CHAR.items()}





if __name__ == '__main__':

    print(f'device: {device}')

    dataset = AlignmentDataset(FILE_PATH, CHAR2IDX)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        collate_fn=collate_fn
    )
    print(f'Load Data Done! Size: {len(dataset)}')

    seq_embedder = SequenceEmbedder(
        SEQ_INPUT_DIM, 
        HIDDEN_DIM, 
        EMBEDDING_DIM, 
        NUM_LAYERS, 
        NUM_DIRECTIONS, 
        DROPOUT
    )
    seq_embedder.load_state_dict(torch.load(SEQ_EMBEDDER_PATH))

    word_embedder = WordEmbedder(
        WORD_INPUT_DIM, 
        HIDDEN_DIM, 
        EMBEDDING_DIM, 
        NUM_LAYERS, 
        NUM_DIRECTIONS, 
        DROPOUT
    )
    word_embedder.load_state_dict(torch.load(WORD_EMBEDDER_PATH))

    model = Alignment(
        seq_embedder, 
        word_embedder, 
        EMBEDDING_DIM, 
        SIMILARITY_WEIGHT, 
        TEMPERATURE        
    ).to(device)
    print('Model Initialized! ')

    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    print('Optimizer Initialized! ')

    print('Starting Training... ')
    start_time = time.time()
    train(model, dataloader, optimizer, device, EPOCHS)
    training_time = time.time() - start_time
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Training completed in {training_time//60:.0f}m {training_time%60:.0f}s")
    print(f"model saved to: {SAVE_PATH}")
