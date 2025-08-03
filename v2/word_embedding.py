import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
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

        self.norm = nn.LayerNorm(embedding_dim)


    def forward(self, src, src_len):

        # self.rnn
        # INPUT
        # src: (batch_size, max_seq_len, hidden_dim)
        # OUTPUT
        # output: (batch_size, max_seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)

        # self.attention
        # (batch_size, hidden_dim * num_directions) -> (batch_size, )

        # self.fc
        # (batch_size, hidden_dim * num_directions) -> (batch_dize, embedding_dim)

        # self.norm
        # (batch_dize, embedding_dim) -> (batch_dize, embedding_dim)

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
        
        attn = self.attention(output)
        # (batch_size, max_seq_lens, 1)
        attn_weight = F.softmax(attn, dim=1)
        # (batch_size, max_length, 1)
        weighted = torch.sum(attn_weight * output, dim=1)
        # (batch_size, hidden_dim * num_directions)
        
        embedding = self.fc(weighted)
        # (batch_size, embedding_dim)
        norm = self.norm(embedding)
        # (batch_size, embedding_dim)
        norm = F.normalize(norm, p=2, dim=1)
        # (batch_size, embedding_dim)

        # vec_mean = norm.mean(dim=0) # TEST
        # norms = torch.norm(norm, p=2, dim=1) # TEST
        # norm_mean = norms.mean() # TEST
        # print(f"vec_mean: {vec_mean}\nnorm_mean: {norm_mean}") # TEST

        return norm # (batch_size, embedding_dim)



class Loss(nn.Module):

    def __init__(self, contrast_weight=0.3, similarity_weight=0.7, 
                 beta=0.1, margin=0.05, temperature=0.5):

        """
        Loss

        PARAMETERS    
        temperature (float): 温度参数控制分布锐度  
        margin (float): 边界值用于正负样本分离
        """

        super().__init__()

        self.contrast_weight = contrast_weight
        self.similarity_weight = similarity_weight
        self.beta = beta
        self.margin = margin
        self.temperature = temperature
    

    def forward(self, embedding, word_similarity):

        """

        INPUT
        
        embedding: (batch_size, embedding_dim)
        word_similarity: (batch_size, batch_size)
        
        OUTPUT  
        loss: (batch_size, )
        """
        
        batch_size = embedding.size(0)

        cos_similarity = F.cosine_similarity(
            embedding.unsqueeze(1), 
            embedding.unsqueeze(0), 
            dim=2
        )
        # (batch_size, batch_size)
        similarity_loss = F.smooth_l1_loss(
            cos_similarity, word_similarity, beta=self.beta
        )
        
        threshold = int(batch_size * batch_size * self.margin)
        positive_mask, negative_mask = self.create_mask(word_similarity, threshold * 2, exclude_diagonal=True)
        # positive_num = positive_mask.sum().item() # TEST
        # negative_num = negative_mask.sum().item() # TEST

        # raw_pos_cos_mean = cos_similarity[positive_mask].mean() if positive_mask.any() else 0  # TEST
        # raw_neg_cos_mean = cos_similarity[negative_mask].mean() if negative_mask.any() else 0  # TEST
        # print(f'raw_pos_cos_mean: {raw_pos_cos_mean:.4f}')  # TEST
        # print(f'raw_neg_cos_mean: {raw_neg_cos_mean:.4f}')  # TEST

        logits = cos_similarity / self.temperature
        positive_loss = 0.0
        if positive_mask.any():
            positive_loss = -torch.log(torch.sigmoid(logits[positive_mask]) + 1e-8).mean()
        if negative_mask.any():
            negative_loss = -torch.log(1 - torch.sigmoid(logits[negative_mask]) + 1e-8).mean()
        contrastive_loss = positive_loss + negative_loss
        # positive_logit_mean = logits[positive_mask].mean() # TEST
        # negative_logit_mean = logits[negative_mask].mean() # TEST
        # positive_logit_std = logits[positive_mask].std() # TEST
        # negative_logit_std = logits[negative_mask].std() # TEST
        # print(f'positive_logit: {positive_logit_mean:.4f} ± {positive_logit_std:.4f}') # TEST 
        # print(f'negative_logit: {negative_logit_mean:.4f} ± {negative_logit_std:.4f}') # TEST
        # similarity_difference = (cos_similarity - word_similarity).abs().mean() # TEST
        # print(f"similarity_difference: {similarity_difference:.4f}") # TEST

        total_loss =  self.contrast_weight * contrastive_loss + self.similarity_weight * similarity_loss
        # print(f'positive_num: {positive_num}') # TEST
        # print(f'negative_num: {negative_num}') # TEST
        # print(f'positive_loss: {positive_loss}') # TEST
        # print(f'negative_loss: {negative_loss}') # TEST
        # print(f'similarity_loss: {similarity_loss}') # TEST

        return total_loss
    

    def create_mask(self, matrix, k, exclude_diagonal=True):

        batch_size = matrix.size(0)
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=matrix.device)

        if exclude_diagonal:

            max_matrix = matrix.clone()
            min_val = matrix.min() - 1
            max_matrix[eye_mask] = min_val

            min_matrix = matrix.clone()
            max_val = matrix.max() + 1
            min_matrix[eye_mask] = max_val
        
        else:

            max_matrix = matrix
            min_matrix = matrix

        max_flat = max_matrix.flatten()
        min_flat = min_matrix.flatten()

        _, max_indices = torch.topk(max_flat, k)
        max_mask_flat = torch.zeros_like(max_flat, dtype=torch.bool)
        max_mask_flat[max_indices] = True

        _, min_indices = torch.topk(min_flat, k, largest=False)
        min_mask_flat = torch.zeros_like(min_flat, dtype=torch.bool)
        min_mask_flat[min_indices] = True

        max_mask = max_mask_flat.view(batch_size, batch_size)
        min_mask = min_mask_flat.view(batch_size, batch_size)
    
        if exclude_diagonal:

            max_mask[eye_mask] = False
            min_mask[eye_mask] = False
        
        return max_mask, min_mask



class SimCalculator():

    def __init__(self, sensitivity=2.0, max_diff=6, min_diff=2):
        
        self.max_diff = max_diff
        self.min_diff = min_diff
        self.sensitivity = sensitivity

    def calculate(self, word1, word2):

        jaccard = self.jaccard(word1, word2)
        # dice = self.dice(word1, word2)
        subseq = self.lc_subseq(word1, word2)
        substr = self.lc_substr(word1, word2)
        pre_jaccard = self.pre_jaccard(word1, word2)
        # pre_dice = self.pre_dice(word1, word2)
        jaro_winkler = self.jaro_winkler(word1, word2)
        # print(f'jaccard: {jaccard}') # TEST
        # # print(f'dice: {dice}') # TEST
        # print(f'lc_subseq: {subseq}') # TEST
        # print(f'substr: {substr}') # TEST
        # print(f'pre_jaccard: {pre_jaccard}') # TEST
        # # print(f'pre_dice: {pre_dice}') # TEST
        # print(f'jaro_winkler: {jaro_winkler}') # TEST

        similarity = (
              jaccard * 0.1 
            + subseq * 0.1 
            + substr * 0.1 
            + pre_jaccard * 0.3 
            + jaro_winkler * 0.4
        )
        panelty = self.panelty(word1, word2)
        similarity = min(similarity * panelty * 2 - 0.5, 1)
        '''注意这一行 人为增大相似度'''

        return similarity

    
    def panelty(self, word1, word2):

        len1 = len(word1)
        len2 = len(word2)
        larger = max(len1, len2)
        smaller = min(len1, len2)
        diff = larger - smaller
        ratio = smaller / larger

        if diff < self.min_diff:
            return 1.0
        elif diff > self.max_diff:
            return 1e-8
        else:
            return math.exp(-self.sensitivity * (1 - ratio))


    def jaccard(self, word1, word2, n=3):

        len1 = len(word1)
        len2 = len(word2)

        if min(len1, len2, n) == 0:
            return 0.0

        set1 = {word1[i: i+n] for i in range(len1 - n + 1)}
        set2 = {word2[i: i+n] for i in range(len2 - n + 1)}

        intersection = set1 & set2
        union = set1 | set2

        jaccard_similarity = len(intersection) / len(union)

        return jaccard_similarity
    

    def dice(self, word1, word2, n=3):

        len1 = len(word1)
        len2 = len(word2)

        if min(len1, len2, n) == 0:
            return 0.0

        set1 = {word1[i: i+n] for i in range(len1 - n + 1)}
        set2 = {word2[i: i+n] for i in range(len2 - n + 1)}

        intersection = set1 & set2
        total = len(set1) + len(set2)

        dice_similarity = len(intersection) / total

        return dice_similarity
    

    def lc_subseq(self, word1, word2):

        len1 = len(word1)
        len2 = len(word2)

        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(1, len1 +1 ):
            for j in range(1, len2 + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        max_len = dp[len1][len2]

        similarity = max_len / max(len1, len2)

        return similarity


    def lc_substr(self, word1, word2):

        len1 = len(word1)
        len2 = len(word2)

        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        max_len = 0

        for i in range(1, len1 +1 ):
            for j in range(1, len2 + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_len = max(max_len, dp[i][j])
                else:
                    dp[i][j] = 0

        similarity = max_len / max(len1, len2)

        return similarity

    def pre_jaccard(self, word1, word2, prefix_len=4, n=2):

        prefix_len = min(len(word1), len(word2), prefix_len)
        prefix1 = word1[: prefix_len]
        prefix2 = word2[: prefix_len]

        similarity = self.jaccard(prefix1, prefix2, n)

        return similarity
    

    def pre_dice(self, word1, word2, prefix_len=4, n=2):

        prefix_len = min(len(word1), len(word2), prefix_len)
        prefix1 = word1[: prefix_len]
        prefix2 = word2[: prefix_len]

        sililarity = self.dice(prefix1, prefix2, n)

        return sililarity


    def jaro_winkler(self, word1, word2, prefix_weight=0.1, scaling=0.25):

        """
        Jaro-Winkler Similarity  
        :param: prefix_weight = 0.1 --前缀权重因子  
        :param:  scaling = 0.25 --缩放因子
        :return: jaro_winkler_similarity [0, 1]
        """

        def jaro_similarity(s1, s2):
            len1, len2 = len(s1), len(s2)
            
            match_distance = max(len1, len2) // 2 - 1
            
            s1_matches = [False] * len1
            s2_matches = [False] * len2
            
            matches = 0
            transpositions = 0
            
            for i in range(len1):
                start = max(0, i - match_distance)
                end = min(i + match_distance + 1, len2)
                
                for j in range(start, end):
                    if not s2_matches[j] and s1[i] == s2[j]:
                        s1_matches[i] = True
                        s2_matches[j] = True
                        matches += 1
                        break
            
            if matches == 0:
                return 0.0
            
            k = 0
            for i in range(len1):
                if not s1_matches[i]:
                    continue
                while not s2_matches[k]:
                    k += 1
                if s1[i] != s2[k]:
                    transpositions += 1
                k += 1
            
            transpositions //= 2
            
            m = matches
            t = transpositions
            return (m/len1 + m/len2 + (m-t)/m) / 3 if m > 0 else 0.0
        
        jaro = jaro_similarity(word1, word2)
        
        prefix_len = 0
        min_len = min(len(word1), len(word2), 4)
        for i in range(min_len):
            if word1[i] == word2[i]:
                prefix_len += 1
            else:
                break
        
        winkler = jaro + prefix_len * prefix_weight * (1 - jaro)

        jaro_winkler_similarity = min(winkler, jaro + scaling * prefix_len * (1 - jaro))
        
        return jaro_winkler_similarity



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



def collate_fn(batch):
    
    char_indices, words = zip(*batch)
    
    lengths = torch.tensor([len(seq) for seq in char_indices])
    
    max_len = max(lengths)
    padded = torch.zeros(len(char_indices), max_len, dtype=torch.long)
    for i, seq in enumerate(char_indices):
        padded[i, :len(seq)] = torch.tensor(seq)
    
    return padded, lengths, words



def train(model, dataloader, optimizer, criterion, calculator, device, epochs=20):
    
    model.train()

    for epoch in range(epochs):

        epoch_loss = 0
        batch_num = len(dataloader)

        for batch_idx, (padded, lengths, words) in enumerate(dataloader):

            padded = padded.to(device)
            lengths = lengths.to(device)

            batch_size = len(words)
            distance_matrix = torch.zeros(batch_size, batch_size)
            for i in range(batch_size):
                for j in range(batch_size):
                    distance_matrix[i, j] = calculator.calculate(words[i], words[j])
            # print(words) # TEST
            # print(distance_matrix) # TEST

            optimizer.zero_grad()
            embeddings = model(padded, lengths)
            
            loss = criterion(embeddings, distance_matrix)

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
CONTRAST_WEIGHT = 0.3
SIMILARITY_WEIGHT = 0.7
SMOOTH_L1_BETA = 0.1
MARGIN = 0.05
TEMPERATURE = 0.5

SENSITIVITY = 2.0
MAX_DIFF = 6
MIN_DIFF = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE_PATH = '3500.txt'
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

    dataset = WordDataset(
        file_path=FILE_PATH, 
        converter=CHAR2IDX
    )
    dataloader = DataLoader(
        dataset=dataset, 
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
    
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=LEARNING_RATE
    )
    criterion = Loss(
        contrast_weight=CONTRAST_WEIGHT, 
        similarity_weight=SIMILARITY_WEIGHT, 
        beta=SMOOTH_L1_BETA, 
        margin=MARGIN, 
        temperature=TEMPERATURE
    )
    calculator = SimCalculator(
        sensitivity=SENSITIVITY, 
        max_diff=MAX_DIFF, 
        min_diff=MIN_DIFF
    )
    print('Optimizer and Criterion Initialized! ')

    print('Starting Training... ')
    start_time = time.time()
    train(
        model=model, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        criterion=criterion, 
        calculator=calculator, 
        device=device, 
        epochs=EPOCHS)
    end_time = time.time()
    training_time = end_time - start_time
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Training completed in {training_time//60:.0f}m {training_time%60:.0f}s")
    print(f"model saved to: {SAVE_PATH}")