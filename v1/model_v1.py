import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import math
from datetime import datetime
from load_data import Load_Data

class Encoder(nn.Module):
    
    '''
    Boidirectional GRU

    INPUT
    src: (batch_size, max_seq_len, input_dim)
    src_lens: (batch_size, )

    OUTPUT
    outputs: (batch_size, max_seq_len, hidden_dim * 2)
    hidden_layer1: (batch_size, hidden_dim * 2)
    hidden_layer2: (batch_size, hidden_dim * 2)
    '''
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        
        super().__init__()
        
        self.rnn = nn.GRU(
            input_size = input_dim, 
            hidden_size = hidden_dim, 
            num_layers = num_layers, 
            bidirectional = True, 
            batch_first = True, 
            dropout = dropout if num_layers > 1 else 0
        )
        self.fc_layer1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_layer2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
    
    def forward(self, src, src_lens):
        
        # self.rnn:
        # INPUT
        # src: (batch_size, max_seq_len, input_dim)
        # OUTPUT
        # outputs: (batch_size, max_seq_len, hidden_dim * 2) caused by bidirection
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)
        # num_direction = 1 if 单向 else 2
        
        packed_src = rnn_utils.pack_padded_sequence(
            src, src_lens.cpu(), batch_first=True, enforce_sorted=False
            )
        packed_outputs, hidden = self.rnn(packed_src)
        # (batch_size, max_seq_len, hidden_dim * 2), (2 * 2, batch_size, hidden_dim)
        outputs, _ = rnn_utils.pad_packed_sequence(
            packed_outputs, batch_first=True
            )
        
        hidden_forward_layer1 = hidden[0, :, :] # (batch_size, hidden_dim)
        hidden_backward_layer1 = hidden[1, :, :] # (batch_size, hidden_dim)
        hidden_reshape_layer1 = torch.cat((hidden_forward_layer1, hidden_backward_layer1), dim=1) # (batch_size, hidden_dim * 2)
        hidden_forward_layer2 = hidden[-2, :, :] # (batch_size, hidden_dim)
        hidden_backward_layer2 = hidden[-1, :, :] # (batch_size, hidden_dim)
        hidden_reshape_layer2 = torch.cat((hidden_forward_layer2, hidden_backward_layer2), dim=1) # (batch_size, hidden_dim * 2)
        # 这里hidden的四个索引分别为 第一层前向 第一层后向 第二层前向 第二层后向
        hidden_layer1 = self.fc_layer1(hidden_reshape_layer1)
        hidden_layer1 = torch.tanh(hidden_layer1) # (batch_size, hidden_dim)
        hidden_layer2 = self.fc_layer2(hidden_reshape_layer2)
        hidden_layer2 = torch.tanh(hidden_layer2) # (batch_size, hidden_dim)
        
        return outputs, hidden_layer1, hidden_layer2
        # (batch_size, max_seq_len, hidden_dim * 2), (batch_size, hidden_dim), 
        # (batch_size, hidden_dim)



class Attention(nn.Module):
    
    '''
    Bahdanau Attention: 产生每一个时间步的权重

    INPUT
    hidden_layer2: (batch_size, hidden_dim)
    encoder_outputs: (batch_size, max_seq_len, hidden_dim * 2)
    mask: (batch_size, max_seq_len)

    OUTPUT
    attention: (batch_size, max_seq_len)
    '''

    def __init__(self, hidden_dim):
        
        super().__init__()
        
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.hidden_dim = hidden_dim
        
        
    def forward(self, hidden_layer2, encoder_outputs, mask):
        
        max_seq_len = encoder_outputs.shape[1]
        hidden_layer2 = hidden_layer2.unsqueeze(1).repeat(1, max_seq_len, 1)
        # (batch_size, hidden_dim) -> (batch_size, max_seq_len, hidden_dim)
        energy = self.attn(torch.cat((hidden_layer2, encoder_outputs), dim=2))
        energy = torch.tanh(energy)
        # (batch_size, max_seq_len, hidden_dim * 3) -> (batch_size, max_seq_len, hidden_dim)
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(attention, dim=1) # (batch_size, max_seq_len)
        # (batch_size, max_seq_len, hidden_dim) -> (batch_size, max_seq_len)

        return attention
        # (batch_size, max_seq_len)



class Decoder(nn.Module):
    
    '''
    Single direction GRU

    INPUT
    input: (batch_size) -- 每次只向decoder提供一个字符
    hidden_layer1: (batch_size, hidden_dim)
    hidden_layer2: (batch_size, hidden_dim)
    encoder_outputs: (batch_size, max_seq_len, hidden_dim * 2)
    mask: (batch_size, max_seq_len)

    OUTPUT
    prediction: (batch_size, alphabet_dim)
    hidden_layer1: (batch_size, hidden_dim)
    hidden_layer2: (batch_size, hidden_dim)
    attn_weights: (batch_size, max_seq_len)

    '''

    def __init__(self, alphabet_dim, hidden_dim, num_layers=2, dropout=0.1):

        super().__init__()
        
        self.embedding = nn.Embedding(alphabet_dim, hidden_dim)
        # 将alphabet_dim大小的字典（这里是字母和符）映射入hidden_dim维向量空间
        self.attention = Attention(hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim * 3,  # 嵌入向量 + 上下文向量
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 4, alphabet_dim)
        self.dropout = nn.Dropout(dropout) # 以dropout比例置零
        self.alphabet_dim = alphabet_dim
        self.hidden_dim = hidden_dim
        

    def forward(self, input, hidden_layer1, hidden_layer2, encoder_outputs, mask):

        embedded = self.dropout(self.embedding(input.unsqueeze(1)))
        # (batch_size) -> (batch_size, 1, hidden_dim)
        attn_weights = self.attention(hidden_layer2, encoder_outputs, mask).unsqueeze(1)
        # (batch_size, 1, max_seq_len)

        context = torch.bmm(attn_weights, encoder_outputs)  # (batch_size, 1, hidden_dim * 2)
        rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, hidden_dim * 3)
        
        hidden = torch.stack((hidden_layer1, hidden_layer2), dim=0) # (2, batch_size, hidden_dim)
        output, hidden = self.rnn(rnn_input, hidden)
        # (batch_size, 1, hidden_dim) (batch_size, 2, hidden_dim)
        
        output = output.squeeze(1)  # (batch_size, hidden_dim)
        context = context.squeeze(1) # (batch_size, hidden_dim * 2)
        embedded = embedded.squeeze(1)  # (batch_size, hidden_dim)
        concat = torch.cat((output, context, embedded), dim=1) # (batch_size, hidden_dim * 4)
        prediction = self.fc(concat) # (batch_size, alphabet_dim)
        attn_weights = attn_weights.squeeze(1) # (batch_size, max_seq_len)
        
        hidden_layer1 = hidden[0, :, :]
        hidden_layer2 = hidden[1, :, :]
        
        return prediction, hidden_layer1, hidden_layer2, attn_weights
        # (batch_size, alphabet_dim), (batch_size, hidden_dim), 
        # (batch_size, hidden_dim), (batch_size, max_seq_len)



class Seq2Seq(nn.Module):
    '''
    Sequence to Sequence
    
    INPUT
    src: (batch_size, max_src_len, input_dim)
    src_lens: (batch_size, )
    trg: (batch_size, max_trg_len) if training else None
    max_trg_len = 20
    teacher_forcing_ratio = 0.1 
    
    OUTPUT
    outputs: (batch_size, max_trg_len, alphabet_dim)
    '''

    def __init__(self, encoder, decoder, device, pad_idx, sos_idx, eos_idx):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
    
    def create_mask(self, src, src_lens):
        
        mask = torch.ones(src.size(0), src.size(1), device=self.device) # (batch_size, max_seq_len)
        for idx, length in enumerate(src_lens):
            mask[idx, length:] = 0
        return mask
    
    def forward(self, src, src_lens, trg=None, max_trg_len=20, teacher_forcing_ratio=0.1):

        encoder_outputs, encoder_hidden1, encoder_hidden2 = self.encoder(src, src_lens)
        batch_size = src.size(0)
        hidden_dim = encoder_hidden1.size(1)
        src_mask = self.create_mask(src, src_lens)
        outputs = torch.zeros(batch_size, max_trg_len, self.decoder.alphabet_dim).to(self.device)
        
        input = torch.full((batch_size, ), self.sos_idx, dtype=torch.long).to(self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)      
        hidden1 = torch.zeros((batch_size, max_trg_len, hidden_dim), dtype = torch.float32)
        hidden2 = torch.zeros((batch_size, max_trg_len, hidden_dim), dtype = torch.float32)
        hidden1[:, 0, :] = encoder_hidden1
        hidden2[:, 0, :] = encoder_hidden2

        for t in range(max_trg_len):

            active = ~finished
            if not active.any():
                break
            
            prediction, decoder_hidden1, decoder_hidden2, _ = self.decoder(
                input[active], 
                hidden1[active, t, :], 
                hidden2[active, t, :], 
                encoder_outputs[active], 
                src_mask[active]
            )
            outputs[active, t] = prediction
            top1 = prediction.argmax(1)
            
            if trg is not None and teacher_forcing_ratio > 0 and t > 0:
                teacher_force = torch.rand(1) < teacher_forcing_ratio
                if teacher_force:
                    input[active] = trg[active, t]
                else:
                    input[active] = top1
            else:
                input[active] = top1
            
            finished[active] = (top1 == self.eos_idx)
            
            if t < max_trg_len - 1:
                hidden1[active, t+1, :] = decoder_hidden1
                hidden2[active, t+1, :] = decoder_hidden2
        
        return outputs
        # (batch_size, max_trg_len, alphabet_dim)



class SlideDataset(Dataset):

    def __init__(self, file_path, input_dim, alphabet_dim, 
                sos_idx, eos_idx, pad_idx, converter, 
                num_samples=1000, max_src_len=150, max_trg_len=20):

        self.load_data = Load_Data(file_path)
        self.input_dim = input_dim
        self.alphabet_size = alphabet_dim
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.converter = converter
        self.data = self.load_data.data
        self.num_samples = self.load_data.data_size
        self.max_src_len = self.load_data.max_length + 2
        self.max_trg_len = max_trg_len


    def __len__(self):

        return self.num_samples
        

    def __getitem__(self, idx):

        data_list, word = self.data[idx]
        
        for point in data_list:
            for val in point:
                if math.isnan(val) or math.isinf(val):
                    print(f"invalid data: {point}")
                    point = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                    
        src_tensor = torch.tensor(data_list, dtype=torch.float)
        src_len = torch.tensor([len(data_list)], dtype=torch.long)
        
        trg_indices = [self.sos_idx] + [self.converter[char] for char in word] + [self.eos_idx]
        trg_len = len(trg_indices)

        trg_tensor = torch.full((self.max_trg_len, ), self.pad_idx, dtype=torch.long)
        if trg_len > self.max_trg_len:
            trg_tensor[:] = torch.tensor(trg_indices[:self.max_trg_len], dtype=torch.long)
        else:
            trg_tensor[:trg_len] = torch.tensor(trg_indices, dtype=torch.long)

        return src_tensor, src_len, trg_tensor



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.01)



def collate_fn(batch):
    '''batch: tuple (src, src_len, trg)'''

    srcs, src_lens, trgs = zip(*batch)
    
    sorted_indices = sorted(range(len(src_lens)), key=lambda i: src_lens[i].item(), reverse=True)
    srcs = [srcs[i] for i in sorted_indices]
    src_lens = torch.cat([src_lens[i] for i in sorted_indices])
    trgs = torch.stack([trgs[i] for i in sorted_indices])

    max_src_len = max(src.size(0) for src in srcs)
    padded_srcs = torch.zeros(len(srcs), max_src_len, INPUT_DIM)
    for i, src in enumerate(srcs):
        padded_srcs[i, :src.size(0)] = src
    
    return padded_srcs, src_lens, trgs



def train(model, train_loader, val_loader, optimizer, criterion, epochs, save_path, max_trg_len=20, teacher_forcing_ratio=0.1):

    model.train()
    best_val_loss = 10000

    for epoch in range(epochs):

        total_loss = 0

        for batch_idx, (src, src_lens, trg) in enumerate(train_loader):

            src = src.to(device)
            src_lens = src_lens.to(device)
            trg = trg.to(device)
            
            optimizer.zero_grad() # 梯度清零
            
            output = model(src, src_lens, trg=trg, max_trg_len=max_trg_len, teacher_forcing_ratio=teacher_forcing_ratio)
            # (batch_size, trg_len, alphabet_dim)
            
            # # 检查输出是否有效
            # if torch.isnan(output).any():
            #     print("模型输出包含NaN值！")
            #     continue
                
            # loss = criterion(output[:, :-1].permute(0, 2, 1), trg[:, 1:])
            
            # # 检查损失是否有效
            # if torch.isnan(loss):
            #     print("损失为NaN，跳过该批次")
            #     continue
                
            # loss.backward()
            
            # # 梯度监控和裁剪
            # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            # if torch.isnan(total_norm):
            #     print("梯度包含NaN值！")
            #     model.zero_grad()
            #     continue
                
            # optimizer.step()
            # optimizer.zero_grad()
            
            loss = criterion(output[:, :-1].permute(0, 2, 1), trg[:, 1:])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # 梯度裁剪
            optimizer.step() # 参数更新
            
            total_loss += loss.item() # batch内损失累计得到epoch总损失
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader) # epoch平均损失
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
    

        val_loss, val_acc = evaluate(model, val_loader, criterion, max_trg_len)
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, save_path)
            print(f"Saved best model at epoch {epoch+1}")



def evaluate(model, val_loader, criterion, max_trg_len=20):

    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, src_lens, trg in val_loader:
            src = src.to(device)
            src_lens = src_lens.to(device)
            trg = trg.to(device)
            
            output = model(src, src_lens, trg=None, max_trg_len=max_trg_len, teacher_forcing_ratio=0)
            
            loss = criterion(output[:, :-1].permute(0, 2, 1), trg[:, 1:])
            total_loss += loss.item()
            
            preds = output[:, :-1].argmax(-1)
            mask = (trg[:, 1:] != PAD_IDX)
            total_correct += ((preds == trg[:, 1:]) & mask).sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, accuracy



# Hyperparameter
INPUT_DIM = 6  # x, y, dx, dy, dtheta, dt
ALPHABET_DIM = 29
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64 // 4
LEARNING_RATE = 0.001 
EPOCHS = 20
MAX_TRG_LEN = 20
TEACHER_FORCING_RATIO = 0.75 / 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IDX2CHAR = {
    0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 
    3: 'a', 4: 'b', 5: 'c', 6: 'd', 7: 'e', 8: 'f', 9: 'g',
    10: 'h', 11: 'i', 12: 'j', 13: 'k', 14: 'l', 15: 'm', 16: 'n',
    17: 'o', 18: 'p', 19: 'q', 20: 'r', 21: 's', 22: 't', 23: 'u',
    24: 'v', 25: 'w', 26: 'x', 27: 'y', 28: 'z'
}
CHAR2IDX = {v: k for k, v in IDX2CHAR.items()}
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


FILE_PATH = 'puma.txt'
file_name = FILE_PATH.split('.')[0]
current_date = datetime.now().strftime('%Y%m%d')
SAVE_PATH = f'v1_{file_name}_{current_date}.pth'


encoder = Encoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
decoder = Decoder(ALPHABET_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
model = Seq2Seq(encoder, decoder, device, PAD_IDX, SOS_IDX, EOS_IDX).to(device)


# Xavier Initialization
# encoder.apply(init_weights)
# decoder.apply(init_weights)


dataset = SlideDataset(FILE_PATH, INPUT_DIM, ALPHABET_DIM, SOS_IDX, EOS_IDX, PAD_IDX, CHAR2IDX)
train_size = int(len(dataset) *  0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn
)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)





if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True): # 梯度异常自动检测
        train(model, train_loader, val_loader, optimizer, criterion, EPOCHS, SAVE_PATH, MAX_TRG_LEN, TEACHER_FORCING_RATIO)