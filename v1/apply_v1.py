import torch
from torch.utils.data import DataLoader
from model_v1 import Encoder, Decoder, Seq2Seq, SlideDataset

INPUT_DIM = 6  # x, y, dx, dy, dtheta, dt
ALPHABET_DIM = 29
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 1
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
MODEL_PATH = 'v1_puma_20250725.pth'
FILE_PATH = 'puma_test.txt'


def load_model():

    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(ALPHABET_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device, PAD_IDX, SOS_IDX, EOS_IDX)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {MODEL_PATH}")
    return model


def predict(model, src, src_len):

    with torch.no_grad():

        src = src.to(device)
        src_len = src_len.to(device)

        output = model(src, src_len, None, MAX_TRG_LEN, 0).squeeze(0)
        output_best = output.argmax(-1).cpu()
        # output_prab = torch.softmax(output.cpu(), dim=1)
        # for posi in range(output.size(0)):
        #     print(f'position {posi}:')
        #     print(' '.join([f'{output_prab[posi][char].item():.4f}' for char in range(output.size(1))]))
        predict = ''
        for idx in range(output_best.size(0)):
            predict += IDX2CHAR[output_best[idx].item()]
        return predict
    

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



if __name__ == '__main__':

    model = load_model()
    test_dataset = SlideDataset(FILE_PATH, INPUT_DIM, ALPHABET_DIM, SOS_IDX, EOS_IDX, PAD_IDX, CHAR2IDX)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    for idx, (src, src_len, trg) in enumerate(test_loader):

        predicted_word = predict(model, src, src_len)
        actual_word = test_dataset.data[idx][1]
        
        print(f"  样本 {idx+1}:")
        print(f"  实际: {actual_word}")
        print(f"  预测: {predicted_word}")
        print("-" * 50)