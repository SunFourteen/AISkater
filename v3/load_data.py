import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

class Load_Data():
    
    def __init__(self, word_path):
        
        self.word_path = word_path
        self.key_seq_fault_rate = 0.1


    def v1(self):

        data, data_size, max_length = self.generate_data()
        
        return data, data_size, max_length


    def v2(self):

        anchor, anchor_size, anchor_max_length = self.generate_data()
        print(f'Anchor Generation Done! \nSize: {anchor_size} \nMax Length: {anchor_max_length} ')
        positive, positive_size, positive_max_length = self.generate_data()
        print(f'Positve Generation Done! \nSize: {positive_size} \nMax Length: {positive_max_length} ')
        negative, negative_size, negative_max_length = self.generate_data()
        random.shuffle(negative)
        print(f'Negative Generation Done! \nSize: {negative_size} \nMax Length: {negative_max_length} ')
       
        assert (anchor_size == self.data_size 
                and positive_size == self.data_size 
                and negative_size== self.data_size)
        
        return anchor, anchor_size, positive, positive_size, negative, negative_size

        
        
    def load_words(self, word_path):
        
        with open(word_path, "r") as f:
            words = f.readlines()
        self.data_size = len(words)

        print('Loading Words Done! ')
        return words # list[string]
    
    
    def load_key_seqs(self, words):

        key_seqs = []
        for word in tqdm(words):
            if '\n' in word:
                 word = word[:-1]
            key_seq = []
            for letter in word:
                key_pair = self.letter2key(letter)
                key_seq.extend(key_pair)
            key_seq = np.array(key_seq)
            key_seq = key_seq[np.where(key_seq[:-1] != key_seq[1:])[0]].tolist() + [int(key_seq[-1])]
            key_seqs.append(key_seq)

        print('Load Key Sequences Done! ')
        return key_seqs # list[list[int]]
    
    
    def load_disturbanced_key_seqs(self, key_seqs):
        
        indices = random.sample(range(self.data_size), int(self.data_size * self.key_seq_fault_rate))
        for idx in indices:
            key_seqs[idx] = self.key_disturbancer(key_seqs[idx])
        indices = random.sample(range(self.data_size), int(self.data_size * self.key_seq_fault_rate ** 2))
        for idx in indices:
            key_seqs[idx] = self.key_disturbancer(key_seqs[idx])
            key_seqs[idx] = self.key_disturbancer(key_seqs[idx])
        
        print('Load Disturbanced Key Sequences Done! ')
        return key_seqs # list[list[int]]
    
    
    def load_twist_seqs(self, key_seqs):
        
        twist_seqs = []
        for key_seq in tqdm(key_seqs):
            twist_seq = []
            for key in key_seq:
                twist_seq.append(self.select_twist(key))
            twist_seqs.append(twist_seq)

        print('Load Twist Sequences Done! ')
        return twist_seqs # list[list[(x, y)]]

    
    def load_seqs(self, twist_seqs):
        
        disturbance_radius = 0.02
        sample_delta = 0.02
        concentrate_factor = 0.2
        concentrate_sample_num = 3
        sparse_sample_num = 4
        spacial_seqs = []
        masks = []
        timely_seqs = []

        for twist_seq in tqdm(twist_seqs):
            combined_spacial_seq = []
            mask = []
            combined_timely_seq = []
            for idx in range(len(twist_seq) - 1):

                spacial_seq = []
                timely_seq = []
                delta_x = twist_seq[idx + 1][0] - twist_seq[idx][0]
                delta_y = twist_seq[idx + 1][1] - twist_seq[idx][1]
                delta = np.linalg.norm((delta_x, delta_y))

                sample_num = int(delta // sample_delta)
                x_seq = np.linspace(twist_seq[idx][0], twist_seq[idx + 1][0], sample_num)[:-1].tolist()
                y_seq = np.linspace(twist_seq[idx][1], twist_seq[idx + 1][1], sample_num)[:-1].tolist()
                for i in range(sample_num - 1):
                    coodinate = (x_seq[i], y_seq[i])
                    coodinate = self.point_disturbancer(coodinate, disturbance_radius)
                    spacial_seq.append(coodinate)
                combined_spacial_seq.extend(spacial_seq)

                mask.append(1)
                for i in range(sample_num - 2):
                    mask.append(0)

                concentrate_delta_x = delta_x * concentrate_factor
                concentrate_delta_y = delta_y * concentrate_factor
                sub1_x = twist_seq[idx][0] + concentrate_delta_x
                sub1_y = twist_seq[idx][1] + concentrate_delta_y
                sub2_x = twist_seq[idx + 1][0] - concentrate_delta_x
                sub2_y = twist_seq[idx + 1][1] - concentrate_delta_y
                x_seq = []
                y_seq = []
                x_seq1 = np.linspace(twist_seq[idx][0], sub1_x, concentrate_sample_num)[:-1].tolist()
                y_seq1 = np.linspace(twist_seq[idx][1], sub1_y, concentrate_sample_num)[:-1].tolist()
                x_seq2 = np.linspace(sub1_x, sub2_x, sparse_sample_num)[:-1].tolist()
                y_seq2 = np.linspace(sub1_y, sub2_y, sparse_sample_num)[:-1].tolist()
                x_seq3 = np.linspace(sub2_x, twist_seq[idx + 1][0], concentrate_sample_num)[:-1].tolist()
                y_seq3 = np.linspace(sub2_y, twist_seq[idx + 1][1], concentrate_sample_num)[:-1].tolist()
                x_seq.extend(x_seq1)
                x_seq.extend(x_seq2)
                x_seq.extend(x_seq3)
                y_seq.extend(y_seq1)
                y_seq.extend(y_seq2)
                y_seq.extend(y_seq3)
                for i in range(concentrate_sample_num * 2 + sparse_sample_num - 3):
                    coodinate = (x_seq[i], y_seq[i])
                    coodinate = self.point_disturbancer(coodinate, disturbance_radius)
                    timely_seq.append(coodinate)
                combined_timely_seq.extend(timely_seq)
                
            combined_spacial_seq.append((twist_seq[-1][0], twist_seq[-1][1]))
            combined_timely_seq.append((twist_seq[-1][0], twist_seq[-1][1]))
            mask.append(1)

            spacial_seqs.append(combined_spacial_seq)
            timely_seqs.append(combined_timely_seq)
            masks.append(mask)

        print('Load Sequences Done! ')
        return spacial_seqs, timely_seqs, masks # list[list[(x, y)]], list[list[(x, y)]] list[list[int]]
        # masks are for spacial_seqs


    def load_data(self, timely_seqs, words):
        
        dataset = []

        for idx in tqdm(range(self.data_size)):
            data = []
            timely_seq = timely_seqs[idx]
            word = words[idx]
            if '\n' in word:
                word = word[:-1]
            length = len(timely_seq)
            
            if length < 3:
                print(f'Skip Word with Short Sequence: Word {word} Length {length}')
                continue
            
            for i in range(1, length-1):
                x_prev, y_prev = timely_seq[i - 1]
                x, y = timely_seq[i]
                x_next, y_next= timely_seq[i + 1]
                dx = (x_next - x_prev) * 3
                dy = (y_next - y_prev) * 3
                vec_prev = np.array([x - x_prev, y - y_prev])
                vec_next = np.array([x_next - x, y_next - y])
                norm_prev = max(np.linalg.norm(vec_prev), 1e-8)
                norm_next = max(np.linalg.norm(vec_next), 1e-8)
                cos_theta = np.dot(vec_prev, vec_next) / (norm_prev * norm_next)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                dtheta = (np.acos(cos_theta) / np.pi).item()
                dt = 1
                data.append((x, y, dx, dy, dtheta, dt)) # normalization
                
            if len(data) > 0:
                dataset.append((data, word))
            else:
                print(f"Skip Word with Empty Sequence: Word {word}")
            
        print('Load Data Done! ')
        return dataset # list[(list(x, y, dx, dy, dtheta, dt), word)]
    

    def generate_data(self):

        words = self.load_words(self.word_path)
        # print(words[0])

        key_seqs = self.load_key_seqs(words)
        # print(key_seqs[0])

        disturbanced_key_seqs = self.load_disturbanced_key_seqs(key_seqs)
        # print(disturbanced_key_seqs[0])

        twist_seqs = self.load_twist_seqs(disturbanced_key_seqs)
        # print(twist_seqs[0])

        spacial_seqs, timely_seqs, masks = self.load_seqs(twist_seqs)
        # print(spacial_seqs[0])
        # print(timely_seqs[0])

        # for seq in spacial_seqs:
        #     self.show_seq(seq)
        #     print(len(seq))
        # for seq in timely_seqs:
        #     self.show_seq(seq)
        #     print(len(seq))

        data = self.load_data(timely_seqs, words)
        # print(data[0])
        data_size = len(data)
        # print(data_size')
        max_length = max(len(data[i][0]) for i in range(data_size))
        # print(self.max_length)

        for i in range(data_size):
            for j in range(len(data[i][0])):
                if len(data[i][0][j]) != 6:
                    print('ERROR! position: {i} {j}')
        
        return data, data_size, max_length


    def letter2key(self, letter):

        key_map = {
                'a': (1, 1), 'b': (1, 2), 'c': (1, 3), 'd': (1, 4), 'e': (1, 5),
                'f': (2, 1), 'g': (2, 2), 'h': (2, 3), 'i': (2, 4), 'j': (2, 5),
                'k': (3, 1), 'l': (3, 2), 'm': (3, 3), 'n': (3, 4), 'o': (3, 5),
                'p': (4, 1), 'q': (4, 2), 'r': (4, 3), 's': (4, 4), 't': (4, 5),
                'u': (5, 1), 'v': (5, 2), 'w': (5, 3), 'x': (5, 4), 'y': (5, 5),
                'z': (5, 5)
            }
        
        return key_map.get(letter.lower())
    
    
    def key_disturbancer(self, key_seq):
        
        if len(key_seq) == 1:
            return key_seq
        key_seq = np.array(key_seq)
        indice = np.random.randint(0, len(key_seq))
        if indice == 0:
            mask = ((np.arange(1, 7) != key_seq[1]))
        elif indice == len(key_seq):
            mask = ((np.arange(1, 7) != key_seq[-1]))
        else:
            mask = (np.arange(1, 7) != key_seq[indice-1]) & (np.arange(1, 7) != key_seq[indice])
        candidates = np.arange(1, 7)[mask] if mask.any() else np.arange(1, 7)
        insert_num = int(np.random.choice(candidates))
        key_seq = key_seq.tolist()
        disturbanced_key_seq = np.concatenate([key_seq[:indice], [insert_num], key_seq[indice:]]).tolist()
        disturbanced_key_seq =[int(x) for x in disturbanced_key_seq]
        
        return disturbanced_key_seq
    
    
    def select_twist(self, key):
        
        radius = 0.25
        disturbance_radius = 0.05
        concentration_factor = 0.3

        if key == 3:
            r = 1
            while r > radius:
                r = abs(random.gauss(0, concentration_factor)) * radius
            theta = random.gauss(0, concentration_factor) * math.pi * 2
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            coodinate = (0.5 + x, 0.5 + y)
        else:
            x = 0.5
            y = 0.5
            while (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius ** 2 or x > 0.5 or y > 0.5:
                x = abs(random.gauss(0, concentration_factor)) * 0.25 + 0.15
                y = abs(random.gauss(0, concentration_factor)) * 0.25 + 0.15
            if key == 1:
                coodinate = (x, y)
            elif key == 2:
                coodinate = (1 - x, y)
            elif key == 4:
                coodinate = (x, 1 - y)
            else:
                coodinate = (1 - x, 1 - y)
        coodinate = self.point_disturbancer(coodinate, disturbance_radius)

        return coodinate
    
    
    def point_disturbancer(self, coodinate, max_radius):

        x, y = coodinate
        r = random.random() * max_radius
        theta = random.random() * math.pi * 2
        x = x + r * math.cos(theta)
        y = y + r * math.sin(theta)
        if x < 0:
            x = 0.0
        if y < 0:
            y = 0.0
        if x > 1:
            x = 1.0
        if y > 1:
            y = 1.0
        coodinate = (x, y)

        return coodinate

    
    def show_seq(self, seq):
        
        x = [seq[i][0] for i in range(len(seq))]
        y = [seq[i][1] for i in range(len(seq))]
        plt.plot(x, y, marker='o', markersize = 1, linestyle='solid')
        plt.xlim(0, 1)
        plt.ylim(1, 0)
        plt.show()
        
        return



if __name__ == '__main__':
    data_path = "puma.txt"
    data = Load_Data(data_path)