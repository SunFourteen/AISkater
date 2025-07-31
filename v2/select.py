import random

def select_random_lines(input_file, output_file, num_lines=200):

    with open(input_file, 'r') as f:
        lines = f.readlines()

    if len(lines) < num_lines:
        raise ValueError(f"输入文件只有 {len(lines)} 行，少于要求的 {num_lines} 行")
    
    selected_lines = random.sample(lines, num_lines)
    
    with open(output_file, 'w') as f:
        f.writelines(selected_lines)





if __name__ == '__main__':
    input_filename = '3500.txt'
    output_filename = '3500_test.txt'
    select_random_lines(input_filename, output_filename, 200)