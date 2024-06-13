import subprocess
import os
kernel_sizes_list = [
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "2 3",
    "3 4",
    "4 5",
    "5 6",
    "6 7",
    "2 4",
    "3 5",
    "4 6",
    "5 7",
    "2 3 4",
    "3 4 5",
    "4 5 6",
    "5 6 7",
    "2 4 6",
    "3 5 7",
    "2 3 4 5",
    "3 4 5 6",
    "4 5 6 7"
]

# 명령어 리스트
char_command = [
    
#########################################################


    "python train_char_only.py --threshold 0.5 --batch_size 256",
    "python train_char_only.py --threshold 0.5 --batch_size 16",
    "python train_char_only.py --threshold 0.5 --batch_size 32",
    "python train_char_only.py --threshold 0.5 --batch_size 64",
    "python train_char_only.py --threshold 0.5 --batch_size 128",

    "python train_char_only.py --threshold 0.5 --max_len 64",
    "python train_char_only.py --threshold 0.5 --max_len 128",
    "python train_char_only.py --threshold 0.5 --max_len 256",

    "python train_char_only.py --num_filters 5",
    "python train_char_only.py --num_filters 10",
    "python train_char_only.py --num_filters 25",
    "python train_char_only.py --num_filters 50",
    "python train_char_only.py --num_filters 100",
    "python train_char_only.py --num_filters 150",
    "python train_char_only.py --num_filters 200",
]
#########################################################
hybrid_command= [   


    "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 256",
    "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 16",
    "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 32",
    "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 64",
    "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 128",

    "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --max_len 64",
    "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --max_len 128",
    "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --max_len 256",


    "python train.py --bert allenai/scibert_scivocab_cased --num_filters 5",
    "python train.py --bert allenai/scibert_scivocab_cased --num_filters 10",
    "python train.py --bert allenai/scibert_scivocab_cased --num_filters 25",
    "python train.py --bert allenai/scibert_scivocab_cased --num_filters 50",
    "python train.py --bert allenai/scibert_scivocab_cased --num_filters 100",
    "python train.py --bert allenai/scibert_scivocab_cased --num_filters 150",
    "python train.py --bert allenai/scibert_scivocab_cased --num_filters 200",

]



#########################################################

word_command = [    
   
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 256",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 16",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 32",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 64",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 128",

    "python train_word_only.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --max_len 64",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --max_len 128",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --max_len 256",


    "python train_word_only.py --bert allenai/scibert_scivocab_cased --num_filters 5",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --num_filters 10",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --num_filters 25",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --num_filters 50",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --num_filters 100",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --num_filters 150",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --num_filters 200",

]
for kernel_sizes in kernel_sizes_list:
    cmd = f"python train_char_only.py --kernel_sizes {kernel_sizes}"
    char_command.append(cmd)
for kernel_sizes in kernel_sizes_list:
    cmd = f"python train.py --bert allenai/scibert_scivocab_cased --kernel_sizes {kernel_sizes}"
    hybrid_command.append(cmd)
for kernel_sizes in kernel_sizes_list:
    cmd = f"python train_word_only.py --bert allenai/scibert_scivocab_cased --kernel_sizes {kernel_sizes}"
    word_command.append(cmd)

# # 각 명령어를 순차적으로 실행
# for command in commands:
#     process = subprocess.run(command, shell=True)
#     if process.returncode != 0:
#         print(f"Command failed: {command}")
#         break
# 명령어 실행
# 디렉토리와 명령어 매핑
import os
directories_commands = [
    ("../05matbert_Hybrid", hybrid_command),
    ("../05matbert_Word", word_command),
    ("../05matbert_String", char_command),
]
failed_commands = []
for directory, commands in directories_commands:
    print(directory)
    os.chdir(directory) 
    print(os.path.abspath(os.curdir) )# 작업 디렉토리 변경
    for command in commands:
        print(command)
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            print(f"Command failed: {command}")
            failed_commands.append(command)
for fc in failed_commands:
    print("failed command:" ,failed_commands)
    # os.chdir("..")  # 작업 디렉토리를 원래 상태로 복원