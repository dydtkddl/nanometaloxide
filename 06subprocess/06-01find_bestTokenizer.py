import subprocess
# 명령어 리스트
word_command = [    
    "python train_word_only.py --bert allenai/scibert_scivocab_uncased --threshold 0.5",
    "python train_word_only.py --bert allenai/scibert_scivocab_cased --threshold 0.5",
    "python train_word_only.py --bert m3rg-iitd/matscibert --threshold 0.5",
    "python train_word_only.py --bert pranav-s/MaterialsBERT --threshold 0.5",
    "python train_word_only.py --bert matbert-base-uncased --threshold 0.5",
    "python train_word_only.py --bert matbert-base-cased --threshold 0.5",
    "python train_word_only.py --bert bert-base-cased --threshold 0.5",
    "python train_word_only.py --bert bert-base-uncased --threshold 0.5",
]
hybrid_command = [    
    # "python train.py --bert allenai/scibert_scivocab_uncased --threshold 0.5",
    # "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5",
    # "python train.py --bert m3rg-iitd/matscibert --threshold 0.5",
    # "python train.py --bert pranav-s/MaterialsBERT --threshold 0.5",
    # "python train.py --bert matbert-base-uncased --threshold 0.5",
    # "python train.py --bert matbert-base-cased --threshold 0.5",
    # "python train.py --bert bert-base-cased --threshold 0.5",
    # "python train.py --bert bert-base-uncased --threshold 0.5",
]

import os
directories_commands = [
    ("../05matbert_Hybrid", hybrid_command),
    ("../05matbert_Word", word_command),
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