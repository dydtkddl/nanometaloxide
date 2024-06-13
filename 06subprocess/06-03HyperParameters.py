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
def make_commands(random_state):
    #########################################################
    hybrid_command= [   
        "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 16 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 256 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 32 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 64 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --batch_size 128 --random_state ",

        "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --max_len 64 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --max_len 128 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --threshold 0.5 --max_len 256 --random_state ",

        "python train.py --bert allenai/scibert_scivocab_cased --num_filters 5 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --num_filters 10 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --num_filters 25 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --num_filters 50 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --num_filters 100 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --num_filters 150 --random_state ",
        "python train.py --bert allenai/scibert_scivocab_cased --num_filters 200 --random_state ",
    ]
    for kernel_sizes in kernel_sizes_list:
        cmd = f"python train.py --bert allenai/scibert_scivocab_cased --kernel_sizes {kernel_sizes} --random_state "
        hybrid_command.append(cmd)
    new_hybrid_command= []
    for co in hybrid_command:
        co+=str(random_state)
        new_hybrid_command.append(co)
    return new_hybrid_command
hybrid_command = []
i_list = []
for i in range(10, 100,10):
    print("random_state : " + str(i))
    added = make_commands(i)
    hybrid_command+=added
    print(added)
    i_list.append(i)
print("총 몇번? : " , len(hybrid_command))
print("랜덤스테이트들? : " , i_list)
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
]
from tqdm import tqdm
failed_commands = []
for directory, commands in directories_commands:
    print(directory)
    os.chdir(directory) 
    print(os.path.abspath(os.curdir) )# 작업 디렉토리 변경
    for command in tqdm(commands):
        print(command)
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            print(f"Command failed: {command}")
            failed_commands.append(command)
for fc in failed_commands:
    print("failed command:" ,failed_commands)
    # os.chdir("..")  # 작업 디렉토리를 원래 상태로 복원