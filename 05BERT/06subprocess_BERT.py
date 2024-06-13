import subprocess
TRAIN_VAL_LOG_PATH = "../06-1processResults/TRAIN_VAL_LOG/BERT"
TRAIN_LOSS_AVG_IMAGE_PATH = "../06-1processResults/TRAIN_LOSS_AVG_IMAGE/BERT"
VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH = "../06-1processResults/VALIDATION_METRIC_IMAGE_NANOPARTICLE/BERT"
VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH = "../06-1processResults/VALIDATION_METRIC_IMAGE_MAIN_SUBJECT/BERT"
VALIDATION_METRIC_IMAGE_AVG_PATH = "../06-1processResults/VALIDATION_METRIC_IMAGE_AVG/BERT"
TEST_METRIC_LOG_PATH = "../06-1processResults/TEST_METRIC_LOG/BERT"
TEST_METRIC_IMAGE_NANOPARTICLE_PATH = "../06-1processResults/TEST_METRIC_IMAGE_NANOPARTICLE/BERT"
TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH = "../06-1processResults/TEST_METRIC_IMAGE_MAIN_SUBJECT/BERT"
TEST_METRIC_IMAGE_AVG_PATH = "../06-1processResults/TEST_METRIC_IMAGE_AVG/BERT"
MODEL_WEIGHT_PATH = "../06-1processResults/MODEL_WEIGHT/BERT"
SSHPATH = ""

TRAIN_VAL_LOG_PATH = "/RESULT/TRAIN_VAL_LOG/BERT_0606_2"
TRAIN_LOSS_AVG_IMAGE_PATH = "/RESULT/TRAIN_LOSS_AVG_IMAGE/BERT_0606_2"
VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH = "/RESULT/VALIDATION_METRIC_IMAGE_NANOPARTICLE/BERT_0606_2"
VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH = "/RESULT/VALIDATION_METRIC_IMAGE_MAIN_SUBJECT/BERT_0606_2"
VALIDATION_METRIC_IMAGE_AVG_PATH = "/RESULT/VALIDATION_METRIC_IMAGE_AVG/BERT_0606_2"
TEST_METRIC_LOG_PATH = "/RESULT/TEST_METRIC_LOG/BERT_0606_2"
TEST_METRIC_IMAGE_NANOPARTICLE_PATH = "/RESULT/TEST_METRIC_IMAGE_NANOPARTICLE/BERT_0606_2"
TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH = "/RESULT/TEST_METRIC_IMAGE_MAIN_SUBJECT/BERT_0606_2"
TEST_METRIC_IMAGE_AVG_PATH = "/RESULT/TEST_METRIC_IMAGE_AVG/BERT_0606_2"
MODEL_WEIGHT_PATH = "/RESULT/MODEL_WEIGHT/BERT_0606_2"
SSHPATH = "./repos/OXIDE"
DATAPATH = "./repos/OXIDE/purify_data.xlsx"


import datetime 
PROCESSID = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")

main_command = '''python ./repos/OXIDE/train.py --train_val_log_path %s --train_loss_avg_image_path %s --validation_metric_image_nanoparticle_path %s --validation_metric_image_main_subject_path %s --validation_metric_image_avg_path %s --test_metric_log_path %s --test_metric_image_nanoparticle_path %s --test_metric_image_main_subject_path %s --test_metric_image_avg_path %s --model_weight_path %s'''%(TRAIN_VAL_LOG_PATH,TRAIN_LOSS_AVG_IMAGE_PATH,VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH,VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH,VALIDATION_METRIC_IMAGE_AVG_PATH,TEST_METRIC_LOG_PATH,TEST_METRIC_IMAGE_NANOPARTICLE_PATH,TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH,TEST_METRIC_IMAGE_AVG_PATH,MODEL_WEIGHT_PATH)
main_command +=''' --threshold 0.5'''
BERT_command= [   
    main_command+ " --epochs 150 --bert ./repos/BERTS/bert-base-uncased --BERTNAME bert-base-uncased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 2e-05",
    main_command+ " --epochs 150 --bert ./repos/BERTS/bert-base-cased --BERTNAME bert-base-cased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 2e-05",
    
    main_command+ " --epochs 150 --bert ./repos/BERTS/scibert_scivocab_cased --BERTNAME scibert_scivocab_cased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 2e-05",
    main_command+ " --epochs 150 --bert ./repos/BERTS/scibert_scivocab_cased --BERTNAME scibert_scivocab_uncased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 2e-05",

    main_command+ " --epochs 150 --bert ./repos/BERTS/matbert-base-uncased --BERTNAME matbert-base-uncased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 2e-05",
    main_command+ " --epochs 150 --bert ./repos/BERTS/matbert-base-cased --BERTNAME matbert-base-cased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 2e-05",

    main_command+ " --epochs 150 --bert ./repos/BERTS/MaterialsBERT --BERTNAME MaterialsBERT" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 2e-05",
    main_command+ " --epochs 150 --bert ./repos/BERTS/matscibert --BERTNAME matscibert" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 2e-05",
]
BERT_COMMANDS = []
for i, command in enumerate(BERT_command):
    process_ID = PROCESSID + "_%s"%i 
    command = command + " --processID %s"%process_ID
    BERT_COMMANDS.append(command)
import os
BERT_COMMANDS2 = []
for i in range(5):
    RANDOM_STATE = (i+1)*11
    for c in BERT_COMMANDS:
        c = c+" --random_state %s"%RANDOM_STATE
        BERT_COMMANDS2.append(c)
print(BERT_COMMANDS2)
print("커맨드 수 : ", len(BERT_COMMANDS2))
directories_commands = [
    ("./repos/OXIDE", BERT_COMMANDS2),
]
failed_commands = []
for directory, commands in directories_commands:
    # print(directory)
    # os.chdir(directory) 
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

################################# Tokenizer 비교 분석 ##########################################
# main_command = '''python ./repos/OXIDE/train.py --train_val_log_path %s --train_loss_avg_image_path %s --validation_metric_image_nanoparticle_path %s --validation_metric_image_main_subject_path %s --validation_metric_image_avg_path %s --test_metric_log_path %s --test_metric_image_nanoparticle_path %s --test_metric_image_main_subject_path %s --test_metric_image_avg_path %s --model_weight_path %s'''%(TRAIN_VAL_LOG_PATH,TRAIN_LOSS_AVG_IMAGE_PATH,VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH,VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH,VALIDATION_METRIC_IMAGE_AVG_PATH,TEST_METRIC_LOG_PATH,TEST_METRIC_IMAGE_NANOPARTICLE_PATH,TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH,TEST_METRIC_IMAGE_AVG_PATH,MODEL_WEIGHT_PATH)
# main_command +=''' --threshold 0.5'''
# BERT_command= [   
#     main_command+ " --epochs 50 --bert ./repos/BERTS/bert-base-uncased --BERTNAME bert-base-uncased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 1e-05",
#     main_command+ " --epochs 50 --bert ./repos/BERTS/bert-base-cased --BERTNAME bert-base-cased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 1e-05",
    
#     main_command+ " --epochs 50 --bert ./repos/BERTS/scibert_scivocab_cased --BERTNAME scibert_scivocab_cased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 1e-05",
#     main_command+ " --epochs 50 --bert ./repos/BERTS/scibert_scivocab_cased --BERTNAME scibert_scivocab_uncased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 1e-05",

#     main_command+ " --epochs 50 --bert ./repos/BERTS/matbert-base-uncased --BERTNAME matbert-base-uncased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 1e-05",
#     main_command+ " --epochs 50 --bert ./repos/BERTS/matbert-base-cased --BERTNAME matbert-base-cased" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 1e-05",

#     main_command+ " --epochs 50 --bert ./repos/BERTS/MaterialsBERT --BERTNAME MaterialsBERT" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 1e-05",
#     main_command+ " --epochs 50 --bert ./repos/BERTS/matscibert --BERTNAME matscibert" + " --data_path %s"%DATAPATH + " --ssh_path %s"%SSHPATH + " --lr 1e-05",
# ]
# BERT_COMMANDS = []
# for i, command in enumerate(BERT_command):
#     process_ID = PROCESSID + "_%s"%i 
#     command = command + " --processID %s"%process_ID
#     BERT_COMMANDS.append(command)
################################# ----------------- ##########################################
