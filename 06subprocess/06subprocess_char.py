import subprocess
TRAIN_VAL_LOG_PATH = "../06-1processResults/TRAIN_VAL_LOG/Char"
TRAIN_LOSS_AVG_IMAGE_PATH = "../06-1processResults/TRAIN_LOSS_AVG_IMAGE/Char"
VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH = "../06-1processResults/VALIDATION_METRIC_IMAGE_NANOPARTICLE/Char"
VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH = "../06-1processResults/VALIDATION_METRIC_IMAGE_MAIN_SUBJECT/Char"
VALIDATION_METRIC_IMAGE_AVG_PATH = "../06-1processResults/VALIDATION_METRIC_IMAGE_AVG/Char"
TEST_METRIC_LOG_PATH = "../06-1processResults/TEST_METRIC_LOG/Char"
TEST_METRIC_IMAGE_NANOPARTICLE_PATH = "../06-1processResults/TEST_METRIC_IMAGE_NANOPARTICLE/Char"
TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH = "../06-1processResults/TEST_METRIC_IMAGE_MAIN_SUBJECT/Char"
TEST_METRIC_IMAGE_AVG_PATH = "../06-1processResults/TEST_METRIC_IMAGE_AVG/Char"
MODEL_WEIGHT_PATH = "../06-1processResults/MODEL_WEIGHT/Char"
main_command = '''python train_char_only.py --train_val_log_path %s --train_loss_avg_image_path %s --validation_metric_image_nanoparticle_path %s --validation_metric_image_main_subject_path %s --validation_metric_image_avg_path %s --test_metric_log_path %s --test_metric_image_nanoparticle_path %s --test_metric_image_main_subject_path %s --test_metric_image_avg_path %s --model_weight_path %s'''%(TRAIN_VAL_LOG_PATH,TRAIN_LOSS_AVG_IMAGE_PATH,VALIDATION_METRIC_IMAGE_NANOPARTICLE_PATH,VALIDATION_METRIC_IMAGE_MAIN_SUBJECT_PATH,VALIDATION_METRIC_IMAGE_AVG_PATH,TEST_METRIC_LOG_PATH,TEST_METRIC_IMAGE_NANOPARTICLE_PATH,TEST_METRIC_IMAGE_MAIN_SUBJECT_PATH,TEST_METRIC_IMAGE_AVG_PATH,MODEL_WEIGHT_PATH)
main_command +=''' --threshold 0.5'''
Char_command= [   
    main_command
]
import os
directories_commands = [
    ("../05matbert_String", Char_command),
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