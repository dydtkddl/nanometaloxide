git lfs clone https://huggingface.co/pranav-s/MaterialsBERT
git lfs clone https://huggingface.co/google-bert/bert-base-uncased

git lfs clone https://huggingface.co/google-bert/bert-base-cased
git lfs clone https://huggingface.co/m3rg-iitd/matscibert
git lfs clone https://huggingface.co/allenai/scibert_scivocab_cased
git lfs clone https://huggingface.co/allenai/scibert_scivocab_uncased
export MODEL_PATH="./"
mkdir $MODEL_PATH/matbert-base-cased $MODEL_PATH/matbert-base-uncased
curl -# -o $MODEL_PATH/matbert-base-cased/config.json https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/config.json
curl -# -o $MODEL_PATH/matbert-base-cased/vocab.txt https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/vocab.txt
curl -# -o $MODEL_PATH/matbert-base-cased/pytorch_model.bin https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/pytorch_model.bin
curl -# -o $MODEL_PATH/matbert-base-uncased/config.json https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_uncased_30522_wd/config.json
curl -# -o $MODEL_PATH/matbert-base-uncased/vocab.txt https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_uncased_30522_wd/vocab.txt
curl -# -o $MODEL_PATH/matbert-base-uncased/pytorch_model.bin https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_uncased_30522_wd/pytorch_model.bin