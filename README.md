This is an implementation of the method described in: "multi-source cross-domain vulnerability detection based on code pre-trained model".

# Setup
1) Clone this repo
   ```shell
   git clone https://github.com/Iretinn/msvd.git
   ```
2) Install the python dependencies
   ```
   pip install -r requirements.txt
   ```
3) Download the data
   
   Download from [google drive](https://drive.google.com/drive/folders/1H-nYa9v7n80j57R_GJrgK9wQfBS0q81B?usp=drive_link), and move the data to `<project root>/data` folder.
   
5) Download the code pre-trained model
   
   Download [CodeBERT](https://huggingface.co/microsoft/codebert-base/tree/main), and move the pre-trained model to `<project root>/code/codebert` folder.

   ```
   msvd
    |--data
    |    |--cross-language
    |    |     |--xxx.csv
    |    |--cross-project
    |          |--xxx.csv
    |--code
         |--analysis
         |     |--xxx.py
         |----msvd
         |     |--xxx.py
         |--codebert
               |--config.json
               |--merges.txt
               |--pytorch_model.bin
               |--special_tokens_map.json
               |--tokenizer_config.json
               |--vocab.json
   ```

# Experiment
