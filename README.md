# audio2mel_preprocessor
A tool aims to transform audio to mel-spectrogram for speech dataset. 

You can use it to prepare the data for training acoustic model(eg. Tacotron2)/vocoder(eg. MelGAN, Hifi-GAN).

## Setup
1. Download or prepare your speech datasets.

2. Transform them from audio to mel-spectrogram(saved as numpy files).
`python preprocess.py --dataset=DataBaker --indir=path/BZNSYP --outdir=./training_data`  

3. To support more dataset type, you can define additional processing script in "./datasets/", just refer to "ljspeech.py" & "databaker.py".(Welcome to commit!)

5. The data will be processed as:

 outdir/

  |--train.txt (the format can be modified in preprocess.py(def write_metadata))
  
  |--audio/
  
  |--mels/
  
  |--linear/


## Supplementary
1. "MultiSets" is used for multi-speaker or multilingual dataset.

2. "config.json" is used to extract mel-spectrogram under different acoustic parameters, we provide 16k and 22.5k as reference. 

    
