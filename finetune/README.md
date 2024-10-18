# Whisper Model Fine-Tuning for Speech Recognition

This project fine-tunes OpenAI's Whisper model on a the recorded dataset. The dataset is processed, tokenized, and trained using Hugging Face's `transformers` library, and the audio data is preprocessed using `pydub`.

Data is loaded from the `../audio/*.parquet` source and you should record files there first with the recording component of this project.

## Requirements

To run this project, you need to install the torch deps first. Use the [Get Started with Torch](https://pytorch.org/get-started/locally/) guide for the right channel based on your setup (CUDA, ROCm, CPU, OSX vs Linux, etc):

```bash
#I personally used this with an AMD card, so installed ROCm:
pip install -r torch-requirements.txt --index-url https://download.pytorch.org/whl/rocm6.2
```

After this step install the rest of the deps. 
```bash
pip install -r requirements.txt
```

## Settings

This project finetunes the small Whisper model. Also, the training parameters are ideal for small datasets and small memory. If you use this code, you should play around with the parameters if you have high caliber hardware.

## How it Works

```bash
python whisper_finetune.py
```

Once the program finishes, it will store a checkpoint in the [whisper_finetuned](/whisper_finetuned) from the root. In other applications you can load Whisper from this folder.


## License

This project is licensed under the MIT License.