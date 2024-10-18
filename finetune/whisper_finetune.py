import pandas as pd
from datasets import Dataset, load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import numpy as np
from pydub import AudioSegment
import io



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


dataset = load_dataset('parquet', data_files = '../audio/*.parquet', streaming = False)
dataset = dataset['train']

# Function to process the audio data
def process_audio(batch):
    audio_data = batch['audio']
    format = batch['format'].lower()  # Ensure the format is in lowercase

    # Read the audio data from bytes
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format)

    # Convert to 16kHz mono audio
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Export audio to raw data
    raw_audio = audio.raw_data

    # Convert raw audio to numpy array
    samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize between -1 and 1

    # Return the processed audio
    return {'audio_array': {'array': samples, 'sampling_rate': 16000}}

# Apply the audio processing function to the dataset
dataset = dataset.map(process_audio, remove_columns=['audio', 'format', 'channels', 'sample_rate'])

# Initialize the Whisper processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Function to prepare the dataset for training
def prepare_dataset(batch):
    # Get the processed audio and text
    audio = batch['audio_array']
    text = batch['sentence']

    # Tokenize the audio
    input_features = processor(audio['array'], sampling_rate=audio['sampling_rate'], return_tensors="pt").input_features[0]

    # Tokenize the text
    labels = processor.tokenizer(text, return_tensors="pt").input_ids[0]

    batch['input_features'] = input_features
    batch['labels'] = labels
    return batch

# Prepare the dataset
dataset = dataset.map(prepare_dataset, remove_columns=['audio_array', 'sentence'])

# Set the dataset format for PyTorch
dataset.set_format(type='torch', columns=['input_features', 'labels'])

train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test['train']
eval_dataset = train_test['test']
train_test = None
dataset = None

# Load the Whisper model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Configure the model for English transcription
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

use_fp16 = torch.cuda.is_available()

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="../whisper_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=2,
    fp16=use_fp16,
    predict_with_generate=True,
)

# Data collator for padding inputs and labels
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
# Initialize the Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

# Start the fine-tuning process
trainer.train()
