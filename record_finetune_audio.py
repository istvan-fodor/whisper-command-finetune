import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from streamlit_mic_recorder import mic_recorder
import os
import transformers
import torch
from transformers import AutoTokenizer

# Initialize the text generation pipeline

model = "meta-llama/Llama-3.2-3B"

@st.cache_resource
def load_generator():
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return pipeline

@st.cache_resource
def create_tokenizer():
    return AutoTokenizer.from_pretrained(model)

tokenizer = create_tokenizer()
pipeline = load_generator()

# State initialization
if 'data' not in st.session_state:
    st.session_state.data = []
    st.session_state.current_sentence = ""
    st.session_state.collection_index = 1  # For numbering the Parquet files

def generate_sentence():
    prompt = prompt = """
                You are a synthetic data generator generating a single sentence command to a robot dog.
                Examples are like this:
                1. Go ahead one meter.
                2. Turn left ninety degrees.
                3. Go right.
                4. Go back three steps.

                Only return a single command!

                Command:"""


    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=pipeline.tokenizer.eos_token_id,
        max_length=200,

    )
    text = sequences[0]['generated_text']
    print(text)
    # Extract the generated command after "Command:"
    sentence = text.split("Command:")[-1].strip().split('\n')[0]

    return sentence

def main():
    st.title("Robot Dog Command Recording App")

    if 'finish_clicked' not in st.session_state:
        st.session_state.finish_clicked = False

    # Finish button to save data and start a new collection
    if st.button("Finish"):
        st.session_state.finish_clicked = True

    if st.session_state.finish_clicked:
        save_data()
        # Reset for a new collection
        st.session_state.finish_clicked = False
        st.session_state.data = []
        st.session_state.current_sentence = ""
        st.rerun()
        return

    if st.session_state.current_sentence == "":
        # Generate a new sentence
        st.write("Generating a new sentence...")
        st.session_state.current_sentence = generate_sentence()

    sentence = st.session_state.current_sentence
    st.header("Please record the following command:")
    st.write(sentence)

    # Record audio using mic_recorder
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        key=f"recorder_{len(st.session_state.data)}"
    )

    if audio and audio['bytes'] != b'':
        # Save the audio file
        audio_filename = f"sentence_{len(st.session_state.data) + 1}.wav"
        with open(audio_filename, 'wb') as f:
            f.write(audio['bytes'])

        # Log the sentence and audio path
        st.session_state.data.append({
            "sentence": sentence,
            "audio_path": os.path.abspath(audio_filename)
        })

        st.success(f"Recorded and saved: {audio_filename}")

        # Reset current sentence to trigger generation of a new one
        st.session_state.current_sentence = ""

        # Rerun to display the next sentence
        st.rerun()

def save_data():
    if st.session_state.data:
        df = pd.DataFrame(st.session_state.data)
        table = pa.Table.from_pandas(df)
        # Use collection index to avoid overwriting files
        filename = f"whisper_training_data_{st.session_state.collection_index}.parquet"
        pq.write_table(table, filename)
        st.session_state.collection_index += 1
        st.success(f"Training data saved to {filename}")

if __name__ == "__main__":
    main()
