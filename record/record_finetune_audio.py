import streamlit as st
import pandas as pd
import json
import random
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI

if 'prev_messages' not in st.session_state:
    st.session_state.prev_messages = []

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI()

# State initialization
if 'data' not in st.session_state:
    st.session_state.data = []
    st.session_state.current_sentences = []
    st.session_state.collection_index = 1  # For numbering the Parquet files

def generate_sentences():
    try:
        response = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "Create simple and clear instructions for a robot dog to follow directions accurately, including movements such as moving forward, turning, and changing directions. You will receive a prompt number, you should return this many commands. Also, randomize what you return, the same prompt should return different results. \n\n- Use concise and unambiguous language.\n- Ensure that each instruction corresponds to a distinct action.\n- Focus on basic movements like moving forward, turning left, turning right, and stopping.\n\n# Steps\n\n1. **Determine the Direction**: Specify the initial direction and angle, if necessary.\n2. **Define the Movement**: Choose whether the robot should move forward, turn left, turn right, or stop.\n3. **Specify the Distance or Degrees**: If moving forward or turning, indicate the specific distance or degrees to turn.\n4. **Sequence the Instructions**: Ensure each instruction follows logically from the previous one.\n\n# Output Format\n\n- Provide each instruction on a separate line.\n- Use a format such as: \"ACTION [PARAMETER]\"\n  - Example: \"MOVE FORWARD 3 meters\", \"TURN LEFT 90 degrees\", \"STOP\"\n\n# Examples\n\n- Input: Next\n- Output:\n  - MOVE FORWARD 5 meters\n  - TURN RIGHT 90 degrees\n  - STOP\n"
                }
            ]
            },
            *st.session_state.prev_messages,
            {
                "role": "user",
                "content": "20"
            }
        ],
        seed = random.randint(1, 1000000),
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
            "name": "sentence_list",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                "sentences": {
                    "type": "array",
                    "description": "A list of sentences.",
                    "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                        "type": "string",
                        "description": "The content of the sentence."
                        }
                    },
                    "required": [
                        "text"
                    ],
                    "additionalProperties": False
                    }
                }
                },
                "required": [
                "sentences"
                ],
                "additionalProperties": False
            }
            }
        }
        )
        sentences = json.loads(response.choices[0].message.content)["sentences"]
        st.session_state.prev_messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return sentences

    except Exception as e:
        st.error(f"Error generating sentence: {e}")
        return None

def main(system_prompt_file):
    with open(system_prompt_file, 'r') as file:
        st.session_state.system_prompt = file.read()

    st.subheader("Robot Command Recording App")
    st.caption("This app allows you to record commands for a robot dog to follow. To run, make sure to set your OPENAI_API_KEY for the sentence generator to work.")
    st.caption("You can record as many commands as you want, and they will be saved in a Parquet file.")
    st.caption("You can press the 'Write to file' button to save the data and start a new collection.")
    st.caption("You can change/repurpose the system prompt in the 'system_prompt.txt' file or use a different one and override it with the --system-prompt-file / -f argument.")


    if 'finish_clicked' not in st.session_state:
        st.session_state.finish_clicked = False

    # Finish button to save data and start a new collection
    if st.button("Write to file"):
        st.session_state.finish_clicked = True

    if st.session_state.finish_clicked:
        save_data()
        # Reset for a new collection
        st.session_state.finish_clicked = False
        st.session_state.data = []
        st.rerun()
        return

    if st.session_state.current_sentences == []:
        # Generate a new sentence
        st.session_state.current_sentences = generate_sentences()
        if not st.session_state.current_sentences:
            st.stop()  # Stop execution if sentence generation failed

    sentence = st.session_state.current_sentences[0]
    st.subheader("Please record the following command:")
    st.write(f"**{sentence['text']}**")

    # Record audio using mic_recorder
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        key=f"recorder_{len(st.session_state.data)}"
    )

    if audio and audio['bytes'] != b'':
        # Save the audio file
        # audio_filename = f"sentence_{len(st.session_state.data) + 1}.wav"
        # with open(audio_filename, 'wb') as f:
        #     f.write(audio['bytes'])

        # Log the sentence and audio path
        st.session_state.data.append({
            "sentence": sentence['text'],
            "audio":  audio['bytes'],
            "format": audio['format'],
            "channels": audio['sample_width'],
            "sample_rate": audio['sample_rate'],

        })

        st.session_state.current_sentences = st.session_state.current_sentences[1:len(st.session_state.current_sentences)]

        # Rerun to display the next sentence
        st.rerun()

def save_data():
    if st.session_state.data:
        df = pd.DataFrame(st.session_state.data)
        filename = f"../audio/whisper_training_data_{st.session_state.collection_index}.parquet"
        df.to_parquet(filename, index=False)
        st.session_state.collection_index += 1
        st.success(f"Training data saved to {filename}")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--system-prompt-file", "-f", type=str, default="system_prompt.txt", help="Path to the file containing the system prompt")
    args = parser.parse_args()

    main(args.system_prompt_file)
