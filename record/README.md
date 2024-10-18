# Robot Command Recording App

This app allows you to generate and record commands for a robot dog using OpenAI's GPT model. The commands are saved as both text and audio for future use, particularly for training purposes. The generated data is stored in Parquet files.

## Features

- **Command Generation**: Generates randomized robot dog commands using OpenAI's GPT model.
- **Audio Recording**: Records voice commands via a microphone.
- **Data Storage**: Saves the commands and audio in Parquet format for easy retrieval and analysis.
- **Customizable Prompt**: Users can modify the system prompt to control the type of commands generated.

## Requirements

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage

### 1. System Prompt File
The app uses a `system_prompt.txt` file to define the type of instructions generated for the robot. This can be customized based on your needs. By default, the prompt is aimed at generating robot movement instructions.

### 2. Recording Commands

The app will display the command, and you can record your voice command based on the provided instructions. The recorded data (both text and audio) is stored in a Parquet file in the [audio](/audio) folder in the root of the project when you click the **Write to file** button.

### 3. Saving Data
Once you're done recording a batch of commands, press the **Write to file** button. This will save the current set of commands and their corresponding audio files into a Parquet file named `whisper_training_data_N.parquet`, where `N` is the collection number to avoid overwriting previous files.

### 4. Command-Line Argument
The application allows passing the path to a custom system prompt file via the `--system-prompt-file` argument.

Example:
```bash
streamlit run app.py -- --system-prompt-file custom_prompt.txt
```

### 5. OpenAI API Key
Ensure you have set your `OPENAI_API_KEY` as an environment variable to enable the sentence generation.

```bash
export OPENAI_API_KEY="your-api-key"
```

## How It Works

1. **Generate Commands**: 
   - The app generates random commands for the robot based on the system prompt provided.
   
2. **Record Audio**: 
   - The user is prompted to record their voice corresponding to each command.
   
3. **Save Data**: 
   - The text and audio data are stored in Parquet format for training purposes.
   
## Running the App

You can run the app using the following command:

```bash
streamlit run app.py
```

You can also provide a custom system prompt file:

```bash
streamlit run app.py -- --system-prompt-file "custom_prompt.txt"
```

## License

This project is licensed under the MIT License.