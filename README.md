# voice-cloning-model

Data Collection: Gather a dataset containing parallel recordings of multiple speakers, including the source and target speakers. Each recording should contain the same content spoken by different speakers.

Feature Extraction: Extract relevant acoustic features from the speech signals, such as pitch, etc. These features capture the spectral and temporal characteristics of the audio.

Model Training; Model is trained on the segmented .wav files.

After the model is trained is trained we feed in our .wav input file and the model outputs a .wav file which will be the audio of the person on which the model has been trained, i.e. in this case kanye west and elon musk.
