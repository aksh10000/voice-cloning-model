# voice-cloning-model

I have used a pretrained model so data collection, features extraction steps were not performed by me but some one else as I don't have the resources to trian the model either offline or online.

Training the Retrieval based Voice conversion model uses the following steps as mentioned by the owner of the model:

https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Instructions-and-tips-for-RVC-training

But since I used the pretrained model, I did not have to apply these steps.
 
After the model is trained is trained we feed in our .wav input file and the model outputs a .wav file which will be the audio of the person on which the model has been trained, i.e. in this case kanye west and elon musk.

To run the model copy of my RVC google colab notebook is linked below:

https://colab.research.google.com/drive/15jB1R-fH44YIQ_3jqkZ15arFNflV2CDr?usp=sharing

We have a pretty good pretrained model for elon musk voice which has been trained for over 200 epochs.

To evaluate the performance we can compare the real voice with the synthetic voice and judge the performace.

In order to make the model better we have to train it for more epochs with more input audio.
