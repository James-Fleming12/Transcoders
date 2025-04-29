
 # Transcoder for R1-distill-LLama-8b layer 3 training script



 We use a similar formula to https://arxiv.org/abs/2501.18823#:~:text=Transcoders%20are%20similar%20to%20SAEs,features%20are%20significantly%20more%20interpretable where we use a skip transcoder architecture, signum optimizer, Multi-TopK and train on the openwebtext corpus for around 1 billions tokens. 
 
 
We provide a colab notebook to look at features from our transcoder. We have not done steering or circuit analysis like in https://arxiv.org/pdf/2406.11944. We also do not have very many dead features(we followed EluetherAI's formula for training so this was expected). 

The hugginface for the transcoders is https://huggingface.co/matboz/TranscoderforDeepSeekR1DistillLlama8Blayer3

the notebook is here https://colab.research.google.com/drive/14f7QP1i-szN43bDz333KdiR5bV84lcYv#scrollTo=Q-tIdqashcOE.


