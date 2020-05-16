%tensorflow_version 1.x
!pip install -q gpt-2-simple
import gpt_2_simple as gpt2
from datetime import datetime
from google.colab import files

#download the model
gpt2.download_gpt2(model_name="124M")
file_name = "original.txt"


#finetune Gpt-2
sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=1000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=200,
              save_every=500
              )

#generation
gpt2.generate(sess,
              length=150,
              temperature=0.7,
              prefix="Two households, both alike in dignity",
              nsamples=5,
              batch_size=5
              )

gpt2.generate(sess,
              length=150,
              temperature=0.7,
              prefix="Who's there?",
              nsamples=5,
              batch_size=5
              )

gpt2.generate(sess,
              length=150,
              temperature=0.7,
              prefix="Who's there? Nay, answer me: stand, and unfold yourself. Long live the king!",
              nsamples=5,
              batch_size=5
              )

gpt2.generate(sess,
              length=150,
              temperature=0.7,
              prefix="Who's there? Nay, answer me: stand, and unfold yourself. Long live the king! Bernardo? He. You come most carefully upon your hour. Tis now struck twelve; get thee to bed, Francisco.",
              nsamples=5,
              batch_size=5
              )