# Training data sources

Every dataset used to train Urdu Sukhan ASR v1, with its licence and the credit it requires.

| Dataset | What it is | Licence | Credit | Clips used |
|---|---|---|---|---:|
| Audiobook recordings and transcripts by Asad Shehzad | Two audiobooks; transcripts drafted by earlier models and proofread by the author | the author's own material | Asad Shehzad | 521 |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices), Urdu | Read and conversational speech from many speakers (AI4Bharat) | CC BY 4.0 | AI4Bharat | 20,000 |
| Common Voice Urdu (Kaggle repack "Urdu dataset 20000") | Read sentences, many speakers (Mozilla) | CC0 | Mozilla Common Voice (courtesy) | 12,000 |
| [UAT](https://huggingface.co/datasets/mahwizzzz/UAT) | Audio from books; its machine-made transcripts were replaced by our own labels (confidence ≥ 0.85) | no public licence; used by agreement with its developers | the UAT authors | 10,000 |
| [UrduSpeech](https://huggingface.co/datasets/humairawan/UrduSpeech) | Urdu part of AI4Bharat's Rasa corpus | CC BY 4.0 | AI4Bharat | 4,999 |
| [nzg73 Urdu_Audio-Realistic](https://huggingface.co/datasets/nzgnzg73/Urdu_Audio-Realistic) | One male speaker, about 20 h | the dataset's own [licence](https://huggingface.co/datasets/nzgnzg73/Urdu_Audio-Realistic/blob/main/LICENSE): free to train and evaluate speech models, credit required, no misuse such as deepfakes | Muhammad Noman – nzg73 | 3,000 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) Urdu (train, dev) and English (train) | Read Wikipedia sentences (Google) | CC BY 4.0 | FLEURS, Google | 4,370 |
| Urdu + English code-switch clips | Built by joining FLEURS / Common Voice Urdu clips with FLEURS English clips | as their sources | as their sources | 782 |
| [MUSAN](https://www.openslr.org/17/) noise and instrumental music | No-speech clips and background noise for augmentation; files held out for the stress test are never used | CC BY 4.0 (some music tracks carry their own Creative Commons licences) | Snyder, Chen and Povey (2015) | 2,800 no-speech clips |

Base model: [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT).
