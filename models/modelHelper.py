import os
import random
import torch
import librosa
import numpy as np
class ModelHelper():
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.noisy_directory = './data/FSDnoisy18k.audio_test/'
        self.noisy_file_type = '.wav'

    def get_noise_file(self):
        files = [f for f in os.listdir(self.noisy_directory) if f.endswith(self.noisy_file_type)]
        random_file = random.choice(files)
        random_file_path = os.path.join(self.noisy_directory, random_file)
        sound = librosa.load(random_file_path, sr=16000, mono=True)[0]
        return sound

    def add_noise_to_file(self, speech):

        if len(speech) < 100000:
            speech = np.pad(speech, (0, 100000 - len(speech)), 'constant')
        else:
            speech = speech[:100000]

        speech_noisy = speech.copy()

        amount_of_noise = random.randint(2, 4)
        for _ in range(amount_of_noise):

            noisee = self.get_noise_file()
            random_index = random.randint(0, len(speech) - len(noisee))

            # Додавання елементів з другого масиву до першого починаючи з випадкового індексу
            for i in range(len(noisee)):
                speech_noisy[random_index + i] += noisee[i]

        return speech_noisy

    def demo_filtering(self, path_to_audio):
        audio = librosa.load(path_to_audio, sr=16000)[0]
        noisy_audio = self.add_noise_to_file(audio)
        filtered_audio = self.filter_from_audio(noisy_audio)
        return audio, noisy_audio, filtered_audio
    def filter_from_file(self, file_path):
        librosa_audio = librosa.load(file_path, sr=16000)[0]
        return self.filter_from_audio(librosa_audio)

    def filter_from_audio(self, audio):
        audion_chunks = []
        for i in range(0, len(audio), 100000):
            audion_chunks.append(audio[i:i + 100000])
        filtered_chunks = []
        for chunk in audion_chunks:
            ch = torch.from_numpy(chunk).float().to(self.device)
            ch = ch.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                filtered = self.model(ch)
            filtered = filtered.squeeze().cpu().numpy()
            filtered_chunks.append(filtered)
        filtered = np.concatenate(filtered_chunks)
        audio = np.concatenate(audion_chunks)
        return audio, filtered