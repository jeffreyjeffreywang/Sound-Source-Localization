import os
import json
import random
import pytube
from pytube import YouTube
import moviepy.editor as mpy

import numpy as np
from scipy import signal
from scipy import ndimage
import torch
import torch.utils.data as data

import librosa
import librosa.core as lc

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in ['.mp4', '.avi'])

def load_from_youtube(json_filename='MUSIC_dataset/MUSIC_solo_videos.json', performance_type='solo'):
    '''
    Load YouTube videos to local directory
    '''
    # Parse json data
    with open(json_filename) as f:
        data = json.load(f)
    videos = data['videos']
    # Download videos and audios
    os.chdir('MUSIC_dataset')
    for category in videos.keys():
        if not os.path.exists('{}_videos'.format(performance_type)):
            os.mkdir('{}_videos'.format(performance_type))
        for key in videos[category]:
            url = 'https://www.youtube.com/watch?v={}'.format(key)
            try:
                yt = YouTube(url)
            except pytube.exceptions.RegexMatchError as e:
                continue
            # Download video
            video_stream = yt.streams.filter(file_extension='mp4').all()[0]
            os.chdir('{}_videos'.format(performance_type))
            if not os.path.exists(category):
                os.mkdir(category)
            video_stream.download(category)
            os.chdir('..')
    os.chdir('..')

def build_video_dict(filepath='MUSIC_dataset/solo_videos'):
    '''
    Build a video dictionary.

    Return:
        Dictionary
            key: category
            value: a list of video filenames in the specified category
    '''
    video_dict = {}
    num_per_category = {}
    total = 0
    for category in os.listdir(filepath):
        if category.startswith('.'):
            continue
        else:
            if category not in video_dict.keys():
                video_dict[category] = []
                num_per_category[category] = 0
            for filename in os.listdir(os.path.join(filepath, category)):
                if is_video_file(filename):
                    video_dict[category].append(filename)
                    num_per_category[category] += 1
                    total += 1
    return video_dict, num_per_category, total

# filename = 'MUSIC_dataset/solo_videos/flute/Flute Solo Country Gardens.mp4'

def trim_video(filename, trim_length=6.0, subsampling_rate=11000, window_size=1022, hop_length=258):
    '''
    Randomly crops from untrimmed videos during training.

    Args:
        filename: str, name of the mp4 file.
        trim_length: float, the length (in sec) of trimmed video

    Return:
        Dictionary
            audio: torch.Tensor, tensor of audio samples (trim_length*subsampling_rate,)
            video: torch.Tensor, tensor of video samples (trim_length*clip.fps, height, width, channel) to be fed in Video Subnetwork
    '''
    clip = mpy.VideoFileClip(filename)
    clip = clip.resize((256,256))
    start = random.randint(0, (int)(clip.duration-trim_length))
    clip = clip.subclip(start, start+trim_length)
    audio_clip = clip.audio
    video_array = None
    for i, frame in enumerate(clip.iter_frames()):
        if i == 0:
            video_array = np.expand_dims(frame, axis=0)
        else:
            video_array = np.concatenate((video_array, np.expand_dims(frame, axis=0)), axis=0)
    # Choose T=3 frames
    perm = np.random.permutation(video_array.shape[0])
    video_array = video_array[perm[:3],:,:,:]
    audio_array = audio_clip.to_soundarray()    # AudioFileClip to numpy.array
    audio_array = signal.resample(audio_array, (int)(trim_length*subsampling_rate))   # Subsample the original signal
    audio_array = np.mean(audio_array, axis=1)  # stereo to mono
    audio_array = to_db_spectrogram(audio_array, window_size, hop_length)
    return {'audio': torch.from_numpy(audio_array), 'video': torch.from_numpy(video_array)}

def to_db_spectrogram(audio_array, window_size, hop_length):
    '''
    Convert mono audio array to a log spectrogram.

    Args:
        audio_array: numpy.array, mono audio array

    Return:
        numpy.array, 256*256 T-F representation on a log-frequency scale to be fed in Audio Subnetwork
    '''
    stft = np.abs(lc.stft(audio_array, n_fft=window_size, hop_length=hop_length))   # stft.shape = (512, 256)
    db_spec = lc.amplitude_to_db(stft, ref=np.max)
    # Re-sample db_spec on a log-frequency scale to obtain a 256*256 T-F representation
    db_spec_resampled = ndimage.zoom(db_spec, (0.5,1))
    return db_spec_resampled

class MUSIC(data.Dataset):
    def __init__(self, basepath, filename, performance_type, trim_length=6.0, subsampling_rate=11000, window_size=1022, hop_length=258):
        super(MUSIC, self).__init__()
        self.basepath = basepath    # 'MUSIC_dataset/solo_videos'
        self.filename = filename # json filename
        self.performance_type = performance_type # solo or duet
        self.trim_length = trim_length
        self.subsampling_rate = subsampling_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.video_dict, self.num_per_category, self.total_videos = build_video_dict('MUSIC_dataset/{}_videos'.format(self.performance_type))

    def __getitem__(self, index):
        '''
        Randomly returns a trimmed video and its corresponding audio spectrogram.

        Args:
            index: int, 0 by default
        '''
        category = random.choice(self.video_dict.keys())
        filename = os.path.join(self.basepath, category, random.choice(self.video_dict[category]))
        item = trim_video(filename, self.trim_length, self.subsampling_rate)
        return item

    def __len__(self):
        return self.total_videos

def mix(audio1, audio2):
    '''
    Mix two soundtracks to perform mix and separate framework.
    '''
    return torch.add(audio1, audio2)

# import youtube_dl
# from pydub import AudioSegment
# from scipy.io.wavfile import read
# # Download youtube videos
# ydl_opts = {}
# with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#     ydl.download(['https://www.youtube.com/watch?v=8DHG_hVSw1o'])
# # youtube-dl -i --extract-audio --audio-format mp3 --audio-quality 0 YT_URL
#
# # Convert mp3 file to wav file
# filename = 'Bad Romance - Flute and Violin cover-8DHG_hVSw1o.mp3'
# audio = AudioSegment.from_mp3(filename)
# wav_filename = filename.split('.')[0]+'.wav'
# audio.export(wav_filename, format="wav")
# sample_rate, audio_data = read(wav_filename)
#
# # Source separation
# signal = nussl.AudioSignal(path_to_input_file=wav_filename)
# nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_iterations=10, random_seed=0)
# nmf_mfcc.run()
# sources = nmf_mfcc.make_audio_signals()
# for i, source in enumerate(sources):
#     output_file_name = str(i) + '.wav'
#     source.write_audio_to_file(output_file_name)
