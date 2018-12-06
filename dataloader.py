import json
import youtube_dl
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import read

import pytube
from pytube import YouTube
import os

def load_from_youtube():
    json_files = ['MUSIC_dataset/MUSIC_solo_videos.json', 'MUSIC_dataset/MUSIC_duet_videos.json']
    performance_type = ['solo', 'duet']

    for idx, filepath in enumerate(json_files):
        # Parse json data
        with open(filepath) as f:
            data = json.load(f)
        videos = data['videos']
        # Download videos and audios
        os.chdir('MUSIC_dataset')
        for category in videos.keys():
            if not os.path.exists('{}_videos'.format(performance_type[idx])):
                os.mkdir('{}_videos'.format(performance_type[idx]))
            # if not os.path.exists('{}_audios'.format(performance_type[idx])):
            #     os.mkdir('{}_audios'.format(performance_type[idx]))
            for key in videos[category]:
                url = 'https://www.youtube.com/watch?v={}'.format(key)
                try:
                    yt = YouTube(url)
                except pytube.exceptions.RegexMatchError as e:
                    continue
                # Download video
                video_stream = yt.streams.filter(file_extension='mp4').all()[0]
                os.chdir('{}_videos'.format(performance_type[idx]))
                if not os.path.exists(category):
                    os.mkdir(category)
                video_stream.download(category)
                # Download audio
                # if len(yt.streams.filter(only_audio=True).all()) is not 0:
                #     audio_stream = yt.streams.filter(only_audio=True).all()[0] #may be empty list
                #     os.chdir('../{}_audios'.format(performance_type[idx]))
                #     if not os.path.exists(category):
                #         os.mkdir(category)
                #     audio_stream.download(category)
                os.chdir('..')
        os.chdir('..')

import skvideo.io
def vid_to_numpy(filename):
    return skvideo.io.vread(filename) # T*H*W*C

import moviepy.editor as mpy
import random
from scipy import signal
from scipy import ndimage
filename = 'MUSIC_dataset/solo_videos/flute/Flute Solo Country Gardens.mp4'

def trim_video(filename, trim_length=6.0, subsample_rate=11000):
    '''
    Randomly crops from untrimmed videos during training.

    Args:
        filename: str, name of the mp4 file.
        trim_length: float, the length (in sec) of trimmed video

    Return:
        Dictionary
            audio: numpy.array, array of audio samples (trim_length*subsample_rate,)
            video: numpy.array, array of video samples (trim_length*clip.fps, height, width, channel) to be fed in Video Subnetwork
    '''
    clip = mpy.VideoFileClip(filename)
    start = random.randint(0, (int)(clip.duration-trim_length))
    clip = clip.subclip(start, start+trim_length)
    audio_clip = clip.audio
    video_array = None
    for i, frame in enumerate(clip.iter_frames()):
        if i == 0:
            video_array = np.expand_dims(frame, axis=0)
        else:
            video_array = np.concatenate((video_array, np.expand_dims(frame, axis=0)), axis=0)
    audio_array = audio_clip.to_soundarray()    # AudioFileClip to numpy.array
    audio_array = signal.resample(audio_array, (int)(trim_length*subsample_rate))   # Subsample the original signal
    audio_array = np.mean(audio_array, axis=1)  # stereo to mono
    return {'audio': audio_array, 'video': video_array}

def to_db_spectrogram(audio_array, window_size=1022, hop_length=258):
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
