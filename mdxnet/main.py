import gc
import hashlib
import os
import queue
import threading
import warnings

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from tqdm import tqdm
from typing import Optional

warnings.filterwarnings("ignore")

stem_naming = {'Vocals': 'Instrumental', 'Other': 'Instruments', 'Instrumental': 'Vocals', 'Drums': 'Drumless', 'Bass': 'Bassless'}

class MDXModel:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.000):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = n_fft // 2 + 1
        self.chunk_size = hop * (dim_t - 1)
        self.window = torch.hann_window(window_length=n_fft, periodic=True).to(device)

        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins - dim_f, dim_t]).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 4, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])

class MDX:
    DEFAULT_SR = 44100
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR
    DEFAULT_PROCESSOR = 0

    def __init__(self, model_path: str, params: MDXModel, processor=DEFAULT_PROCESSOR):
        self.device = torch.device(f'cuda:{processor}') if processor >= 0 else torch.device('cpu')
        self.provider = ['CUDAExecutionProvider'] if processor >= 0 else ['CPUExecutionProvider']
        self.model = params

        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        self.ort.run(None, {'input': torch.rand(1, 4, params.dim_f, params.dim_t).numpy()})
        self.process = lambda spec: self.ort.run(None, {'input': spec.cpu().numpy()})[0]
        self.prog = None

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, 'rb') as f:
                f.seek(- 10000 * 1024, 2)
                return hashlib.md5(f.read()).hexdigest()
        except:
            return hashlib.md5(open(model_path, 'rb').read()).hexdigest()

    @staticmethod
    def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE):
        if combine:
            processed_wave = None
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if processed_wave is None:
                    processed_wave = segment[:, start:end]
                else:
                    processed_wave = np.concatenate((processed_wave, segment[:, start:end]), axis=-1)
            return processed_wave
        else:
            processed_wave = []
            sample_count = wave.shape[-1]
            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count
            if margin_size > chunk_size:
                margin_size = chunk_size

            for skip in range(0, sample_count, chunk_size):
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - (0 if skip == 0 else margin_size)
                processed_wave.append(wave[:, start:end].copy())
                if end == sample_count:
                    break
            return processed_wave

    def pad_wave(self, wave):
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        wave_p = np.concatenate((np.zeros((2, trim)), wave, np.zeros((2, pad)), np.zeros((2, trim))), 1)
        mix_waves = [wave_p[:, i:i+self.model.chunk_size] for i in range(0, n_sample + pad, gen_size)]
        return torch.tensor(mix_waves, dtype=torch.float32).to(self.device), pad, trim

    def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int):
        mix_waves = mix_waves.split(1)
        with torch.no_grad():
            pw = []
            for mix_wave in mix_waves:
                self.prog.update()
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(processed_spec.to(self.device))
                processed_wav = processed_wav[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).cpu().numpy()
                pw.append(processed_wav)
        processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        q.put({_id: processed_signal})

    def process_wave(self, wave: np.array, mt_threads=1):
        self.prog = tqdm(total=0)
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, False, chunk)

        q = queue.Queue()
        threads = []
        for c, batch in enumerate(waves):
            mix_waves, pad, trim = self.pad_wave(batch)
            self.prog.total = len(mix_waves) * mt_threads
            thread = threading.Thread(target=self._process_wave, args=(mix_waves, trim, pad, q, c))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.prog.close()

        processed_batches = [list(q.get().values())[0] for _ in range(len(waves))]
        return self.segment(processed_batches, True, chunk)

class MDXProcessor:
    stem_naming = stem_naming

    def __init__(self, model_path: str, model_params: dict, processor: int = 0):
        self.model_path = model_path
        self.model_params = model_params
        self.processor = processor
        self.device = torch.device(f'cuda:{processor}') if processor >= 0 else torch.device('cpu')
        
        self.model_hash = MDX.get_hash(model_path)
        self.model_config = self.model_params.get(self.model_hash)
        if not self.model_config:
            raise ValueError(f"Model parameters for hash {self.model_hash} not found")

        self.mdx_model = MDXModel(
            self.device,
            dim_f=self.model_config["mdx_dim_f_set"],
            dim_t=2 ** self.model_config["mdx_dim_t_set"],
            n_fft=self.model_config["mdx_n_fft_scale_set"],
            stem_name=self.model_config["primary_stem"],
            compensation=self.model_config["compensate"]
        )
        self.mdx_sess = MDX(model_path, self.mdx_model, processor=processor)

    def process(
        self,
        input_path: str,
        output_dir: str,
        exclude_main: bool = False,
        exclude_inversion: bool = False,
        suffix: Optional[str] = None,
        invert_suffix: Optional[str] = None,
        denoise: bool = False,
        keep_orig: bool = True,
        m_threads: Optional[int] = None
    ) -> tuple:
        if m_threads is None:
            if self.device.type == 'cuda':
                vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
                m_threads = 1 if vram_gb < 8 else 2
            else:
                m_threads = 1

        wave, sr = librosa.load(input_path, mono=False, sr=44100)
        peak = max(np.max(wave), abs(np.min(wave)))
        wave /= peak

        if denoise:
            processed_neg = self.mdx_sess.process_wave(-wave, m_threads)
            processed_pos = self.mdx_sess.process_wave(wave, m_threads)
            wave_processed = (processed_pos - processed_neg) * 0.5
        else:
            wave_processed = self.mdx_sess.process_wave(wave, m_threads)

        wave_processed *= peak

        stem_name = self.mdx_model.stem_name if suffix is None else suffix
        main_filepath = None
        if not exclude_main:
            os.makedirs(output_dir, exist_ok=True)
            main_filepath = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_{stem_name}.wav")
            sf.write(main_filepath, wave_processed.T, sr)

        invert_filepath = None
        if not exclude_inversion:
            diff_stem_name = self.stem_naming.get(stem_name) if invert_suffix is None else invert_suffix
            diff_stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
            invert_filepath = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_{diff_stem_name}.wav")
            sf.write(invert_filepath, (-wave_processed.T * self.mdx_model.compensation) + wave.T, sr)

        if not keep_orig:
            os.remove(input_path)

        del wave_processed, wave
        gc.collect()
        return main_filepath, invert_filepath

