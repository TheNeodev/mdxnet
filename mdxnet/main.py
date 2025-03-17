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

warnings.filterwarnings("ignore")
# Global stem naming dictionary used in inversion logic.
stem_naming = {
    'Vocals': 'Instrumental', 
    'Other': 'Instruments', 
    'Instrumental': 'Vocals', 
    'Drums': 'Drumless', 
    'Bass': 'Bassless'
}


class MDXModel:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.000):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        out_c = self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window,
                       center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 4, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])


class MDX:
    DEFAULT_SR = 44100
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR
    DEFAULT_PROCESSOR = 0

    def __init__(self, model_path: str, model: MDXModel, processor=DEFAULT_PROCESSOR):
        self.device = torch.device(f'cuda:{processor}') if processor >= 0 else torch.device('cpu')
        self.provider = ['CUDAExecutionProvider'] if processor >= 0 else ['CPUExecutionProvider']
        self.model = model

        # Load ONNX model using ONNX Runtime
        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        # Preload the model for faster performance
        self.ort.run(None, {'input': torch.rand(1, 4, model.dim_f, model.dim_t).numpy()})
        self.process = lambda spec: self.ort.run(None, {'input': spec.cpu().numpy()})[0]
        self.prog = None

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, 'rb') as f:
                f.seek(-10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()
        return model_hash

    @staticmethod
    def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE):
        if combine:
            processed_wave = None
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:
                    processed_wave = segment[:, start:end]
                else:
                    processed_wave = np.concatenate((processed_wave, segment[:, start:end]), axis=-1)
        else:
            processed_wave = []
            sample_count = wave.shape[-1]
            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count
            if margin_size > chunk_size:
                margin_size = chunk_size
            for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin
                cut = wave[:, start:end].copy()
                processed_wave.append(cut)
                if end == sample_count:
                    break
        return processed_wave

    def pad_wave(self, wave):
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size
        wave_p = np.concatenate((np.zeros((2, trim)), wave, np.zeros((2, pad)), np.zeros((2, trim))), axis=1)
        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)
        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)
        return mix_waves, pad, trim

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
        return processed_signal

    def process_wave(self, wave: np.array, mt_threads=1):
        self.prog = tqdm(total=0)
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, combine=False, chunk_size=chunk)
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
        processed_batches = []
        while not q.empty():
            processed_batches.append(q.get())
        processed_batches = [list(wave.values())[0] for wave in
                             sorted(processed_batches, key=lambda d: list(d.keys())[0])]
        assert len(processed_batches) == len(waves), 'Incomplete processed batches, please reduce batch size!'
        return self.segment(processed_batches, combine=True, chunk_size=chunk)


class MDXRunner:
    """
    This class encapsulates the functionality of running the MDX process.
    It sets up the device, loads the model parameters, processes the input
    audio file, and writes the output files.
    """
    def __init__(self, model_params, output_dir, model_path, filename,
                 exclude_main=False, exclude_inversion=False, suffix=None,
                 invert_suffix=None, denoise=False, keep_orig=True, m_threads=2):
        self.model_params = model_params
        self.output_dir = output_dir
        self.model_path = model_path
        self.filename = filename
        self.exclude_main = exclude_main
        self.exclude_inversion = exclude_inversion
        self.suffix = suffix
        self.invert_suffix = invert_suffix
        self.denoise = denoise
        self.keep_orig = keep_orig
        self.m_threads = m_threads

        # Setup the device; use CUDA if available.
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
            device_properties = torch.cuda.get_device_properties(self.device)
            vram_gb = device_properties.total_memory / 1024**3
            # If VRAM is low, force single-threaded processing.
            self.m_threads = 1 if vram_gb < 8 else self.m_threads

    def run(self):
        # Calculate model hash and obtain parameters.
        model_hash = MDX.get_hash(self.model_path)
        mp = self.model_params.get(model_hash)
        if mp is None:
            raise ValueError(f"Model hash {model_hash} not found in provided model parameters.")

        # Instantiate the MDX model.
        model = MDXModel(
            device=self.device,
            dim_f=mp["mdx_dim_f_set"],
            dim_t=2 ** mp["mdx_dim_t_set"],
            n_fft=mp["mdx_n_fft_scale_set"],
            stem_name=mp["primary_stem"],
            compensation=mp["compensate"]
        )
        mdx_sess = MDX(self.model_path, model)
        # Load and normalize the input wave.
        wave, sr = librosa.load(self.filename, mono=False, sr=44100)
        peak = max(np.max(wave), abs(np.min(wave)))
        wave /= peak

        # Process wave with optional denoising.
        if self.denoise:
            wave_processed = -(mdx_sess.process_wave(-wave, self.m_threads)) + \
                             (mdx_sess.process_wave(wave, self.m_threads))
            wave_processed *= 0.5
        else:
            wave_processed = mdx_sess.process_wave(wave, self.m_threads)
        # Restore original peak.
        wave_processed *= peak

        # Determine stem name for output files.
        stem_name = model.stem_name if self.suffix is None else self.suffix
        main_filepath = None
        if not self.exclude_main:
            main_filepath = os.path.join(
                self.output_dir,
                f"{os.path.basename(os.path.splitext(self.filename)[0])}_{stem_name}.wav"
            )
            sf.write(main_filepath, wave_processed.T, sr)

        invert_filepath = None
        if not self.exclude_inversion:
            diff_stem_name = stem_naming.get(stem_name) if self.invert_suffix is None else self.invert_suffix
            stem_name_inv = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
            invert_filepath = os.path.join(
                self.output_dir,
                f"{os.path.basename(os.path.splitext(self.filename)[0])}_{stem_name_inv}.wav"
            )
            sf.write(invert_filepath, (-wave_processed.T * model.compensation) + wave.T, sr)

        if not self.keep_orig:
            os.remove(self.filename)

        del mdx_sess, wave_processed, wave
        gc.collect()
        return main_filepath, invert_filepath


