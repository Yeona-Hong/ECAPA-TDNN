import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import torchaudio
import soundfile as sf
import tqdm

from pre_emphasis import PreEmphasis
from tools import *

TRT_LOGGER = trt.Logger()

# Define the TorchMelSpectrogram transform as a function
def create_mel_spectrogram_transform():
    return torch.nn.Sequential(
        PreEmphasis(),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            window_fn=torch.hamming_window,
            n_mels=80
        ),
    )


def mel_change(data, torchfbank):
    data = torchfbank(data) + 1e-6
    data = data.log()
    data -= torch.mean(data, dim=-1, keepdim=True)
    return data

def load_engine(engine_filepath):
    with open(engine_filepath, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def setup_context(engine):
    return engine.create_execution_context()

def load_dataset(eval_list_path):
    with open(eval_list_path, 'r') as f:
        lines = f.read().splitlines()

    files = [line.split()[1] for line in lines] + [line.split()[2] for line in lines]
    unique_files = sorted(set(files))

    return unique_files, lines

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}"

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine, context):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in range(engine.num_bindings):
        size = trt.volume(context.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        buffer = HostDeviceMem(host_mem, device_mem)
        (inputs if engine.binding_is_input(binding) else outputs).append(buffer)

    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    
    stream.synchronize()
    return [out.host for out in outputs]

def process_audio_file(file_path, max_audio):
    audio, _ = sf.read(file_path)

    if audio.shape[0] <= max_audio:
        shortage = max_audio - audio.shape[0]
        audio = np.pad(audio, (0, shortage), 'wrap')

    return audio

def compute_embeddings(context, engine, audio_data, output_shapes):
    context.set_binding_shape(0, audio_data.shape)
    inputs, outputs, bindings, stream = allocate_buffers(engine, context)
    np.copyto(inputs[0].host, audio_data.ravel())

    output_data = do_inference_v2(context, bindings, inputs, outputs, stream)
    return [output.reshape(shape) for output, shape in zip(output_data, output_shapes)]

def main():
    engine_name = "./models/quantized_model.engine"
    engine = load_engine(engine_name)
    context = setup_context(engine)
    
    files, lines = load_dataset('/data/yeonahong/vox/test/veri_test.txt')
    max_audio = 300 * 160 + 240

    output_shapes = [(1, 192)]
    output2_shapes = [(5, 192)]

    embeddings = {}

    torchfbank = create_mel_spectrogram_transform()

    for file in tqdm.tqdm(files, total=len(files)):
        file_path = os.path.join("/data/yeonahong/vox/test/wav", file)
        audio = process_audio_file(file_path, max_audio)

        # Full utterance
        data_1 = torch.FloatTensor(np.stack([audio], axis=0))
        data_1 = mel_change(data_1,torchfbank).numpy().astype(np.float32, order='C')

        embedding_1 = compute_embeddings(context, engine, data_1, output_shapes)

        # Split utterance matrix
        feats = [audio[int(start):int(start) + max_audio] for start in np.linspace(0, audio.shape[0] - max_audio, num=5)]
        data_2 = torch.FloatTensor(np.stack(feats, axis=0))
        data_2 = mel_change(data_2,torchfbank).numpy().astype(np.float32, order='C')

        embedding_2 = compute_embeddings(context, engine, data_2, output2_shapes)

        embeddings[file] = [embedding_1, embedding_2]

    scores, labels = [], []

    for line in lines:
        embedding_11, embedding_12 = embeddings[line.split()[1]]
        embedding_21, embedding_22 = embeddings[line.split()[2]]

        score_1 = np.mean(np.dot(embedding_11[0], embedding_21[0].T))
        score_2 = np.mean(np.dot(embedding_12[0], embedding_22[0].T))
        score = (score_1 + score_2) / 2

        scores.append(score)
        labels.append(int(line.split()[0]))

    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    print(engine_name)
    print(f"EER: {EER:.4f}%, minDCF: {minDCF:.4f}%")

if __name__ == "__main__":
    main()
