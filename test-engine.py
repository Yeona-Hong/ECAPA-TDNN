import tensorrt as trt
import os
import numpy as np
import pycuda.driver as cuda
from pre_emphasis import PreEmphasis
import torchaudio, tqdm, soundfile
import pdb
import pycuda.autoinit
import torch
from tools import *

TRT_LOGGER = trt.Logger()


def load_engine(engine_filepath, trt_logger):
    with open(engine_filepath, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

torchfbank = torch.nn.Sequential(
        PreEmphasis(),
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
    )
def mel_change(data):
    data = torchfbank(data) + 1e-6
    data = data.log()
    data = data - torch.mean(data, dim=-1, keepdim=True)
    return data

def setup_context(engine):
    return engine.create_execution_context()

def load_dataset(eval_list):
    files = []
    embeddings = {}
    lines = open(eval_list).read().splitlines()
    for line in lines:
        files.append(line.split()[1])
        files.append(line.split()[2])
    setfiles = list(set(files))
    setfiles.sort()
    return setfiles, lines

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
def allocate_buffers(engine, context):
    """Allocates host and device buffer for TRT engine inference.

    This function is similair to the one in ../../common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.

    Args:
        engine (trt.ICudaEngine): TensorRT engine

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in range(engine.num_bindings):
        size = trt.volume(context.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

if __name__ == "__main__":
    engine_name = "./models/quantized_model.engine"
    engine = load_engine(engine_name, TRT_LOGGER)
    context = setup_context(engine)
    output_shapes = [(1, 192)]
    output2_shapes = [(5, 192)]
    
    files, lines = load_dataset('/data/yeonahong/vox/test/veri_test.txt')
    
    embeddings = {}
    for idx, file in tqdm.tqdm(enumerate(files), total = len(files)):
        audio, _  = soundfile.read(os.path.join("/data/yeonahong/vox/test/wav", file))
        
        # Full utterance
        data_1 = torch.FloatTensor(np.stack([audio],axis=0))#.to(device)
        # Spliited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        feats = np.stack(feats, axis = 0).astype(float)
        data_2 = torch.FloatTensor(feats)
        data_1 = mel_change(data_1)
        data_1 = np.array(data_1, dtype=np.float32, order='C')
        
        context.set_binding_shape(0, data_1.shape)
        inputs, outputs, bindings, stream = allocate_buffers(engine, context)
        np.copyto(inputs[0].host, data_1.ravel())
        embedding_1 = do_inference_v2(
        context, bindings=bindings, inputs=inputs,
        outputs=outputs, stream=stream)
        embedding_1 = [output.reshape(shape) for output, shape in zip(embedding_1, output_shapes)]
        
        data_2 = mel_change(data_2)
        data_2 = np.array(data_2, dtype=np.float32, order='C')
        #0이 결국 input
        context.set_binding_shape(0, data_2.shape)
        inputs2, outputs2, bindings2, stream2 = allocate_buffers(engine, context)
        np.copyto(inputs2[0].host, data_2.ravel())
        embedding_2 = do_inference_v2(
        context, bindings=bindings2, inputs=inputs2,
        outputs=outputs2, stream=stream2)
        embedding_2 = [output2.reshape(shape2) for output2, shape2 in zip(embedding_2, output2_shapes)]
        
        embeddings[file] = [embedding_1, embedding_2]
    scores, labels  = [], []
        
    for line in lines:			
        embedding_11, embedding_12 = embeddings[line.split()[1]]
        embedding_21, embedding_22 = embeddings[line.split()[2]]
        score_1 = np.mean(np.dot(embedding_11[0], embedding_21[0].T))  # higher is positive
        score_2 = np.mean(np.dot(embedding_12[0], embedding_22[0].T))
        score = (score_1 + score_2) / 2
        scores.append(score)
        labels.append(int(line.split()[0]))
    
    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    print(engine_name)
    print("EER %2.4f%%, minDCF %.4f%%"%(EER, minDCF))
