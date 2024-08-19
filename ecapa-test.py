import argparse
import os
import warnings
import numpy as np
import torch
import torchaudio
import soundfile
from tqdm import tqdm
from tensorboardX import SummaryWriter
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from tools import init_args, tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from model_quant_full import ECAPA_TDNN
from pre_emphasis import PreEmphasis

def parse_arguments():
    parser = argparse.ArgumentParser(description="ECAPA_trainer")
    parser.add_argument('--eval_list',  type=str,   default="/data/yeonahong/vox/test/veri_test.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
    parser.add_argument('--eval_path',  type=str,   default="/data/yeonahong/vox/test/wav",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
    parser.add_argument('--save_path',  type=str,   default="exps",                                     help='Path to save the score.txt and models')
    parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')

    parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')

    args = parser.parse_args()
    return init_args(args)

def setup_environment():
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('No GPU available')
    return device

def create_mel_spectrogram():
    return torch.nn.Sequential(
        PreEmphasis(),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80
        )
    )

def mel_change(data, torchfbank):
    data = torchfbank(data) + 1e-6
    data = data.log()
    return data - torch.mean(data, dim=-1, keepdim=True)

def collect_stats(model, eval_list, eval_path, device, torchfbank):
	"""Feed data to the network and collect statistics for quantization calibration."""
	for name, module in model.named_modules():
		if isinstance(module, quant_nn.TensorQuantizer):
			if module._calibrator is not None:
				module.disable_quant()
				module.enable_calib()
			else:
				module.disable()

	files = []
	lines = open(eval_list).read().splitlines()
	for line in lines:
		files.append(line.split()[1])
		files.append(line.split()[2])
	setfiles = list(set(files))
	setfiles.sort()

	# Processing only the specified number of batches for stats collection
	processed_batches = 0
	for idx, file in tqdm(enumerate(setfiles), total = len(setfiles)):
		audio, _  = soundfile.read(os.path.join(eval_path, file))
		data_1 = torch.FloatTensor(np.stack([audio],axis=0))#.to(device)
		data_1 = mel_change(data_1,torchfbank)
		data_1 = data_1.to(device)
        # Speaker embeddings

		embedding_1 = model(data_1, aug = False)
		processed_batches += 1

    # Disable calibrators and re-enable quantization
	for name, module in model.named_modules():
		if isinstance(module, quant_nn.TensorQuantizer):
			if module._calibrator is not None:
				module.enable_quant()
				module.disable_calib()
			else:
				module.enable()

def compute_amax(model, **kwargs):
	# Load calib result
	for name, module in model.named_modules():
		if isinstance(module, quant_nn.TensorQuantizer):
			if module._calibrator is not None:
				if isinstance(module._calibrator, calib.MaxCalibrator):
					module.load_calib_amax()
				else:
					module.load_calib_amax(**kwargs)
			# print(F"{name:40}: {module}")
	model.cuda()    

def load_model(args, device):
    model = ECAPA_TDNN(C=args.C).to(device)
    loaded_state = torch.load(args.initial_model)
    model.load_state_dict(loaded_state, strict=False)
    print(model)
    print(f"Model {args.initial_model} loaded from previous state!")
    return model

def extract_embeddings(model, args, torchfbank, device):
    embeddings = {}
    files = set()
    with open(args.eval_list, 'r') as f:
        for line in f:
            parts = line.strip().split()
            files.update(parts[1:3])
    
    for file in tqdm(sorted(files), desc="Extracting embeddings"):
        audio, _ = soundfile.read(os.path.join(args.eval_path, file))
        embeddings[file] = process_audio(audio, model, torchfbank, device)
    
    return embeddings

def process_audio(audio, model, torchfbank, device):
    # Full utterance
    data_1 = torch.FloatTensor(np.stack([audio], axis=0))
    
    # Split utterance
    max_audio = 300 * 160 + 240
    if audio.shape[0] <= max_audio:
        audio = np.pad(audio, (0, max_audio - audio.shape[0]), 'wrap')
    
    startframes = np.linspace(0, audio.shape[0] - max_audio, num=5)
    feats = [audio[int(sf):int(sf) + max_audio] for sf in startframes]
    data_2 = torch.FloatTensor(np.stack(feats, axis=0))
    
    with torch.no_grad():
        data_1 = mel_change(data_1, torchfbank).to(device)
        data_2 = mel_change(data_2, torchfbank).to(device)
        
        embedding_1 = torch.nn.functional.normalize(model(data_1, aug=False), p=2, dim=1)
        embedding_2 = torch.nn.functional.normalize(model(data_2, aug=False), p=2, dim=1)
    
    return [embedding_1, embedding_2]

def compute_scores(embeddings, eval_list):
    scores, labels = [], []
    with open(eval_list, 'r') as f:
        for line in f:
            parts = line.strip().split()
            emb11, emb12 = embeddings[parts[1]]
            emb21, emb22 = embeddings[parts[2]]
            
            score_1 = torch.mean(torch.matmul(emb11, emb21.T))
            score_2 = torch.mean(torch.matmul(emb12, emb22.T))
            score = ((score_1 + score_2) / 2).cpu().numpy()
            
            scores.append(score)
            labels.append(int(parts[0]))
    
    return scores, labels

def main():
    args = parse_arguments()
    device = setup_environment()
    writer = SummaryWriter('./runs/sv-64-seed-1234-ECAPA')
    
    torchfbank = create_mel_spectrogram()
    model = load_model(args, device)
    
    model.eval()
    with torch.no_grad():
        collect_stats(model, args.eval_list, args.eval_path, device, torchfbank)
        compute_amax(model, method="percentile", percentile=100)
    
    embeddings = extract_embeddings(model, args, torchfbank, device)
    scores, labels = compute_scores(embeddings, args.eval_list)
    
    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    
    print(f"EER {EER:.4f}%, minDCF {minDCF:.4f}%")
    
    # torch.save(model.state_dict(), './models/quantized_model_direct_8_1.pth')

if __name__ == "__main__":
    main()
