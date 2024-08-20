import torch
import numpy as np
import torchaudio
import argparse
from model_quant_full import ECAPA_TDNN
from pre_emphasis import PreEmphasis

def parse_arguments():
    parser = argparse.ArgumentParser(description="ECAPA_trainer")
    parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')
    parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
    return parser.parse_args()

def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        print('No GPU')
        return torch.device('cpu')

def load_model(args, device):
    model = ECAPA_TDNN(C=args.C).to(device)
    print(f"Model {args.initial_model} loaded from previous state!")
    loaded_state = torch.load(args.initial_model)
    model.load_state_dict(loaded_state, strict=False)
    return model

def create_mel_spectrogram():
    return torch.nn.Sequential(
        PreEmphasis(),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80
        )
    )

def preprocess_input(torchfbank, B, L, PRECISION, device):
    x = torch.zeros((B, L), dtype=PRECISION)
    with torch.no_grad():
        x = torchfbank(x) + 1e-6
        x = x.log()
        x = x - torch.mean(x, dim=-1, keepdim=True)
    return x.to(device)

def export_to_onnx(model, x, onnx_file_path):
    #output shae : [batch, channel, length]
    dynamic_axes = {
        #[practice] put the dynamic axes
        'input': {}
    }
    torch.onnx.export(model,
                      x,
                      onnx_file_path,
                      export_params=True,
                      opset_version=13,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=dynamic_axes)
    print(f"Model saved in ONNX format: {onnx_file_path}")

def main():
    args = parse_arguments()
    device = setup_device()
    
    model = load_model(args, device)
    torchfbank = create_mel_spectrogram()
    
    #[practice] put the batch size and random audio length for inference and remove annotation on line 67 
    # B, L = 
    PRECISION = torch.float32
    x = preprocess_input(torchfbank, B, L, PRECISION, device)
    
    onnx_file_path = "models/quantized_model.onnx"
    export_to_onnx(model, x, onnx_file_path)

if __name__ == "__main__":
    main()