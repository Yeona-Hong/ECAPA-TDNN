import torch
# import torch.onnx
import numpy as np
from model_quant_full import ECAPA_TDNN
from pre_emphasis import PreEmphasis
import argparse, torch, warnings, torchaudio

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
parser.add_argument('--initial_model',  type=str,   default="", help='Path of the initial_model')
parser.add_argument('--C',       type=int,   default=1024, help='Channel size for the speaker encoder')

## Initialization
# warnings.simplefilter("ignore")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else print('No GPU'))
s = ECAPA_TDNN(C=args.C).to(device)
############ 모델 로드
print("Model %s loaded from previous state!"%args.initial_model)
self_state = s.state_dict()
loaded_state = torch.load(args.initial_model)
s.load_state_dict(loaded_state, strict=False)

B, L = 1, 13554
PRECISION = torch.float32
torchfbank = torch.nn.Sequential(
        PreEmphasis(),
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
    )
x = torch.zeros((B, L), dtype=PRECISION)#.to(device)
with torch.no_grad():
    x = torchfbank(x) + 1e-6
    x = x.log()
    x = x - torch.mean(x, dim=-1, keepdim=True)
x = x.to(device)
# 동적 축 설정
dynamic_axes = {
    'input': {0:'B', 2:'L'}  # 배치 사이즈(B), 채널(C)-Height, 길이(L)-Width
}

# ONNX 파일로 변환
onnx_file_path = "models/quantized_model.onnx"
torch.onnx.export(s,
                  x, 
                  onnx_file_path, 
                  export_params=True, 
                  opset_version=13, 
                  input_names=['input'], 
                  output_names=['output'],
                  dynamic_axes=dynamic_axes)

print(f"모델이 ONNX 포맷으로 저장되었습니다: {onnx_file_path}")


# trtexec --onnx=20240514_quantized_model_direct_8_se_res_1_new_new.onnx --saveEngine=20240514_quantized_model_direct_8_se_res_1_new_new.engine --explicitBatch --minShapes=input:1x80x75 --optShapes=input:1x80x400 --maxShapes=input:5x80x10000 --int8
# trtexec --onnx=temp.onnx --saveEngine=temp.engine --explicitBatch --minShapes=input:1x80x75 --optShapes=input:1x80x150 --maxShapes=input:1x80x300 --int8