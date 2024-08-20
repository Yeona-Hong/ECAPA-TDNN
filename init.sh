cd ../
mv ECAPA-TDNN TensorRT/tools/pytorch-quantization/tests
cd TensorRT/tools/pytorch-quantization/

pip install -r requirements.txt
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python setup.py install
pip install soundfile
pip install scikit-learn
pip install torchaudio==2.2.1
pip install tqdm
pip install tensorboardX
pip install onnx
#pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

