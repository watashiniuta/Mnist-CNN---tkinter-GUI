# 🧠 MNIST Handwritten Digit Classification (PyTorch)

PyTorch 기반 CNN 모델로 **MNIST 손글씨 숫자 분류**를 학습하고,  
**Tkinter GUI를 통해 직접 숫자를 그려 예측 결과를 확인**할 수 있는 프로젝트입니다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/299fea4a-377e-4f0d-824c-9cc0a145281e" width="220" height="240">
</p>

해당 프로젝트는 다음을 목표 선정으로 두었습니다:

- PyTorch CNN 모델 구현
- 학습 / 검증 파이프라인 분리
- 학습된 모델을 GUI 추론에 활용

---

## 📁 Project Structure

```text
mnist-cnn/
 ├─ model.py          # CNN 모델 정의
 ├─ dataset.py        # MNIST DataLoader & transforms
 ├─ engine.py         # train / evaluation loop
 ├─ utils.py          # device, seed 설정
 ├─ train.py          # argparse 기반 인자 추출 및 학습
 ├─ predict_gui.py    # Tkinter 숫자 그리기 + 예측
 ├─ mnist_cnn.pth     # 학습된 모델 가중치
 └─ README.md
```

---

## How to running

1. Clone the repository
```bash
git clone https://github.com/watashiniuta/Mnist-CNN---tkinter-GUI.git
cd Mnist-CNN---tkinter-GUI
```
2. install Dependencies
```bash
pip install torch torchvision pillow numpy
```
3. Install the Mnist dataset provided by Pythorch into the working directory
```bash
python dataset.py
```
4. training model
```bash
python train.py -batch_size 64 --epochs 500 --augment --cuda ...
```
5. running
```bash
python predict_gui.py
```