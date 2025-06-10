FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

WORKDIR /tyee

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["echo", "Tyee environment with PyTorch 2.5.0 and cuDNN 9 is ready."]