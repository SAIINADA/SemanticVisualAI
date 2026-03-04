cd ~/SemanticVisualAI
source venv/bin/activate

python3.11 train_fake_model.py
python3.11 train_text_emotion.py
CUDA_VISIBLE_DEVICES=-1 python3.11 train_face_emotion.py

CUDA_VISIBLE_DEVICES=-1 python3.11 app.py
