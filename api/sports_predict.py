import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from torchvision import transforms
from model import CheckSportsModel

sports_predict_router = APIRouter(prefix='/sports_predict', tags=['Sports Predict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckSportsModel()
model.load_state_dict(torch.load('sports_model.pth', map_location=device))
model.to(device)
model.eval()

classes = torch.load('sports_classes.pth')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def predict_image(image_bytes: bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail='Файл не является изображением')

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)[0]

    top_probs, top_ids = torch.topk(probs, 3)
    top3 = [
        {'class_id': class_id, 'label': classes[class_id], 'confidence': round(prob * 100, 2)}
        for class_id, prob in zip(top_ids.tolist(), top_probs.tolist())
    ]

    return {**top3[0], 'top3': top3}


@sports_predict_router.post('/')
async def sports_predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return predict_image(image_bytes)
