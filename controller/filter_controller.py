from fastapi import APIRouter, UploadFile, File
from service.filter_service import CrochetDetector
from PIL import Image
import io

router = APIRouter(prefix="/filterService", tags=["filter"])
crochet = CrochetDetector()


@router.post("/processImage")
async def is_a_crochet_image(image: UploadFile = File(...)):
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')
    image = image.resize((224, 224))
    return crochet.is_a_crochet(image)
