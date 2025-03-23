from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from tempfile import NamedTemporaryFile
import shutil
from .middleware import APIKeyMiddleware
from sqlalchemy.orm import Session
from ..database.database import get_db, init_db
from ..database.models import ShoeRequest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.predict_damage import predict_damage
from .models import ShoeAnalysisRequest, ShoeAnalysisResponse

app = FastAPI(title="Shoe Damage Analysis API")

# Add middlewares
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DAMAGE_CONFIDENCE_THRESHOLD = 60.0

@app.on_event("startup")
async def startup_event():
    init_db()

@app.post("/api/getprice", response_model=ShoeAnalysisResponse)
async def analyze_shoe(
    image: UploadFile = File(...), 
    description: str = Form(...),
    suggested_price: float = Form(None),
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db)
):
    # API key validation is handled by the middleware
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File uploaded is not an image")
            
        # Save image data
        image_data = await image.read()
        
        # Create temporary file for processing
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        # Process the image and get prediction
        result = predict_damage(temp_path, verbose=False, display_image=False)
        
        # Check if damage confidence is above threshold
        if result["confidence"] < DAMAGE_CONFIDENCE_THRESHOLD:
            # Create response for shoes with no significant damage
            response = ShoeAnalysisResponse(
                damage_type="No significant damage",
                material_type=result["material_type"],
                shoe_type=result["shoe_type"],
                confidence=result["confidence"],
                description="The shoe appears to be in good condition",
                repair_cost_min=0.0,
                repair_cost_max=0.0,
                analysis_details=f"Based on your description: '{description}' and the uploaded image, "
                               f"we did not detect any significant damage. The shoe appears to be in good condition."
            )
        else:
            # Save request to database only if damage is detected
            db_request = ShoeRequest(
                description=description,
                image_data=image_data,
                image_path=image.filename,
                suggested_price=suggested_price,
                damage_type=result["damage_type"],
                material_type=result["material_type"],
                confidence=result["confidence"],
                repair_cost_min=result["repair_cost_min"],
                repair_cost_max=result["repair_cost_max"]
            )
            db.add(db_request)
            db.commit()
            
            # Create response for damaged shoes
            response = ShoeAnalysisResponse(
                damage_type=result["damage_type"],
                material_type=result["material_type"],
                shoe_type=result["shoe_type"],
                confidence=result["confidence"],
                description=result["description"],
                repair_cost_min=result["repair_cost_min"],
                repair_cost_max=result["repair_cost_max"],
                analysis_details=f"Based on your description: '{description}' and the uploaded image, "
                               f"we detected {result['damage_type']} damage with {result['confidence']:.1f}% confidence."
            )

        # Clean up temporary file
        os.unlink(temp_path)
        
        return response
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
