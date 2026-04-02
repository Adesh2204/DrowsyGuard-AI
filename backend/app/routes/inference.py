import json
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.services.inference_service import InferenceService
from app.utils.preprocessing import decode_base64_image

router = APIRouter()


class PredictImageRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image payload.")
    session_id: str | None = Field(default=None, description="Optional session key for temporal buffering.")


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.inference_service


@router.post("/predict-image")
async def predict_image(
    payload: PredictImageRequest,
    inference_service: InferenceService = Depends(get_inference_service),
) -> dict:
    try:
        frame_bgr = decode_base64_image(payload.image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id = payload.session_id or "rest-default"
    return inference_service.predict(frame_bgr, session_id=session_id)


@router.websocket("/live-detection")
async def live_detection(websocket: WebSocket) -> None:
    await websocket.accept()

    inference_service: InferenceService = websocket.app.state.inference_service
    session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())

    await websocket.send_json({"type": "session", "session_id": session_id})

    try:
        while True:
            raw_message = await websocket.receive()

            if raw_message.get("bytes") is not None:
                np_buffer = np.frombuffer(raw_message["bytes"], dtype=np.uint8)
                frame_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    await websocket.send_json({"type": "error", "detail": "Invalid image bytes."})
                    continue
            else:
                text_payload = raw_message.get("text")
                if text_payload is None:
                    continue
                try:
                    payload = json.loads(text_payload)
                    image_payload = payload.get("image")
                except json.JSONDecodeError:
                    image_payload = text_payload

                if not image_payload:
                    await websocket.send_json({"type": "error", "detail": "Missing image payload."})
                    continue

                try:
                    frame_bgr = decode_base64_image(image_payload)
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue

            prediction = inference_service.predict(frame_bgr, session_id=session_id)
            await websocket.send_json(prediction)

    except WebSocketDisconnect:
        inference_service.close_session(session_id)
