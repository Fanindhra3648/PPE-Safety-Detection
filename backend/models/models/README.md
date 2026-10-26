# Model weights

Place the trained Ultralytics YOLO weights here as `best.pt`.

Expected path: `models/best.pt`

This starter project does not include model weights. The API starts without weights,
but `/api/health` reports `model_missing` and inference endpoints return HTTP 503 until
a trained model is added. You can set `PPE_MODEL_PATH` to an alternate location.
