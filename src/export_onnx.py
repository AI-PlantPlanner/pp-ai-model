"""7단계: v2 모델(joblib)을 ONNX로 변환 — 게임 등 non-Python 런타임 배포용.

joblib 크기가 아니라 ONNX 변환 후 크기가 배포 가능 여부의 실제 기준이라 변환 후 크기를
출력한다. sklearn 원본과 ONNX(onnxruntime) 예측값이 실질적으로 같은지도 함께 검증한다.
"""

import os

import numpy as np
import pandas as pd
from joblib import load
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

from src import config
from src.train_model import build_features_v2

ONNX_OUTPUT_PATH = config.MODEL_OUTPUT_PATH_V2.with_suffix(".onnx")


def convert(model_path=config.MODEL_OUTPUT_PATH_V2, onnx_path=ONNX_OUTPUT_PATH) -> tuple:
    bundle = load(model_path)
    model, feature_columns = bundle["model"], bundle["feature_columns"]

    initial_type = [("input", FloatTensorType([None, len(feature_columns)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=17)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return onnx_path, model, feature_columns


def verify(onnx_path, model, feature_columns, n_samples=2000) -> dict:
    """sklearn 원본 모델과 ONNX(onnxruntime) 예측값이 실질적으로 같은지 확인."""
    df = pd.read_csv(config.SYNTHETIC_OUTPUT_PATH).sample(n=n_samples, random_state=42)
    X = build_features_v2(df).reindex(columns=feature_columns, fill_value=0)
    X_np = X.to_numpy(dtype=np.float32)

    sk_pred = model.predict(X)  # DataFrame 그대로 넘겨야 학습 때와 동일하게 컬럼명 기반으로 예측한다.

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    onnx_pred = sess.run(None, {input_name: X_np})[0].ravel()

    diff = np.abs(sk_pred - onnx_pred)
    return {"max_diff": float(diff.max()), "mean_diff": float(diff.mean()), "n": n_samples}


def main() -> None:
    onnx_path, model, feature_columns = convert()
    size_mb = os.path.getsize(onnx_path) / 1e6
    joblib_size_mb = os.path.getsize(config.MODEL_OUTPUT_PATH_V2) / 1e6
    result = verify(onnx_path, model, feature_columns)

    print(f"ONNX 변환 완료 -> {onnx_path}")
    print(f"joblib 크기: {joblib_size_mb:.2f}MB -> ONNX 크기: {size_mb:.2f}MB")
    print(
        f"sklearn vs ONNX 예측값 차이 (n={result['n']}): "
        f"최대={result['max_diff']:.6f}, 평균={result['mean_diff']:.6f}"
    )


if __name__ == "__main__":
    main()
