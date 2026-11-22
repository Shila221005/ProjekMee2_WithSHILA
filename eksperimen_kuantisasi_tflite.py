import argparse
import json
import platform
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


FITUR_KLINIS = [
    "Age",
    "BMI",
    "HighBP",
    "HighChol",
    "PhysActivity",
    "Smoker",
    "Fruits",
    "Veggies",
]
TARGET = "Diabetes_012"
JUMLAH_BIN_BMI = 10
SEED = 42


def quantize_bmi(features, quantizer):
    transformed = features.copy()
    transformed["BMI"] = (
        quantizer.transform(features[["BMI"]]).ravel().astype(np.int32)
    )
    return transformed


def make_model(input_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_int8_model(model_path, inputs):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(
        input_details["index"], [len(inputs), inputs.shape[1]], strict=False
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    if input_scale <= 0:
        raise ValueError("Model INT8 tidak memiliki skala input yang valid.")
    model_input = np.clip(
        np.rint(inputs / input_scale + input_zero_point), -128, 127
    ).astype(np.int8)

    interpreter.set_tensor(input_details["index"], model_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    output_scale, output_zero_point = output_details["quantization"]
    if output_scale <= 0:
        raise ValueError("Model INT8 tidak memiliki skala output yang valid.")
    return (output.astype(np.float32) - output_zero_point) * output_scale


def metric_summary(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def benchmark_latency(
    model_path,
    inputs,
    iterations=100,
    warmup_iterations=20,
    inferences_per_measurement=50,
):
    if len(inputs) < iterations:
        raise ValueError("Jumlah data uji harus cukup untuk seluruh iterasi latensi.")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(
        input_details["index"], [1, inputs.shape[1]], strict=False
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    if input_scale <= 0:
        raise ValueError("Model INT8 tidak memiliki skala input yang valid.")
    quantized_inputs = np.clip(
        np.rint(inputs[:iterations] / input_scale + input_zero_point),
        -128,
        127,
    ).astype(np.int8)

    def invoke(sample):
        interpreter.set_tensor(input_details["index"], sample.reshape(1, -1))
        interpreter.invoke()
        interpreter.get_tensor(output_details["index"])

    for index in range(warmup_iterations):
        invoke(quantized_inputs[index % len(quantized_inputs)])

    latencies_ms = []
    for index in range(iterations):
        sample = quantized_inputs[index]
        start = time.perf_counter_ns()
        for _ in range(inferences_per_measurement):
            invoke(sample)
        latencies_ms.append(
            (time.perf_counter_ns() - start)
            / 1_000_000
            / inferences_per_measurement
        )

    return latencies_ms


def main():
    parser = argparse.ArgumentParser(
        description="Latih dan evaluasi model neural network TFLite INT8 saja."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name(
            "diabetes_012_health_indicators_BRFSS2015.csv"
        ),
        help="Path ke dataset BRFSS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("hasil_kuantisasi_nn"),
        help="Folder keluaran model, tabel, metrik, dan grafik.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--latency-iterations", type=int, default=100)
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError("Dataset tidak ditemukan: {}".format(args.data))
    if args.epochs < 1:
        raise ValueError("--epochs harus lebih besar dari 0.")
    if args.latency_iterations < 2:
        raise ValueError("--latency-iterations minimal 2.")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    df = pd.read_csv(args.data)
    X = df[FITUR_KLINIS]
    y = df[TARGET].astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=SEED,
    )

    bmi_quantizer = KBinsDiscretizer(
        n_bins=JUMLAH_BIN_BMI,
        encode="ordinal",
        strategy="quantile",
    )
    bmi_quantizer.fit(X_fit[["BMI"]])

    X_fit = quantize_bmi(X_fit, bmi_quantizer)
    X_validation = quantize_bmi(X_validation, bmi_quantizer)
    X_test = quantize_bmi(X_test, bmi_quantizer)

    X_resampled, y_resampled = SMOTE(random_state=SEED).fit_resample(X_fit, y_fit)
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled).astype(np.float32)
    X_validation_scaled = scaler.transform(X_validation).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    model = make_model(X_resampled_scaled.shape[1])
    model.fit(
        X_resampled_scaled,
        y_resampled,
        validation_data=(X_validation_scaled, y_validation),
        epochs=args.epochs,
        batch_size=2048,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    int8_path = args.output_dir / "model_int8.tflite"

    calibration_data = scaler.transform(X_fit).astype(np.float32)
    calibration_data = calibration_data[: min(500, len(calibration_data))]

    def representative_dataset():
        for sample in calibration_data:
            yield [sample.reshape(1, -1)]

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = representative_dataset
    int8_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    int8_path.write_bytes(int8_converter.convert())

    int8_outputs = run_int8_model(int8_path, X_test_scaled)
    int8_predictions = np.argmax(int8_outputs, axis=1)
    int8_metrics = metric_summary(y_test, int8_predictions)
    latency_samples = benchmark_latency(
        int8_path,
        X_test_scaled,
        iterations=args.latency_iterations,
    )
    latency_summary = {
        "iterations": len(latency_samples),
        "mean_ms": float(np.mean(latency_samples)),
        "std_dev_ms": float(np.std(latency_samples, ddof=1)),
        "min_ms": float(np.min(latency_samples)),
        "max_ms": float(np.max(latency_samples)),
    }

    pd.DataFrame([{"model": "INT8", **latency_summary}]).to_csv(
        args.output_dir / "ringkasan_latensi_inferensi.csv", index=False
    )
    pd.DataFrame(
        [
            {"iteration": index + 1, "latency_ms": value}
            for index, value in enumerate(latency_samples)
        ]
    ).to_csv(args.output_dir / "data_latensi_inferensi.csv", index=False)

    int8_size = int8_path.stat().st_size
    pd.DataFrame(
        [
            {
                "model": "TFLite INT8",
                "ekstensi": ".tflite",
                "ukuran_byte": int8_size,
                "ukuran_kb": int8_size / 1024,
                "ukuran_mb": int8_size / 1_000_000,
            }
        ]
    ).to_csv(args.output_dir / "ukuran_model_int8.csv", index=False)

    metric_labels = [
        ("Accuracy", "accuracy"),
        ("Precision Macro", "precision_macro"),
        ("Recall Macro", "recall_macro"),
        ("F1-Score Macro", "f1_macro"),
    ]
    performance_rows = [
        {"metric": label, "int8_percent": int8_metrics[key] * 100}
        for label, key in metric_labels
    ]
    pd.DataFrame(performance_rows).to_csv(
        args.output_dir / "metrik_klasifikasi_int8.csv", index=False
    )

    metrics = {
        "dataset": str(args.data),
        "features": FITUR_KLINIS,
        "target": TARGET,
        "split": "80% train, 20% test; stratified; random_state=42",
        "training": "SMOTE hanya pada subset train; BMI 10 bin quantile; StandardScaler",
        "architecture": (
            "Dense(128, ReLU) -> Dense(64, ReLU) -> "
            "Dense(32, ReLU) -> Dense(3, softmax)"
        ),
        "quantization": "Post-training full-integer quantization; INT8 input/output",
        "int8": int8_metrics,
        "int8_model_size_bytes": int8_size,
        "latency_measurement": {
            "device": platform.processor() or platform.machine(),
            "system": platform.platform(),
            "runtime": "TensorFlow Lite CPU",
            "batch_size": 1,
            "warmup_iterations": 20,
            "measured_iterations": args.latency_iterations,
            "inferences_per_measurement": 50,
            "timing_scope": (
                "set_tensor, invoke, dan get_tensor; preprocessing di luar pengukuran"
            ),
        },
        "latency_summary": latency_summary,
        "latency_samples_ms": latency_samples,
    }
    (args.output_dir / "hasil_evaluasi.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["TFLite INT8"],
        [int8_size / 1_000_000],
        color="#F58518",
    )
    plt.ylabel("Ukuran file model (MB)")
    plt.title("Ukuran Model TFLite INT8")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.bar_label(
        bars,
        labels=["{:.3f} MB".format(int8_size / 1_000_000)],
        padding=3,
    )
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "ukuran_model_int8.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    chart_x = np.arange(len(performance_rows))
    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        chart_x,
        [row["int8_percent"] for row in performance_rows],
        color="#F58518",
    )
    plt.xticks(chart_x, [row["metric"] for row in performance_rows])
    plt.ylabel("Nilai metrik (%)")
    plt.ylim(0, 100)
    plt.title("Metrik Klasifikasi Model MLP TFLite INT8")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "metrik_klasifikasi_int8.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [latency_samples],
        tick_labels=["INT8"],
        showmeans=True,
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "#777777",
            "markersize": 3,
            "alpha": 0.45,
        },
    )
    plt.ylabel("Latensi inferensi per sampel (ms)")
    plt.title(
        "Latensi Inferensi Model TFLite INT8\n"
        "{} pengukuran, rerata 50 inferensi, batch size 1, CPU".format(
            args.latency_iterations
        )
    )
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "latensi_inferensi_int8.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    obsolete_outputs = [
        "model_fp32.tflite",
        "perbandingan_ukuran_model.csv",
        "perbandingan_ukuran_model.png",
        "perbandingan_performa_klasifikasi.csv",
        "perbandingan_performa_klasifikasi.png",
        "perbandingan_latensi_inferensi.png",
    ]
    for filename in obsolete_outputs:
        obsolete_path = args.output_dir / filename
        if obsolete_path.exists():
            obsolete_path.unlink()

    print("Eksperimen INT8 selesai.")
    print("Folder hasil:", args.output_dir)
    print("Ukuran INT8: {:.3f} MB".format(int8_size / 1_000_000))
    print("Metrik INT8:", json.dumps(int8_metrics, ensure_ascii=False))
    print(
        "Latensi INT8: mean {:.4f} ms, std {:.4f} ms, min {:.4f} ms, "
        "max {:.4f} ms".format(
            latency_summary["mean_ms"],
            latency_summary["std_dev_ms"],
            latency_summary["min_ms"],
            latency_summary["max_ms"],
        )
    )


if __name__ == "__main__":
    main()
