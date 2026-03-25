#!/usr/bin/env python3
"""
Детекция событий за столиком в ресторане: ROI, YOLOv8 (люди), машина состояний, аналитика.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class TableState(Enum):
    EMPTY = auto()
    OCCUPIED = auto()


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Детекция занятости столика по видео (YOLOv8n, ROI)."
    )
    p.add_argument(
        "--video",
        type=str,
        required=True,
        help="Путь к входному видеофайлу.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Путь к весам YOLOv8 (по умолчанию: yolov8n.pt).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Путь к выходному видео (по умолчанию: output.mp4).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=2,
        metavar="N",
        help="Обрабатывать каждый N-й кадр детектором (по умолчанию: 2). "
        "Между ними используется последнее решение.",
    )
    p.add_argument(
        "--debounce",
        type=int,
        default=30,
        metavar="F",
        help="Порог гистерезиса: новое состояние подтверждается только после "
        "F последовательных кадров с одинаковым результатом (по умолчанию: 30).",
    )
    return p.parse_args()


def select_roi_first_frame(cap: cv2.VideoCapture, window: str = "ROI — выделите стол") -> ROI:
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Не удалось прочитать первый кадр видео.")
    clone = frame.copy()
    cv2.imshow(window, clone)
    cv2.waitKey(1)
    r = cv2.selectROI(window, clone, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    x, y, w, h = map(int, r)
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI не выбран: ширина или высота равны нулю.")
    return ROI(x=x, y=y, w=w, h=h)


def person_in_table_roi(
    bbox_xyxy: np.ndarray,
    roi: ROI,
) -> bool:
    """
    Центр bbox внутри ROI или не менее 50% площади bbox пересекается с ROI.
    """
    x1, y1, x2, y2 = map(float, bbox_xyxy.flatten()[:4])
    rx, ry, rw, rh = roi.x, roi.y, roi.w, roi.h
    rx2, ry2 = rx + rw, ry + rh

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if rx <= cx <= rx2 and ry <= cy <= ry2:
        return True

    ix1 = max(x1, rx)
    iy1 = max(y1, ry)
    ix2 = min(x2, rx2)
    iy2 = min(y2, ry2)
    if ix1 >= ix2 or iy1 >= iy2:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = (x2 - x1) * (y2 - y1)
    if area <= 1e-6:
        return False
    return (inter / area) >= 0.5


def detect_occupied(
    frame: np.ndarray,
    model: YOLO,
    roi: ROI,
) -> tuple[bool, list[tuple[int, int, int, int]]]:
    """Класс 0 — person (COCO)."""
    results = model.predict(
        frame,
        classes=[0],
        verbose=False,
        imgsz=640,
    )
    boxes_out: list[tuple[int, int, int, int]] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for b in r.boxes:
            xyxy = b.xyxy.cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy.flatten()[:4])
            boxes_out.append((x1, y1, x2, y2))
            if person_in_table_roi(xyxy, roi):
                return True, boxes_out
    return False, boxes_out


def format_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:05.2f}"
    return f"{m:d}:{s:05.2f}"


def main() -> None:
    args = parse_args()
    stride = max(1, args.stride)
    debounce_threshold = max(1, args.debounce)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Пауза на первом кадре: выделите зону столика мышью, затем ENTER или SPACE.")
    roi = select_roi_first_frame(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Загрузка модели YOLO...")
    model = YOLO(args.model)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Ошибка: не удалось создать выходное видео: {args.output}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    logs: list[dict] = []
    empty_to_approach_durations: list[float] = []

    state: Optional[TableState] = None
    time_empty_since: Optional[float] = None
    state_enter_time: float = 0.0

    frame_idx = 0
    current_occupied = False
    last_processed_occupied: Optional[bool] = None
    last_person_boxes: list[tuple[int, int, int, int]] = []

    debounce_counter: int = 0
    pending_state: Optional[TableState] = None

    window = "Table events"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1
        video_time = (frame_idx - 1) / fps if fps > 0 else 0.0

        process_this = (frame_idx % stride == 0) or (frame_idx == 1)
        if process_this:
            current_occupied, person_boxes = detect_occupied(frame, model, roi)
            last_processed_occupied = current_occupied
            last_person_boxes = person_boxes
        else:
            person_boxes = last_person_boxes
            if last_processed_occupied is not None:
                current_occupied = last_processed_occupied

        raw_state = TableState.OCCUPIED if current_occupied else TableState.EMPTY

        if state is None:
            state = raw_state
            state_enter_time = video_time
            pending_state = None
            debounce_counter = 0
            logs.append(
                {
                    "timestamp": video_time,
                    "event_type": "INIT",
                    "state": state.name,
                }
            )
        elif raw_state != state:
            if pending_state == raw_state:
                debounce_counter += 1
            else:
                pending_state = raw_state
                debounce_counter = 1

            if debounce_counter >= debounce_threshold:
                if state == TableState.OCCUPIED and raw_state == TableState.EMPTY:
                    time_empty_since = video_time
                    logs.append(
                        {
                            "timestamp": video_time,
                            "event_type": "STATE_EMPTY",
                            "state": TableState.EMPTY.name,
                        }
                    )
                elif state == TableState.EMPTY and raw_state == TableState.OCCUPIED:
                    logs.append(
                        {
                            "timestamp": video_time,
                            "event_type": "APPROACH",
                            "state": TableState.OCCUPIED.name,
                        }
                    )
                    if time_empty_since is not None:
                        empty_to_approach_durations.append(video_time - time_empty_since)
                    time_empty_since = None
                state = raw_state
                state_enter_time = video_time
                pending_state = None
                debounce_counter = 0
        else:
            pending_state = None
            debounce_counter = 0

        duration_in_state = video_time - state_enter_time if state else 0.0
        if state == TableState.EMPTY:
            roi_color = (0, 255, 0)
            label = "EMPTY (стол пуст)"
        else:
            roi_color = (0, 0, 255)
            label = "OCCUPIED (стол занят)"

        rx, ry, rw, rh = roi.rect
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), roi_color, 2)
        for (bx1, by1, bx2, by2) in person_boxes:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 1)

        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            roi_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"t в состоянии: {format_hms(duration_in_state)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"видео: {format_hms(video_time)}  кадр {frame_idx}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        out.write(frame)
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(logs)
    if not df.empty:
        print("\n--- Журнал событий (первые строки) ---")
        print(df.head(20).to_string(index=False))
        if len(df) > 20:
            print(f"... всего записей: {len(df)}")

    if empty_to_approach_durations:
        avg_delay = float(np.mean(empty_to_approach_durations))
        print(
            f"\nСреднее время от «стол пуст» (после занятости) до следующего подхода: "
            f"{avg_delay:.2f} с ({format_hms(avg_delay)})"
        )
        print(f"Количество интервалов: {len(empty_to_approach_durations)}")
    else:
        print(
            "\nНет полных интервалов OCCUPIED→EMPTY→APPROACH: "
            "средняя задержка «пусто → подход» не вычислена."
        )


if __name__ == "__main__":
    main()
