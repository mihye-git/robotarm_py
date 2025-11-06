# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
[YOLO + 카메라보정 기반 좌표변환 + 스레드 분리 + 감지 후 카메라 자동종료 + 좌표저장 v8.0]

📌 전체 순서
-------------------------------------------------
1️⃣ 카메라 스레드: 프레임 송출만 수행
2️⃣ 메인 루프: ROI 내 YOLO 감지 → 3초 유지 시
3️⃣ 좌표 계산(pixel_to_robot) + JSON 저장
4️⃣ 카메라 종료 → 로봇 이동 (Home→Pick→Place→Home)
"""

import threading
import cv2
import time
import argparse
import numpy as np
import json
import os
from ultralytics import YOLO

# ======================================================
# 0️⃣ 로봇 클래스 로드
# ======================================================
try:
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    from pymycobot.mycobot import MyCobot as CobotClass


# ======================================================
# 1️⃣ 포즈 정의
# ======================================================
POSES = {
    "Home":  [59.8, -215.9, 354.6, -175.33, 8.65, 86.68],
    "Clear": [264.0, -1.0, 379.0, -153, 11, -106],
    "Place": [333.0, 11.0, 170.0, -175, -0.08, -89.0],
}
DEFAULT_SPEED = 20


# ======================================================
# 2️⃣ 카메라 보정값 로드
# ======================================================
def load_camera_params(yaml_path="/home/vboxuser/robotarm/camera_info.yaml"):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"❌ '{yaml_path}' 파일을 열 수 없습니다.")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    print("📷 카메라 보정 파라미터 로드 완료")
    return camera_matrix, dist_coeffs


# ======================================================
# 3️⃣ 픽셀 → 로봇 좌표 변환 (오프셋 포함)
# ======================================================
def pixel_to_robot(cx, cy, distance_cm, camera_matrix, dist_coeffs):
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=None)
    norm_x, norm_y = undistorted[0, 0]

    # 깊이 계산 (cm → mm)
    scale_z = distance_cm * 10.0
    x_cam = norm_x * scale_z
    y_cam = norm_y * scale_z

    # ----------------------------------------
    # 📏 오프셋 (테스트 기준)
    # ----------------------------------------
    TCP_BASE_OFFSET_X = 59.8
    TCP_BASE_OFFSET_Y = -215.9
    TCP_BASE_OFFSET_Z = 354.6
    CAMERA_TO_TCP_OFFSET_X = 90.0   # ← 카메라가 X방향으로 90mm 앞에 있음
    CAMERA_TO_TCP_OFFSET_Y = 0.0
    CAMERA_TO_TCP_OFFSET_Z = 170.0  # ← 실제 높이 차이 (현재는 사용 안 함)

    # ----------------------------------------
    # 로봇 좌표 계산
    # ----------------------------------------
    robot_x = TCP_BASE_OFFSET_X + CAMERA_TO_TCP_OFFSET_X + y_cam
    robot_y = TCP_BASE_OFFSET_Y + CAMERA_TO_TCP_OFFSET_Y + x_cam

    # Z는 현재 테스트에서는 고정 (움직이지 않음)
    robot_z = TCP_BASE_OFFSET_Z   # scale_z 적용 안 함

    return {"x": round(robot_x, 2), "y": round(robot_y, 2), "z": round(robot_z, 2)}



# ======================================================
# 4️⃣ YOLO 감지 함수
# ======================================================
def detect_yolo(model, frame):
    results = model.predict(frame, imgsz=640, conf=0.6, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    frame_vis = results[0].plot()
    detected_info = []
    FIXED_DISTANCE_CM = 30.0

    if len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        detected_info.append(("object", (cx, cy), FIXED_DISTANCE_CM))
    return frame_vis, detected_info


# ======================================================
# 5️⃣ 카메라 스레드 (프레임 송출만)
# ======================================================
def camera_capture_thread(stop_event, frame_container):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("⚠️ 카메라를 열 수 없습니다.")
        return

    print("📷 카메라 스레드 시작 (프레임 송출 중...)")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame_container["frame"] = frame
    cap.release()
    print("📷 카메라 스레드 종료")


# ======================================================
# 6️⃣ 로봇 이동 헬퍼
# ======================================================
def move_to(mc, name, speed=DEFAULT_SPEED):
    if name not in POSES:
        print(f"⚠️ Unknown pose: {name}")
        return
    target = POSES[name]
    mc.send_coords(target, speed, 1)
    time.sleep(2)
    print(f"✅ Move → {name}")


# ======================================================
# 7️⃣ 좌표 JSON 저장
# ======================================================
def save_pick_coordinate(coord, filename="picking_target.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(coord, f, indent=4, ensure_ascii=False)
    print(f"💾 좌표 저장 완료 → {filename} : {coord}")


# ======================================================
# 8️⃣ 메인 루프
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--model", type=str, default="/home/vboxuser/robotarm/best.pt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # YOLO 모델 로드
    print(f"🧠 YOLO 모델 로드 중: {args.model}")
    model = YOLO(args.model)
    print("✅ YOLO 모델 로드 완료")

    # 카메라 보정값 로드
    camera_matrix, dist_coeffs = load_camera_params()

    # 로봇 연결
    mc = None
    if not args.dry_run:
        mc = CobotClass(args.port, args.baud)
        time.sleep(0.5)
        mc.power_on()
        print("🔌 Power ON 완료")
        move_to(mc, "Home", args.speed)
        mc.set_gripper_mode(0)
        mc.set_electric_gripper(0)
        mc.set_gripper_value(0, 20, 1)  # 열림
    else:
        print("🟡 dry-run 모드 (로봇 미연결)")

    # 카메라 스레드 시작
    frame_container = {"frame": None}
    stop_event = threading.Event()
    cam_thread = threading.Thread(
        target=camera_capture_thread, args=(stop_event, frame_container), daemon=True
    )
    cam_thread.start()

    print("✅ 메인 루프 시작 (ROI 감지 후 3초 유지 시 실행)")
    roi_detect_start = None
    DETECT_HOLD_TIME = 3.0
    detected_coord = None

    try:
        while not stop_event.is_set():
            frame = frame_container.get("frame")
            if frame is None:
                continue

            # ROI 표시
            h, w, _ = frame.shape
            roi_x1, roi_y1 = int(w * 0.3), int(h * 0.3)
            roi_x2, roi_y2 = int(w * 0.7), int(h * 0.7)
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            cv2.drawMarker(frame, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

            processed_frame, detected = detect_yolo(model, frame)
            in_roi = False

            if detected:
                _, (cx, cy), dist = detected[0]
                if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                    in_roi = True

            if in_roi:
                if roi_detect_start is None:
                    roi_detect_start = time.time()
                    print("🔵 ROI 감지 시작 (3초 유지 시 좌표 확정)")
                else:
                    elapsed = time.time() - roi_detect_start
                    cv2.putText(processed_frame, f"감지 중... {elapsed:.1f}s", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    if elapsed >= DETECT_HOLD_TIME:
                        print("🟢 감지 유지 3초 → 좌표 계산 시작")
                        detected_coord = pixel_to_robot(cx, cy, dist, camera_matrix, dist_coeffs)
                        print(f"🎯 물체 좌표: {detected_coord}")
                        # 감지된 물체 좌표
                        # save_pick_coordinate(detected_coord)

                        # ✅ 카메라 종료
                        stop_event.set()
                        cam_thread.join()
                        cv2.destroyAllWindows()
                        print("📷 카메라 종료 완료")

                        break
            else:
                roi_detect_start = None

            cv2.imshow("Camera View", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    finally:
        stop_event.set()
        cam_thread.join()
        cv2.destroyAllWindows()

    # ==================================================
    # ✅ 감지된 좌표가 있으면 로봇 이동
    # ==================================================
    if detected_coord:
        print("🤖 로봇 이동 시작...")
        if not args.dry_run and mc:
            mc.set_gripper_state(0, 80)   # 완전 열기
            mc.send_coords([detected_coord["x"], detected_coord["y"], 300.0, -175.33, 8.65, 86.68], 25, 1)
            time.sleep(3)
            mc.send_coords([detected_coord["x"]-15, detected_coord["y"], 260.0+30, -175.33, 8.65, 86.68], 15, 1)
            time.sleep(2)
            mc.set_gripper_state(1, 80)   # 닫기
            mc.send_coords([detected_coord["x"], detected_coord["y"], 260.0+50, -175.33, 8.65, 86.68], 15, 1)
            time.sleep(2)
            #멈춤
            # exit()
            time.sleep(1.5)
            move_to(mc, "Clear", args.speed)
            move_to(mc, "Place", args.speed)
            mc.set_gripper_state(0, 80)
            move_to(mc, "Home", args.speed)
        else:
            print(f"🟢 [dry-run] 좌표 기반 시뮬레이션 완료: {detected_coord}")

    if mc:
        mc.power_off()
    print("🔒 종료 완료")


if __name__ == "__main__":
    main()
