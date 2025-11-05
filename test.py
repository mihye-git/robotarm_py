# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 (pymycobot)
카메라로 특정 색을 감지한 뒤,
- 감지한 물체의 화면 내 좌표와 거리(대략)를 계산하고
- 그 값을 로봇 좌표계로 변환해서 JSON 파일로 저장하기 위한 예제

※ 현재 버전에서는 "로봇이 실제로 정렬해서 움직이는 부분"이 주석 처리돼 있고,
   카메라에서 색을 찾고 좌표를 만드는 흐름이 남아 있음.
"""

# -----------------------------
# 기본 라이브러리 임포트
# -----------------------------
import threading       # 카메라를 별도 스레드로 돌리기 위해 사용
import cv2             # OpenCV: 카메라 캡처, 색 추출, 화면 표시
import time            # 대기(sleep) 처리
import argparse        # 실행 시 옵션(--port, --color 등) 받기
import numpy as np     # 영상 처리 시 배열 연산
import json, os        # 좌표를 JSON으로 저장 / 파일 존재 여부 확인

# 전역 플래그: 한 번 피킹 좌표를 저장하면 True로 바꿔서 중복 저장을 방지
picking_done = False

# -----------------------------
# 로봇 클래스 임포트
# -----------------------------
# 사용 환경에 따라 mycobot320이 있기도 하고, 일반 mycobot만 있을 수도 있어
# 두 경우를 모두 커버하기 위한 try-except
try:
    # MyCobot 320 전용 클래스
    from pymycobot.mycobot320 import MyCobot320 as CobotClass
except Exception:
    # 위 임포트가 실패하면 일반 MyCobot 클래스를 사용
    from pymycobot.mycobot import MyCobot as CobotClass

# -----------------------------
# 자주 쓰는 포즈(좌표) 정의
# 이 값은 사용자가 테스트하면서 미리 뽑아둔 좌표라고 보면 됨.
# send_coords 형태의 6자유도 포맷: [x, y, z, rx, ry, rz]
# -----------------------------
POSES = {
    "Home":  [59.8, -215.9, 354.6, -175.33, 8.65, 86.68],  # 시작/대기 위치
    "Place": [-16.08, 84.01, -15.20, 3.86, -86.39, -16.96],  # 예: 내려둘 위치
}

# 로봇 이동 시 기본 속도
DEFAULT_SPEED = 20


# ======================================================================
# 1. 픽셀 좌표 → 로봇 좌표 대략 변환 함수
# ======================================================================
def pixel_to_robot(cx, cy, distance_cm, frame_w, frame_h):
    """
    화면(이미지) 상의 중심점(cx, cy)과 실제 거리 값(대략)을 받아
    로봇이 이해할 수 있는 x, y, z 좌표로 바꿔주는 함수.

    실제로는 카메라와 로봇의 상대 위치, 카메라 높이, 각도에 따라
    꽤 많은 보정이 필요하지만 여기서는 '대략 이렇게 변환한다'는 예시를 보여줌.
    """

    # 카메라 화면의 중심점(픽셀). 여기 기준으로 얼마나 벗어났는지 계산하려고 구해둠.
    center_x, center_y = frame_w / 2, frame_h / 2

    # 1픽셀이 실제 몇 mm인지에 대한 스케일값.
    # 실제 환경에서는 캘리브레이션으로 이 값을 맞춰야 함.
    scale = 0.4  # mm/pixel

    # ------------------------------------------------------------
    # 방향 보정
    # ------------------------------------------------------------
    # cx - center_x : 화면 중심에서 얼마나 오른쪽(+)으로 치우쳐 있는지
    # cy - center_y : 화면 중심에서 얼마나 아래쪽(+)으로 치우쳐 있는지
    #
    # 그런데 로봇 좌표계와 카메라 좌표계의 축 방향이 다를 수 있으므로
    # 여기서는 음수(-)를 붙여서 "카메라 오른쪽 → 로봇 왼쪽" 식으로 반대 변환
    dx = -(cx - center_x) * scale        # X축 보정량 (mm)
    dy = -(cy - center_y) * scale        # Y축 보정량 (mm)

    # z는 거리 기반으로 계산.
    # distance_cm는 카메라에서 물체까지의 거리를 "대략" 잰 값.
    # 여기서는 물체에 완전히 붙지 않고 20cm 정도 떨어져 멈추도록 (distance_cm - 20)
    # 그리고 로봇 좌표는 mm 단위로 쓴다고 가정해서 * 10
    dz = (distance_cm - 20) * 10

    # ------------------------------------------------------------
    # 로봇 기준 오프셋
    # ------------------------------------------------------------
    # 카메라가 로봇 툴 중앙에 딱 달려있지 않은 경우가 많음.
    # 예를 들어 카메라가 로봇 기준으로 x쪽으로 120mm 떨어져 있다면
    # 이만큼을 기본값으로 더해줘야 함.
    ROBOT_OFFSET_X = 120.0
    ROBOT_OFFSET_Y = 0.0
    ROBOT_OFFSET_Z = 30.0

    # 카메라 기준에서 로봇 기준으로 변환한 좌표
    robot_x = ROBOT_OFFSET_X + dx
    robot_y = ROBOT_OFFSET_Y + dy
    robot_z = ROBOT_OFFSET_Z + dz

    # 소수점 2자리까지 반올림해서 dict로 반환
    return {
        "x": round(robot_x, 2),
        "y": round(robot_y, 2),
        "z": round(robot_z, 2)
    }


# ======================================================================
# 2. 계산된 피킹 좌표를 JSON 파일로 저장하는 함수
# ======================================================================
def save_pick_coordinate(data, filename="picking_target.json"):
    """
    data: {"x": ..., "y": ..., "z": ...} 형태의 dict
    filename: 저장할 파일명
    """
    with open(filename, "w", encoding="utf-8") as f:
        # indent=4 로 예쁘게 들여쓰기, ensure_ascii=False로 한글도 그대로
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"💾 피킹 좌표 저장 완료 → {filename} / {data}")


# ======================================================================
# 3. 프레임(이미지)에서 특정 색을 찾고, 그 위치와 거리까지 계산하는 함수
# ======================================================================
def detect_color_and_distance(frame, target_color="blue"):
    """
    1) 입력받은 frame에서 ROI(가운데 영역)를 지정
    2) 해당 영역에서 HSV 색공간으로 변환
    3) 지정한 색 범위에 맞는 마스크를 만들고
    4) 가장 큰 컨투어(색 덩어리)를 찾아서
    5) 그 중심점, 바운딩 박스 크기 → 거리, 중심과의 오프셋을 계산해서 돌려줌
    """

    # 원본 프레임의 높이/너비
    h, w, _ = frame.shape

    # 화면 중앙 좌표 (전체 프레임 기준)
    center_x, center_y = w // 2, h // 2

    # -----------------------------
    # ROI(Region of Interest) 설정
    # 화면 전체에서 찾으면 노이즈도 많고 정확도 떨어질 수 있으니,
    # 화면 가운데 30%~70% 구간만 본다는 의미
    # -----------------------------
    roi_x1, roi_y1 = int(w * 0.3), int(h * 0.3)  # 좌상단
    roi_x2, roi_y2 = int(w * 0.7), int(h * 0.7)  # 우하단

    # 실제 ROI 이미지 잘라오기
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # 디버깅을 위해 ROI 영역을 화면에 표시 (녹색 사각형)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    # 화면 중앙에도 십자 마커 그리기 (로봇이 맞출 기준점)
    cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0),
                   cv2.MARKER_CROSS, 15, 2)

    # ROI를 HSV로 변환 (색 검출은 HSV가 더 안정적)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # -----------------------------
    # 색상 범위 사전
    # 필요에 따라 여기 추가(orange, purple 등)
    # -----------------------------
    color_ranges = {
        "red":    [(0, 120, 70),  (10, 255, 255)],
        "green":  [(35, 80, 40),  (85, 255, 255)],
        "blue":   [(100, 80, 40), (140, 255, 255)],
        "yellow": [(20, 100, 100), (35, 255, 255)],
    }

    # 만약 사용자가 지정한 색이 위에 없으면 그냥 빈 결과 반환
    if target_color not in color_ranges:
        return frame, []

    # 선택된 색 범위 가져오기
    lower, upper = color_ranges[target_color]

    # 색 범위에 해당하는 마스크 생성
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    # 마스크에서 외곽선(컨투어) 찾기
    # RETR_EXTERNAL: 가장 바깥쪽 것만
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 최종 결과를 담을 리스트
    # (색이름, (cx, cy), distance, offset_x, offset_y) 형태로 넣을 예정
    detected_info = []

    # 거리 계산용 상수
    # KNOWN_WIDTH: 실제 물체의 폭(cm)
    # FOCAL_LENGTH: 카메라 초점거리 (테스트값)
    KNOWN_WIDTH, FOCAL_LENGTH = 2.5, 620

    # 컨투어가 하나라도 있다면
    if contours:
        # 가장 큰 컨투어만 사용 (가장 가까이 있거나 가장 확실한 물체라고 가정)
        c = max(contours, key=cv2.contourArea)

        # 너무 작은 컨투어는 노이즈이므로 무시 (영역이 300px 이상일 때만 진행)
        if cv2.contourArea(c) > 300:
            # 컨투어를 사각형으로 둘러싸는 바운딩 박스
            x, y, w_box, h_box = cv2.boundingRect(c)

            # ROI 기준으로의 중심점 → 전체 프레임 기준으로 바꿔줌
            cx = roi_x1 + x + w_box // 2
            cy = roi_y1 + y + h_box // 2

            # 거리 추정 (아주 단순한 비례식 기반)
            # distance = (실제폭 * 초점거리) / 영상에서의 폭
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w_box

            # 화면 중심으로부터 얼마나 벗어났는지(픽셀 단위)
            offset_x = cx - center_x
            offset_y = cy - center_y

            # 최종 정보 리스트에 넣기
            detected_info.append(
                (target_color, (cx, cy), distance, offset_x, offset_y)
            )

            # ----- 시각화: 프레임 위에 표시 -----
            # 검출된 물체 박스 표시
            cv2.rectangle(
                frame,
                (roi_x1 + x, roi_y1 + y),
                (roi_x1 + x + w_box, roi_y1 + y + h_box),
                (255, 255, 0),
                2
            )
            # 물체 중심점 표시
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 15, 2)
            # 화면 중심 → 물체 중심으로 선 긋기
            cv2.line(frame, (center_x, center_y), (cx, cy), (0, 0, 255), 2)
            # 텍스트로 색, 거리, 오프셋 표시
            cv2.putText(
                frame,
                f"{target_color} {distance:.1f}cm Δ({offset_x},{offset_y})",
                (roi_x1 + x, roi_y1 + y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # 처리된 프레임(시각화 포함), 검출 정보 반환
    return frame, detected_info


# ======================================================================
# 4. 카메라 스레드
#    - 메인 스레드와 별도로 카메라를 계속 읽으면서 색을 찾음
#    - 찾으면 좌표 변환하고 JSON 저장
# ======================================================================
def camera_capture_thread(stop_event, frame_container,
                          target_color="blue", mc=None):
    """
    stop_event: 외부에서 True로 바꾸면 이 스레드가 종료되도록 하는 플래그
    frame_container: 최신 프레임을 메인 스레드에 건네주기 위한 dict
    target_color: 찾고 싶은 색상
    mc: 로봇 객체 (여기서는 실제 이동 부분이 주석 처리돼 있음)
    """
    global picking_done

    # 카메라 열기 (기본 0번 카메라)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ 카메라를 열 수 없습니다.")
        return

    print(f"📷 카메라 스레드 시작 (타깃 색상: {target_color})")

    # stop_event가 설정될 때까지 계속 반복
    while not stop_event.is_set():
        # 한 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            # 프레임을 못 받았으면 잠깐 쉬고 다음 루프
            time.sleep(0.01)
            continue

        # 프레임에서 색 검출 + 거리/오프셋 계산
        processed_frame, detected = detect_color_and_distance(
            frame, target_color
        )

        # 최신 프레임을 딕셔너리에 저장해서
        # 메인 루프에서 화면에 띄울 수 있게 함
        frame_container["frame"] = processed_frame

        # detected 리스트 안에는 여러 개가 있을 수도 있지만
        # 여기서는 감지된 것들을 한 번씩 돌면서 처리
        for color_name, (cx, cy), dist, offset_x, offset_y in detected:
            # 이미 한 번 좌표 저장이 끝났거나,
            # 로봇 객체가 없으면(=카메라만 테스트 중이면) 스킵
            if picking_done or mc is None:
                continue

            # 감지 정보 콘솔에 출력
            print(
                f"🎯 감지: {color_name} ({cx},{cy}) "
                f"dist={dist:.1f}cm Δx={offset_x}"
            )

            # ----------------------------------------------------------------
            # 아래 블록은 "감지된 물체를 화면 중앙에 오도록 로봇을 움직인다"는
            # 실제 제어 예시인데, 지금은 테스트용이라 주석 처리해둔 상태
            # 사용자가 환경에 맞게 기본 좌표, 보정 비율 등을 바꿔서 켜면 됨.
            # ----------------------------------------------------------------
            # # ✅ 실제 로봇(당신 테스트 기준)에 맞춘 X축 보정
            # if abs(offset_x) > 15:
            #     scale = 0.4  # mm/pixel
            #     dx_mm = offset_x * scale
            #
            #     # 기본 기준점 (테스트 환경에서 물체가 있을 법한 위치)
            #     base_x, base_y, base_z = -120.0, 0.0, 80.0
            #
            #     # offset_x가 클수록 x를 조금씩 조정
            #     # 여기서는 "오른쪽으로 치우쳤으면 앞으로 가라"는 식의
            #     # 사용자 환경에 맞춘 임의 보정이 들어감
            #     adj_x = base_x - dx_mm * 0.5
            #
            #     print(f"🤖 X축 중심 보정(앞으로): Δx={offset_x} → 이동 X={adj_x:.2f}")
            #     # 로봇에 좌표 명령 보내기
            #     mc.send_coords([adj_x, base_y, base_z, 180, 0, 90], 20, 1)
            #     time.sleep(1.5)
            #
            # # 중심 근처에 들어왔으면 좌표 저장
            # if abs(offset_x) <= 10:
            #     h, w, _ = frame.shape
            #     coord = pixel_to_robot(cx, cy, dist, w, h)
            #     save_pick_coordinate(coord)
            #     picking_done = True
            #     print(f"✅ 중심 정렬 완료 & 좌표 저장: {coord}")
            #     time.sleep(1)

    # while 종료 → 카메라 반납
    cap.release()
    print("📷 카메라 스레드 종료")


# ======================================================================
# 5. 로봇을 미리 정의한 포즈로 이동시키는 간단한 함수
# ======================================================================
def move_to(mc, name, speed=DEFAULT_SPEED):
    """
    mc   : 로봇 객체
    name : 위에서 정의한 POSES 딕셔너리의 키("Home", "Place" 등)
    speed: 이동 속도
    """
    if name not in POSES:
        print(f"⚠️ Unknown pose: {name}")
        return

    # 각도 기반이 아니라 좌표 기반 포맷이라면 send_coords를 써도 됨.
    # 여기서는 send_angles가 아니라 send_angles와 비슷한 역할을 하는
    # 명령을 쓴다고 보면 됨.
    mc.send_angles(POSES[name], speed)
    print(f"➡️ Move: {name}")
    time.sleep(2)  # 이동이 끝날 때까지 잠깐 대기


# ======================================================================
# 6. 메인 루프
#    - 실행 옵션 파싱
#    - 로봇 연결 및 홈 포즈 이동
#    - 카메라 스레드 시작
#    - 화면 표시
# ======================================================================
def main():
    # ----------------------------------------
    # 1) 명령줄 인자 파싱
    #    예) python this.py --color red --port COM3
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--color", type=str, default="blue")
    args = parser.parse_args()

    # ----------------------------------------
    # 2) 기존에 저장된 피킹 좌표가 있으면 지움
    #    매 실행마다 새로 잡는다고 가정하는 흐름
    # ----------------------------------------
    if os.path.exists("picking_target.json"):
        os.remove("picking_target.json")
        print("🧹 이전 picking_target.json 삭제 완료")

    # 최신 카메라 프레임을 다른 스레드와 공유하기 위한 컨테이너
    # 딕셔너리로 한 이유는 참조를 공유해서 바로 값만 바꿀 수 있게 하려고
    frame_container = {"frame": None}

    # 스레드를 멈출 때 쓰는 이벤트 객체
    stop_event = threading.Event()

    # ----------------------------------------
    # 3) 로봇 연결
    # ----------------------------------------
    mc = CobotClass(args.port, args.baud)  # 포트/보드레이트는 옵션에서
    time.sleep(0.5)                        # 연결 안정화 대기
    mc.power_on()                          # 서보 전원 ON
    print("🔌 Power ON 완료")

    # ----------------------------------------
    # 4) 홈 포즈로 이동
    # ----------------------------------------
    print("🏠 홈 위치로 이동 중...")
    # 여기서는 좌표 기반 send_coords 사용 (POSES["Home"]이 좌표 포맷이라서)
    mc.send_coords(POSES["Home"], args.speed)
    time.sleep(3)  # 이동 완료까지 대기
    print("✅ 홈 위치 도달 완료")

    # ----------------------------------------
    # 5) 카메라 스레드 시작
    # ----------------------------------------
    cam_thread = threading.Thread(
        target=camera_capture_thread,
        args=(stop_event, frame_container, args.color, mc),
        daemon=True  # 메인 프로그램 끝나면 자동 종료
    )
    cam_thread.start()

    # ----------------------------------------
    # 6) 메인 루프: 화면 띄우기 & q로 종료
    # ----------------------------------------
    print("✅ 메인 루프 시작 (q로 종료)")
    while not stop_event.is_set():
        # 카메라에서 최신으로 들어온 프레임 가져오기
        frame = frame_container.get("frame")
        if frame is not None:
            # 윈도우에 출력
            cv2.imshow("Camera View", frame)

        # 키보드 입력 확인
        # cv2.waitKey(1)은 1ms 대기 후 키 값을 받음
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # q를 누르면 stop_event를 세팅해서 모든 스레드 종료
            stop_event.set()
            break

    # 메인 루프가 끝나면 스레드도 종료시킴
    stop_event.set()
    cam_thread.join()  # 카메라 스레드가 완전히 끝날 때까지 대기

    # OpenCV 창 닫기
    cv2.destroyAllWindows()
    print("🔒 종료")


# ======================================================================
# 7. 파이썬 스크립트 엔트리포인트
# ======================================================================
if __name__ == "__main__":
    main()
