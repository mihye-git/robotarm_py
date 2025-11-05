# -*- coding: utf-8 -*-
"""
MyCobot 320 M5 - 수동 캘리브레이션 테스트
------------------------------------------------
1️⃣ 토크 해제 → 로봇 팔을 손으로 움직이기
2️⃣ 원하는 자세에서 Enter 키 입력
3️⃣ 현재 좌표를 send_coords() 포맷으로 출력
"""

import time
from pymycobot.mycobot320 import MyCobot320

# ---------------------------------------------
# 기본 설정
# ---------------------------------------------
PORT = "/dev/ttyACM0"   # 실제 연결된 포트 확인 필요
BAUD = 115200

mc = MyCobot320(PORT, BAUD)
time.sleep(0.5)

print("🔌 로봇 연결 완료")

# ---------------------------------------------
# 토크 해제 (수동 조정 가능)
# ---------------------------------------------
print("\n⚙️ 서보 토크 해제 중... 손으로 로봇을 원하는 자세로 움직이세요.")
mc.release_all_servos()
print("✅ 토크 해제 완료. 자세 조정 후 Enter 키를 누르세요.\n")

input("👉 준비되면 Enter를 누르세요... ")

# ---------------------------------------------
# 현재 좌표 읽기
# ---------------------------------------------
coords = mc.get_coords()
angles = mc.get_angles()

print("\n📍 현재 로봇 위치 정보:")
print(f"  ➤ 좌표 (coords): {coords}")
print(f"  ➤ 조인트각 (angles): {angles}")

# ---------------------------------------------
# send_coords 포맷 출력
# ---------------------------------------------
if coords and len(coords) == 6:
    formatted = f"mc.send_coords({coords}, 20, 1)"
    print("\n✅ send_coords 명령어:")
    print("   " + formatted)
else:
    print("⚠️ 좌표를 읽을 수 없습니다. 연결 또는 전원 상태를 확인하세요.")

print("\n🔒 테스트 종료")
