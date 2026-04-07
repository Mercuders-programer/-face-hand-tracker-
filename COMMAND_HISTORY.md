# 명령어 및 질문 이력 (COMMAND_HISTORY)

> 사용자가 AI에게 내린 모든 명령·질문을 기록합니다.
> 새로운 명령/질문이 있을 때마다 업데이트합니다.

---

## 2026-04-06

| # | 명령 / 질문 | 결과 |
|---|-------------|------|
| 1 | 첫 번째 버튼(녹화 중지)의 AE/JSON 저장 기능을 영상 분석 UI에도 버튼으로 추가해줘 | `src/video_panel.py` — JSON/AE 내보내기 버튼 추가 |
| 2 | 영상 저장 버튼 추가, 랜드마크 체크 시 오버레이 렌더링된 MP4로 저장 | `src/video_panel.py` — 영상 저장 버튼 추가 |
| 3 | 방금 추가된 내용 GitHub에 push 해줘 | push 완료 (73db5a5) |
| 4 | 영상 분석 UI에 파일 이름·프레임 수·경로·용량 등 영상 정보 표시, UI 배치 이쁘게 | 우측 사이드 패널 추가, 파일정보 7항목 표시 |
| 5 | 랜드마크·JSON·AE·영상저장 버튼도 오른쪽 패널에 붙여줘 | 좌측 컨트롤 바 제거, 우측 패널로 통합 |
| 6 | UI 변경 내용 GitHub에 push 해줘 | push 완료 (08cfad8) |
| 7 | session_2026-04-06.md는 왜 push가 안됐지? | .gitignore에 sessions/ 제외 설정 확인 |
| 8 | (gitignore에서 제거 후 push 선택) | sessions/ gitignore 제거 후 push 완료 (edd8252) |
| 9 | PHS_AI_Rule.md 미참조·세션 파일 미업데이트 이유 설명, ERROR_HISTORY.md 만들고 push 해줘 | ERROR_HISTORY.md 생성, 오류 원인·재발방지 기록, push 완료 (9a86b8a) |
| 10 | 처음 실행 화면 버튼 두 개 오른쪽에 버전 정보 v1.0.0 넣어줘 | `app.py` — 버튼 우측 사이드에 v1.0.0 추가 |
| 11 | 방금 추가한 기능 GitHub push 해줘 | push 완료 (faee194) |
| 12 | git push 해줘 | 이미 최신 상태 (변경사항 없음) |
| 13 | git push는 자동으로 하지 말고 내가 요청할 때만 해줘 | MEMORY.md에 Git 규칙 추가 |
| 14 | 카메라 추적·영상 분석 버튼 크기 동일하게 해줘 | `app.py` — `width=12` 명시로 통일 |
| 15 | 버전 정보를 처음 화면 오른쪽 하단 작은 글씨로 바꿔줘 | `app.py` — `place(relx=1.0, rely=1.0)` 우측 하단 고정, 8pt |
| 16 | 지금 내용 GitHub push 해줘 | push 완료 (7408464) |
| 17 | 지금까지 사용된 기능들 라이센스 문제 없나요 (가능 여부 질문) | Apache 2.0 / BSD / HPND / GPL예외 — 모두 문제없음 답변 |
| 18 | EXE 실행 파일 만드는 배치파일을 make_exe_file 이름으로 만들어줘 | `make_exe_file.bat` 생성 (6단계 자동화) |
| 19 | 경로가 어디죠 | `C:\Users\PHS\Desktop\Claude_Code\my_openPose_proj\make_exe_file.bat` 안내 |
| 20 | 배치파일 실행 결과를 x64\현재시각\ 폴더에 저장하게 해줘 | `make_exe_file.bat` — x64\YYYY-MM-DD_HH-mm-ss\ 출력 경로 추가 |
| 21 | 지금 내용 GitHub push 해줘 | push 완료 (6988537 → 배치파일 인코딩 오류 수정본 포함) |
| 22 | (배치파일 실행 오류 보고: 한글 깨짐, windowed 명령 오류 등) | 한글 전부 제거, ASCII 영문으로 재작성, ^ 줄이음 제거 |
| 23 | 지금 내용 GitHub push 해줘 | push 완료 (6988537) |
| 24 | 현재 프로젝트를 윈도우 실행파일로 만들 수 있나요 (가능 여부 질문) | PyInstaller로 가능, 주의사항 안내 |
| 25 | 코 랜드마크가 view에 안 보이고 저장도 안 됨 — 카메라·영상 분석 양쪽 다 적용해줘 | `camera_panel.py`, `video_panel.py` — cv2로 코 오버레이 직접 추가 |
| 26 | 기능 적용할 때마다 테스트하고 검증하는 거죠? | 문법 검사만 하고 있었음을 인정, ERROR_HISTORY 기록 |
| 27 | WSL2에서 직접 실행해서 테스트하게 하려면 어떻게 하지? | WSLg v1.0.71 이미 설치됨 확인, 직접 실행 가능 안내 |
| 28 | 앞으로 기능 만들 때마다 WSL2에서 직접 실행해서 확인, 안 되면 수정해줘 | MEMORY.md 테스트 규칙 추가, WSL2 실행 테스트 완료 |
| 29 | 방금 추가한 기능 GitHub push 해줘 | push 완료 (99fbd67) |
| 30 | 명령어 및 질문 모두 저장하는 파일 만들고 앞으로도 기록해줘 | `COMMAND_HISTORY.md` 생성 |
| 31 | 카메라/영상 분석 뷰에서 몸, 팔, 다리 위치 추적 추가 (어깨, 팔꿈치, 손목, 골반, 무릎, 발목) | `models/pose_landmarker_full.task` 다운로드, `tracker.py` PoseLandmarks 추가, `camera_panel.py`/`video_panel.py` 포즈 오버레이 추가, `exporter.py` 포즈 JSON/AE 내보내기 추가 |
| 32 | 랜드마크를 얼굴/몸/손 체크박스 3개로 나누고 저장(AE,JSON,영상)에도 적용 | `camera_panel.py`/`video_panel.py` — 3개 BooleanVar+체크박스 분리, `exporter.py` — include_face/body/hands 파라미터 추가 |
| 33 | make_exe_file.bat도 따로 반영할 건 없지? | STEP 3에 pose_landmarker_full.task 존재 확인 추가 |
| 34 | cmd에서 어떻게 실행하더라? | `cd C:\...\my_openPose_proj` → `python app.py` 안내 |
| 35 | command_history.md에 추가가 안 되는 이유 | 직전 질문 미기록 확인, 즉시 추가 |

---

<!-- 새 명령/질문 발생 시 위 표에 행 추가 -->
