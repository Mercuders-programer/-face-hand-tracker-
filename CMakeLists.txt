cmake_minimum_required(VERSION 3.18)
project(PoseTracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================================
# 경로 설정 — CMake GUI 또는 cmake -D 옵션으로 지정
# =========================================================================

# OpenPose 설치 경로 (예: C:/openpose 또는 빌드된 경우 C:/openpose/build)
set(OPENPOSE_DIR "C:/openpose" CACHE PATH "OpenPose 설치/빌드 루트 경로")

# =========================================================================
# OpenCV 탐색 (OpenPose와 같이 설치된 버전 사용 권장)
# =========================================================================
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV: ${OpenCV_VERSION}  libs: ${OpenCV_LIBS}")

# =========================================================================
# nlohmann/json — 단일 헤더 자동 다운로드
# =========================================================================
set(JSON_HPP "${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann/json.hpp")
if(NOT EXISTS "${JSON_HPP}")
    message(STATUS "nlohmann/json.hpp 다운로드 중...")
    file(DOWNLOAD
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
        "${JSON_HPP}"
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
    )
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_CODE)
    if(NOT DOWNLOAD_CODE EQUAL 0)
        message(FATAL_ERROR "nlohmann/json 다운로드 실패. 수동으로 다운받아 include/nlohmann/json.hpp 에 놓으세요.")
    endif()
endif()

# =========================================================================
# 실행 파일 빌드
# =========================================================================
set(SOURCES
    src/main.cpp
    src/Tracker.cpp
    src/Exporter.cpp
)

add_executable(PoseTracker ${SOURCES})

target_include_directories(PoseTracker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # OpenPose 헤더
    ${OPENPOSE_DIR}/include
    # OpenCV 헤더
    ${OpenCV_INCLUDE_DIRS}
)

# =========================================================================
# 라이브러리 링크
# =========================================================================

# openpose.lib 위치: 빌드된 경우 build/Release/, 설치된 경우 lib/
find_library(OPENPOSE_LIB
    NAMES openpose
    PATHS
        "${OPENPOSE_DIR}/lib"
        "${OPENPOSE_DIR}/build/Release"
        "${OPENPOSE_DIR}/build/Debug"
        "${OPENPOSE_DIR}/x64/Release"
    NO_DEFAULT_PATH
)

if(NOT OPENPOSE_LIB)
    message(FATAL_ERROR
        "openpose.lib 를 찾을 수 없습니다.\n"
        "OPENPOSE_DIR 을 올바르게 설정하세요: ${OPENPOSE_DIR}\n"
        "cmake -DOPENPOSE_DIR=<경로> ..")
endif()
message(STATUS "OpenPose lib: ${OPENPOSE_LIB}")

target_link_libraries(PoseTracker PRIVATE
    ${OPENPOSE_LIB}
    ${OpenCV_LIBS}
)

# =========================================================================
# Windows: OpenPose DLL 빌드 결과 폴더로 복사
# =========================================================================
if(WIN32)
    # openpose.dll
    find_file(OPENPOSE_DLL
        NAMES openpose.dll
        PATHS
            "${OPENPOSE_DIR}/bin"
            "${OPENPOSE_DIR}/build/Release"
            "${OPENPOSE_DIR}/x64/Release"
        NO_DEFAULT_PATH
    )
    if(OPENPOSE_DLL)
        add_custom_command(TARGET PoseTracker POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENPOSE_DLL}"
                "$<TARGET_FILE_DIR:PoseTracker>"
            COMMENT "openpose.dll 복사"
        )
    endif()

    # Release 모드 최적화
    target_compile_options(PoseTracker PRIVATE $<$<CONFIG:Release>:/O2>)
endif()

# =========================================================================
# 빌드 후 models 폴더 안내
# =========================================================================
add_custom_command(TARGET PoseTracker POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo
        "빌드 완료. OpenPose models/ 폴더가 실행 파일 옆에 있어야 합니다."
    COMMENT "안내"
)
