#pragma once
#include "TrackingData.hpp"
#include <string>
#include <vector>
#include <functional>

namespace tracker {

// 프레임 처리 진행 콜백: (현재프레임, 전체프레임)
using ProgressCallback = std::function<void(int current, int total)>;

class Tracker {
public:
    Tracker();
    ~Tracker();

    // GPU 디바이스 번호 설정 (기본값 0)
    void setGpuDevice(int deviceId);

    // 얼굴 검출기 모드
    // 0 = OpenPose body 기반 (기본, 빠름)
    // 2 = RetinaFace (전신 없이 얼굴만 찍을 때 더 안정적)
    void setFaceDetectorMode(int mode);

    // 신뢰도 임계값 (이 이하인 랜드마크는 미검출로 처리)
    void setConfidenceThreshold(float threshold);

    // 영상을 프레임 단위로 처리
    // videoPath : 입력 영상 파일 경로
    // outFrames : 프레임별 추적 결과 (출력)
    // outInfo   : 영상 메타데이터 (출력)
    // showPreview : OpenCV 창으로 실시간 미리보기 여부
    // callback  : 진행 상황 콜백 (nullptr 가능)
    bool processVideo(const std::string&      videoPath,
                      std::vector<FrameData>& outFrames,
                      VideoInfo&              outInfo,
                      bool                    showPreview = true,
                      ProgressCallback        callback    = nullptr);

private:
    int   m_gpuDevice   = 0;
    int   m_faceMode    = 0;
    float m_confThresh  = 0.05f;

    FaceLandmarks extractFace(const float* data, int numPeople, int numKP) const;
    HandLandmarks extractHand(const float* data, int numPeople, int numKP,
                              const std::string& side) const;
};

} // namespace tracker
