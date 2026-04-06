#include "Tracker.hpp"

// OpenPose
#include <openpose/flags.hpp>
#include <openpose/headers.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdexcept>

namespace tracker {

// -----------------------------------------------------------------------
// 얼굴 랜드마크 인덱스 (OpenPose 70-point)
// -----------------------------------------------------------------------
namespace FaceIdx {
    constexpr int RIGHT_EYE_OUTER  = 36;
    constexpr int RIGHT_EYE_INNER  = 39;
    constexpr int LEFT_EYE_INNER   = 42;
    constexpr int LEFT_EYE_OUTER   = 45;
    constexpr int NOSE_BRIDGE_TOP  = 27;
    constexpr int NOSE_TIP         = 33;
    constexpr int MOUTH_RIGHT      = 48;
    constexpr int MOUTH_TOP        = 51;
    constexpr int MOUTH_LEFT       = 54;
    constexpr int MOUTH_BOTTOM     = 57;
    constexpr int RIGHT_PUPIL      = 68;
    constexpr int LEFT_PUPIL       = 69;
    constexpr int TOTAL            = 70;
}

// -----------------------------------------------------------------------
// 손 랜드마크 인덱스
// -----------------------------------------------------------------------
namespace HandIdx {
    constexpr int TOTAL = 21;
}

// -----------------------------------------------------------------------
Tracker::Tracker()  = default;
Tracker::~Tracker() = default;

void Tracker::setGpuDevice(int deviceId)       { m_gpuDevice  = deviceId; }
void Tracker::setFaceDetectorMode(int mode)    { m_faceMode   = mode;     }
void Tracker::setConfidenceThreshold(float t)  { m_confThresh = t;        }

// -----------------------------------------------------------------------
// op::Array<float>에서 Point2D 추출 헬퍼
// -----------------------------------------------------------------------
static Point2D toPoint(const op::Array<float>& arr, int person, int kpIdx) {
    Point2D p;
    p.x          = arr[{person, kpIdx, 0}];
    p.y          = arr[{person, kpIdx, 1}];
    p.confidence = arr[{person, kpIdx, 2}];
    return p;
}

// -----------------------------------------------------------------------
// 얼굴 키포인트 → FaceLandmarks
// -----------------------------------------------------------------------
FaceLandmarks Tracker::extractFace(const float* /*unused*/,
                                   int           /*unused*/,
                                   int           /*unused*/) const
{
    // 실제 사용 안 함 — processVideo 내부에서 직접 op::Array 접근
    return {};
}

FaceLandmarks extractFaceFromDatum(const op::Array<float>& faceKP, float thresh) {
    FaceLandmarks f;
    if (faceKP.empty() || faceKP.getSize(0) == 0) return f;

    const int person = 0; // 첫 번째 사람만 사용

    // 전체 70개 저장
    f.all.resize(FaceIdx::TOTAL);
    for (int i = 0; i < FaceIdx::TOTAL; ++i) {
        f.all[i] = toPoint(faceKP, person, i);
    }

    // 명명된 포인트
    f.rightEyeOuterCorner = f.all[FaceIdx::RIGHT_EYE_OUTER];
    f.rightEyeInnerCorner = f.all[FaceIdx::RIGHT_EYE_INNER];
    f.rightPupil          = f.all[FaceIdx::RIGHT_PUPIL];
    f.leftEyeInnerCorner  = f.all[FaceIdx::LEFT_EYE_INNER];
    f.leftEyeOuterCorner  = f.all[FaceIdx::LEFT_EYE_OUTER];
    f.leftPupil           = f.all[FaceIdx::LEFT_PUPIL];
    f.noseBridgeTop       = f.all[FaceIdx::NOSE_BRIDGE_TOP];
    f.noseTip             = f.all[FaceIdx::NOSE_TIP];
    f.mouthRightCorner    = f.all[FaceIdx::MOUTH_RIGHT];
    f.mouthUpperCenter    = f.all[FaceIdx::MOUTH_TOP];
    f.mouthLeftCorner     = f.all[FaceIdx::MOUTH_LEFT];
    f.mouthLowerCenter    = f.all[FaceIdx::MOUTH_BOTTOM];

    // 주요 포인트 중 하나라도 유효하면 detected = true
    f.detected = f.rightPupil.valid(thresh) || f.leftPupil.valid(thresh)
              || f.noseTip.valid(thresh)    || f.mouthUpperCenter.valid(thresh);
    return f;
}

// -----------------------------------------------------------------------
// 손 키포인트 → HandLandmarks
// -----------------------------------------------------------------------
HandLandmarks Tracker::extractHand(const float* /*unused*/,
                                   int           /*unused*/,
                                   int           /*unused*/,
                                   const std::string& /*unused*/) const
{
    return {};
}

HandLandmarks extractHandFromDatum(const op::Array<float>& handKP,
                                   const std::string&       side,
                                   float                    thresh) {
    HandLandmarks h;
    h.side = side;
    if (handKP.empty() || handKP.getSize(0) == 0) return h;

    const int person = 0;
    h.landmarks.resize(HandIdx::TOTAL);
    for (int i = 0; i < HandIdx::TOTAL; ++i) {
        h.landmarks[i] = toPoint(handKP, person, i);
    }

    // 손목이 유효하면 detected
    h.detected = h.wrist().valid(thresh);
    return h;
}

// -----------------------------------------------------------------------
// 핵심: 영상 처리
// -----------------------------------------------------------------------
bool Tracker::processVideo(const std::string&      videoPath,
                            std::vector<FrameData>& outFrames,
                            VideoInfo&              outInfo,
                            bool                    showPreview,
                            ProgressCallback        callback) {
    outFrames.clear();

    // ── OpenCV 영상 열기 ─────────────────────────────────────────────────
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[Tracker] 영상 파일을 열 수 없습니다: " << videoPath << "\n";
        return false;
    }

    outInfo.width       = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    outInfo.height      = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    outInfo.fps         = cap.get(cv::CAP_PROP_FPS);
    outInfo.totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    if (outInfo.fps <= 0.0) outInfo.fps = 30.0;

    std::cout << "[Tracker] 영상: " << outInfo.width << "x" << outInfo.height
              << " @ " << outInfo.fps << " fps, 총 " << outInfo.totalFrames << " 프레임\n";

    // ── OpenPose 초기화 ──────────────────────────────────────────────────
    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};

    // Body pose
    op::WrapperStructPose poseParams{};
    poseParams.gpuNumber     = 1;
    poseParams.gpuNumberStart = m_gpuDevice;
    poseParams.netInputSize  = op::Point<int>{-1, 368};
    opWrapper.configure(poseParams);

    // Face
    op::WrapperStructFace faceParams{};
    faceParams.enable     = true;
    faceParams.detector   = static_cast<op::FaceDetectorMode>(m_faceMode);
    faceParams.netInputSize = op::Point<int>{368, 368};
    opWrapper.configure(faceParams);

    // Hands
    op::WrapperStructHand handParams{};
    handParams.enable      = true;
    handParams.detector    = op::HandDetectorMode::BodyWithTracking;
    handParams.netInputSize = op::Point<int>{368, 368};
    handParams.scaleNumber = 1;
    opWrapper.configure(handParams);

    // 화면 출력 비활성화 (직접 OpenCV 창 사용)
    opWrapper.configure(op::WrapperStructOutput{});

    opWrapper.start();
    std::cout << "[Tracker] OpenPose 초기화 완료 (GPU " << m_gpuDevice << ")\n";

    // ── 프레임 처리 루프 ─────────────────────────────────────────────────
    cv::Mat frame;
    int frameIdx = 0;

    while (cap.read(frame)) {
        // OpenPose 처리
        auto datumProcessed = opWrapper.emplaceAndPop(OP_CV2OPCONSTMAT(frame));

        FrameData fd;
        fd.index     = frameIdx;
        fd.timestamp = static_cast<double>(frameIdx) / outInfo.fps;

        if (datumProcessed != nullptr && !datumProcessed->empty()) {
            const auto& datum = datumProcessed->at(0);

            // 얼굴
            if (!datum.faceKeypoints.empty()) {
                fd.face = extractFaceFromDatum(datum.faceKeypoints, m_confThresh);
            }

            // 왼손 (index 0), 오른손 (index 1)
            if (datum.handKeypoints.size() > 0 && !datum.handKeypoints[0].empty()) {
                fd.leftHand = extractHandFromDatum(datum.handKeypoints[0], "left", m_confThresh);
            }
            if (datum.handKeypoints.size() > 1 && !datum.handKeypoints[1].empty()) {
                fd.rightHand = extractHandFromDatum(datum.handKeypoints[1], "right", m_confThresh);
            }

            // 미리보기
            if (showPreview) {
                cv::Mat preview = OP_OP2CVCONSTMAT(datum.cvOutputData);
                if (!preview.empty()) {
                    cv::imshow("PoseTracker Preview  (Q: 종료)", preview);
                    if (cv::waitKey(1) == 'q') {
                        std::cout << "[Tracker] 사용자가 미리보기를 종료했습니다.\n";
                        showPreview = false;
                        cv::destroyAllWindows();
                    }
                }
            }
        }

        outFrames.push_back(std::move(fd));

        // 진행 콜백
        if (callback) {
            callback(frameIdx + 1, outInfo.totalFrames);
        } else if (frameIdx % 30 == 0) {
            int pct = outInfo.totalFrames > 0
                      ? static_cast<int>(100.0 * frameIdx / outInfo.totalFrames)
                      : 0;
            std::cout << "\r[Tracker] 처리 중... " << pct << "% ("
                      << frameIdx << "/" << outInfo.totalFrames << ")" << std::flush;
        }

        ++frameIdx;
    }

    std::cout << "\n[Tracker] 완료: " << outFrames.size() << " 프레임 처리됨\n";

    if (showPreview) cv::destroyAllWindows();
    opWrapper.stop();
    return true;
}

} // namespace tracker
