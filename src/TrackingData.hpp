#pragma once
#include <string>
#include <vector>

namespace tracker {

// -----------------------------------------------------------------------
// 기본 2D 좌표 + 신뢰도
// -----------------------------------------------------------------------
struct Point2D {
    float x          = 0.0f;
    float y          = 0.0f;
    float confidence = 0.0f;

    bool valid(float threshold = 0.05f) const { return confidence >= threshold; }
};

// -----------------------------------------------------------------------
// 얼굴 랜드마크 (OpenPose 70-point face model)
//
// OpenPose 인덱스 매핑 (68-point face alignment + 2 pupils):
//  0-16   jaw line
//  17-21  right eyebrow    22-26  left eyebrow
//  27-30  nose bridge      31-35  nose bottom
//  36-41  right eye        42-47  left eye
//  48-67  mouth
//  68     right pupil      69     left pupil
// -----------------------------------------------------------------------
struct FaceLandmarks {
    bool detected = false;

    // --- 눈 ---
    Point2D rightEyeOuterCorner;  // idx 36
    Point2D rightEyeInnerCorner;  // idx 39
    Point2D rightPupil;           // idx 68
    Point2D leftEyeInnerCorner;   // idx 42
    Point2D leftEyeOuterCorner;   // idx 45
    Point2D leftPupil;            // idx 69

    // --- 코 ---
    Point2D noseBridgeTop;        // idx 27
    Point2D noseTip;              // idx 33 (콧등 중앙 하단)

    // --- 입 ---
    Point2D mouthRightCorner;     // idx 48
    Point2D mouthUpperCenter;     // idx 51
    Point2D mouthLeftCorner;      // idx 54
    Point2D mouthLowerCenter;     // idx 57

    // 전체 70개 원본 좌표 (raw)
    std::vector<Point2D> all;     // size = 70
};

// -----------------------------------------------------------------------
// 손 랜드마크 (OpenPose / MediaPipe 공통 21-point hand model)
//
//  0   wrist
//  1-4   thumb  (CMC→MCP→IP→TIP)
//  5-8   index  (MCP→PIP→DIP→TIP)
//  9-12  middle
// 13-16  ring
// 17-20  pinky
// -----------------------------------------------------------------------
struct HandLandmarks {
    bool        detected = false;
    std::string side;              // "left" or "right"

    std::vector<Point2D> landmarks; // size = 21

    // 편의 접근자
    const Point2D& wrist()     const { return landmarks[0];  }
    const Point2D& thumbTip()  const { return landmarks[4];  }
    const Point2D& indexTip()  const { return landmarks[8];  }
    const Point2D& middleTip() const { return landmarks[12]; }
    const Point2D& ringTip()   const { return landmarks[16]; }
    const Point2D& pinkyTip()  const { return landmarks[20]; }
};

// -----------------------------------------------------------------------
// 프레임 단위 추적 결과
// -----------------------------------------------------------------------
struct FrameData {
    int    index     = 0;
    double timestamp = 0.0;   // seconds

    FaceLandmarks face;
    HandLandmarks leftHand;
    HandLandmarks rightHand;
};

// -----------------------------------------------------------------------
// 영상 메타데이터
// -----------------------------------------------------------------------
struct VideoInfo {
    int    width       = 0;
    int    height      = 0;
    double fps         = 30.0;
    int    totalFrames = 0;
};

} // namespace tracker
