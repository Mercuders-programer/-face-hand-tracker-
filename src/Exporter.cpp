#include "Exporter.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace tracker {

// -----------------------------------------------------------------------
// 헬퍼: Point2D → json object
// -----------------------------------------------------------------------
static json pointToJson(const Point2D& p) {
    return {{"x", p.x}, {"y", p.y}, {"confidence", p.confidence}};
}

// -----------------------------------------------------------------------
// JSON 전체 내보내기
// -----------------------------------------------------------------------
bool Exporter::exportJSON(const std::vector<FrameData>& frames,
                          const VideoInfo&               info,
                          const std::string&             outPath) {
    fs::create_directories(fs::path(outPath).parent_path());

    json root;

    // 메타데이터
    root["metadata"] = {
        {"fps",         info.fps},
        {"total_frames", info.totalFrames},
        {"width",       info.width},
        {"height",      info.height}
    };

    // 랜드마크 이름 테이블 (After Effects 참조용)
    root["landmark_info"] = {
        {"face", {
            {"total_points", 70},
            {"named", {
                {"right_eye_outer_corner", 36},
                {"right_eye_inner_corner", 39},
                {"right_pupil",            68},
                {"left_eye_inner_corner",  42},
                {"left_eye_outer_corner",  45},
                {"left_pupil",             69},
                {"nose_bridge_top",        27},
                {"nose_tip",               33},
                {"mouth_right_corner",     48},
                {"mouth_upper_center",     51},
                {"mouth_left_corner",      54},
                {"mouth_lower_center",     57}
            }}
        }},
        {"hand", {
            {"total_points", 21},
            {"names", {
                "wrist",
                "thumb_cmc","thumb_mcp","thumb_ip","thumb_tip",
                "index_mcp","index_pip","index_dip","index_tip",
                "middle_mcp","middle_pip","middle_dip","middle_tip",
                "ring_mcp","ring_pip","ring_dip","ring_tip",
                "pinky_mcp","pinky_pip","pinky_dip","pinky_tip"
            }}
        }}
    };

    // 프레임별 데이터
    json framesArr = json::array();
    for (const auto& fd : frames) {
        json jf;
        jf["frame"]     = fd.index;
        jf["timestamp"] = fd.timestamp;

        // --- 얼굴 ---
        json jface;
        jface["detected"] = fd.face.detected;
        if (fd.face.detected) {
            jface["named"] = {
                {"right_eye_outer_corner", pointToJson(fd.face.rightEyeOuterCorner)},
                {"right_eye_inner_corner", pointToJson(fd.face.rightEyeInnerCorner)},
                {"right_pupil",            pointToJson(fd.face.rightPupil)},
                {"left_eye_inner_corner",  pointToJson(fd.face.leftEyeInnerCorner)},
                {"left_eye_outer_corner",  pointToJson(fd.face.leftEyeOuterCorner)},
                {"left_pupil",             pointToJson(fd.face.leftPupil)},
                {"nose_bridge_top",        pointToJson(fd.face.noseBridgeTop)},
                {"nose_tip",               pointToJson(fd.face.noseTip)},
                {"mouth_right_corner",     pointToJson(fd.face.mouthRightCorner)},
                {"mouth_upper_center",     pointToJson(fd.face.mouthUpperCenter)},
                {"mouth_left_corner",      pointToJson(fd.face.mouthLeftCorner)},
                {"mouth_lower_center",     pointToJson(fd.face.mouthLowerCenter)}
            };

            // 전체 70개 raw
            json allArr = json::array();
            for (const auto& p : fd.face.all) allArr.push_back(pointToJson(p));
            jface["all_landmarks"] = allArr;
        }
        jf["face"] = jface;

        // --- 손 ---
        auto handToJson = [](const HandLandmarks& h) {
            json jh;
            jh["detected"] = h.detected;
            jh["side"]     = h.side;
            if (h.detected) {
                static const char* names[] = {
                    "wrist",
                    "thumb_cmc","thumb_mcp","thumb_ip","thumb_tip",
                    "index_mcp","index_pip","index_dip","index_tip",
                    "middle_mcp","middle_pip","middle_dip","middle_tip",
                    "ring_mcp","ring_pip","ring_dip","ring_tip",
                    "pinky_mcp","pinky_pip","pinky_dip","pinky_tip"
                };
                json lmArr = json::array();
                for (int i = 0; i < (int)h.landmarks.size(); ++i) {
                    json lm = pointToJson(h.landmarks[i]);
                    lm["index"] = i;
                    if (i < 21) lm["name"] = names[i];
                    lmArr.push_back(lm);
                }
                jh["landmarks"] = lmArr;
            }
            return jh;
        };

        jf["left_hand"]  = handToJson(fd.leftHand);
        jf["right_hand"] = handToJson(fd.rightHand);

        framesArr.push_back(jf);
    }
    root["frames"] = framesArr;

    // 파일 저장
    std::ofstream ofs(outPath);
    if (!ofs.is_open()) {
        std::cerr << "[Exporter] JSON 파일 쓰기 실패: " << outPath << "\n";
        return false;
    }
    ofs << std::setw(2) << root << "\n";
    std::cout << "[Exporter] JSON 저장 완료: " << outPath << "\n";
    return true;
}

// -----------------------------------------------------------------------
// AE Keyframe Data 단일 파일 작성
// -----------------------------------------------------------------------
bool Exporter::writeAEFile(const std::string&              path,
                           const std::string&              layerName,
                           const VideoInfo&                info,
                           const std::vector<FrameData>&   frames,
                           std::function<Point2D(const FrameData&)> getter) {
    fs::create_directories(fs::path(path).parent_path());

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[Exporter] AE 파일 쓰기 실패: " << path << "\n";
        return false;
    }

    ofs << std::fixed << std::setprecision(4);

    // ── 헤더 ──────────────────────────────────────────────────────────
    ofs << "Adobe After Effects 8.0 Keyframe Data\n\n";
    ofs << "\tUnits Per Second\t" << info.fps   << "\n";
    ofs << "\tSource Width\t"     << info.width  << "\n";
    ofs << "\tSource Height\t"    << info.height << "\n";
    ofs << "\tSource Pixel Aspect Ratio\t1\n";
    ofs << "\tComp Pixel Aspect Ratio\t1\n\n";

    // ── 키프레임 섹션 ─────────────────────────────────────────────────
    ofs << "Transform\tPosition\n";
    ofs << "\tFrame\tX pixels\tY pixels\tZ pixels\t\n";

    int written = 0;
    for (const auto& fd : frames) {
        Point2D p = getter(fd);
        if (!p.valid(0.05f)) continue;   // 미검출 프레임 스킵 (AE가 보간)

        ofs << "\t" << fd.index
            << "\t" << p.x
            << "\t" << p.y
            << "\t" << "0.0000" << "\t\n";
        ++written;
    }

    ofs << "\nEnd of Keyframe Data\n";

    if (written == 0) {
        std::cout << "[Exporter] 경고: '" << layerName << "' 는 유효한 키프레임 없음\n";
    }
    return true;
}

// -----------------------------------------------------------------------
// AE Keyframe Data 전체 내보내기
// -----------------------------------------------------------------------
bool Exporter::exportAEKeyframes(const std::vector<FrameData>& frames,
                                 const VideoInfo&               info,
                                 const std::string&             outDir) {
    const std::string faceDir      = outDir + "/face";
    const std::string handLeftDir  = outDir + "/hands/left";
    const std::string handRightDir = outDir + "/hands/right";

    bool ok = true;

    // ── 얼굴 (12 포인트) ─────────────────────────────────────────────
    struct FaceEntry {
        std::string name;
        std::function<Point2D(const FrameData&)> getter;
    };

    const std::vector<FaceEntry> facePoints = {
        {"right_eye_outer_corner", [](const FrameData& f){ return f.face.rightEyeOuterCorner; }},
        {"right_eye_inner_corner", [](const FrameData& f){ return f.face.rightEyeInnerCorner; }},
        {"right_pupil",            [](const FrameData& f){ return f.face.rightPupil;           }},
        {"left_eye_inner_corner",  [](const FrameData& f){ return f.face.leftEyeInnerCorner;  }},
        {"left_eye_outer_corner",  [](const FrameData& f){ return f.face.leftEyeOuterCorner;  }},
        {"left_pupil",             [](const FrameData& f){ return f.face.leftPupil;            }},
        {"nose_bridge_top",        [](const FrameData& f){ return f.face.noseBridgeTop;        }},
        {"nose_tip",               [](const FrameData& f){ return f.face.noseTip;              }},
        {"mouth_right_corner",     [](const FrameData& f){ return f.face.mouthRightCorner;     }},
        {"mouth_upper_center",     [](const FrameData& f){ return f.face.mouthUpperCenter;     }},
        {"mouth_left_corner",      [](const FrameData& f){ return f.face.mouthLeftCorner;      }},
        {"mouth_lower_center",     [](const FrameData& f){ return f.face.mouthLowerCenter;     }},
    };

    for (const auto& fp : facePoints) {
        ok &= writeAEFile(faceDir + "/" + fp.name + ".txt",
                          fp.name, info, frames, fp.getter);
    }

    // ── 손 — 주요 6개 포인트 × 2 (왼/오른) ──────────────────────────
    struct HandEntry {
        std::string name;
        std::function<Point2D(const HandLandmarks&)> getter;
        int lmIdx; // for all-landmarks export
    };

    const std::vector<HandEntry> handPoints = {
        {"wrist",      [](const HandLandmarks& h){ return h.wrist();      }, 0},
        {"thumb_tip",  [](const HandLandmarks& h){ return h.thumbTip();   }, 4},
        {"index_tip",  [](const HandLandmarks& h){ return h.indexTip();   }, 8},
        {"middle_tip", [](const HandLandmarks& h){ return h.middleTip();  }, 12},
        {"ring_tip",   [](const HandLandmarks& h){ return h.ringTip();    }, 16},
        {"pinky_tip",  [](const HandLandmarks& h){ return h.pinkyTip();   }, 20},
    };

    for (const auto& hp : handPoints) {
        // 왼손
        ok &= writeAEFile(handLeftDir  + "/" + hp.name + ".txt",
                          "left_"  + hp.name, info, frames,
                          [&hp](const FrameData& f){ return hp.getter(f.leftHand);  });
        // 오른손
        ok &= writeAEFile(handRightDir + "/" + hp.name + ".txt",
                          "right_" + hp.name, info, frames,
                          [&hp](const FrameData& f){ return hp.getter(f.rightHand); });
    }

    // ── 전체 손 랜드마크 21개 (선택적 — 상세 제어용) ─────────────────
    static const char* handLMNames[] = {
        "wrist",
        "thumb_cmc","thumb_mcp","thumb_ip","thumb_tip",
        "index_mcp","index_pip","index_dip","index_tip",
        "middle_mcp","middle_pip","middle_dip","middle_tip",
        "ring_mcp","ring_pip","ring_dip","ring_tip",
        "pinky_mcp","pinky_pip","pinky_dip","pinky_tip"
    };

    for (int i = 0; i < 21; ++i) {
        const int idx = i;
        ok &= writeAEFile(handLeftDir  + "/all/" + std::string(handLMNames[i]) + ".txt",
                          "left_"  + std::string(handLMNames[i]), info, frames,
                          [idx](const FrameData& f) -> Point2D {
                              if (!f.leftHand.detected || (int)f.leftHand.landmarks.size() <= idx)
                                  return {};
                              return f.leftHand.landmarks[idx];
                          });
        ok &= writeAEFile(handRightDir + "/all/" + std::string(handLMNames[i]) + ".txt",
                          "right_" + std::string(handLMNames[i]), info, frames,
                          [idx](const FrameData& f) -> Point2D {
                              if (!f.rightHand.detected || (int)f.rightHand.landmarks.size() <= idx)
                                  return {};
                              return f.rightHand.landmarks[idx];
                          });
    }

    std::cout << "[Exporter] AE Keyframe Data 저장 완료: " << outDir << "\n";
    return ok;
}

} // namespace tracker
