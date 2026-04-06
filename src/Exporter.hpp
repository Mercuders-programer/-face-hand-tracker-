#pragma once
#include "TrackingData.hpp"
#include <string>
#include <vector>

namespace tracker {

class Exporter {
public:
    // JSON 내보내기
    // outPath 예: "output/tracking_data.json"
    static bool exportJSON(const std::vector<FrameData>& frames,
                           const VideoInfo&               info,
                           const std::string&             outPath);

    // After Effects Keyframe Data (.txt) 내보내기
    // outDir 예: "output/ae_keyframes"
    // 각 랜드마크별로 별도 .txt 파일 생성:
    //   face/right_pupil.txt, face/nose_tip.txt, ...
    //   hands/left/index_tip.txt, ...
    static bool exportAEKeyframes(const std::vector<FrameData>& frames,
                                  const VideoInfo&               info,
                                  const std::string&             outDir);

private:
    // 단일 랜드마크 AE keyframe .txt 작성
    static bool writeAEFile(const std::string&              path,
                            const std::string&              layerName,
                            const VideoInfo&                info,
                            const std::vector<FrameData>&   frames,
                            std::function<Point2D(const FrameData&)> pointGetter);
};

} // namespace tracker
