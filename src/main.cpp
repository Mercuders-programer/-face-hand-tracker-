#include "Tracker.hpp"
#include "Exporter.hpp"

#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// 사용법 출력
// -----------------------------------------------------------------------
static void printUsage(const char* prog) {
    std::cout <<
        "사용법:\n"
        "  " << prog << " --video <영상경로> [옵션]\n\n"
        "필수:\n"
        "  --video   <path>   입력 영상 파일 (mp4, avi, mov 등)\n\n"
        "선택:\n"
        "  --output  <dir>    결과 저장 폴더 (기본: ./output)\n"
        "  --gpu     <id>     GPU 번호 (기본: 0)\n"
        "  --no-preview       미리보기 창 비활성화\n"
        "  --face-detector <n> 0=OpenPose(기본), 2=RetinaFace(얼굴 클로즈업 권장)\n\n"
        "출력:\n"
        "  <output>/tracking_data.json       전체 추적 데이터\n"
        "  <output>/ae_keyframes/face/        얼굴 랜드마크별 AE 키프레임\n"
        "  <output>/ae_keyframes/hands/left/  왼손 AE 키프레임\n"
        "  <output>/ae_keyframes/hands/right/ 오른손 AE 키프레임\n\n"
        "예시:\n"
        "  " << prog << " --video C:/Videos/actor.mp4 --output C:/Output --gpu 0\n";
}

// -----------------------------------------------------------------------
// 간단한 인수 파서
// -----------------------------------------------------------------------
struct Args {
    std::string videoPath;
    std::string outputDir  = "./output";
    int         gpuId      = 0;
    int         faceMode   = 0;
    bool        showPreview = true;
    bool        valid       = false;
};

static Args parseArgs(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--video" && i + 1 < argc) {
            a.videoPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            a.outputDir = argv[++i];
        } else if (arg == "--gpu" && i + 1 < argc) {
            a.gpuId = std::stoi(argv[++i]);
        } else if (arg == "--face-detector" && i + 1 < argc) {
            a.faceMode = std::stoi(argv[++i]);
        } else if (arg == "--no-preview") {
            a.showPreview = false;
        } else if (arg == "--help" || arg == "-h") {
            return a;
        }
    }
    a.valid = !a.videoPath.empty();
    return a;
}

// -----------------------------------------------------------------------
// 진행 표시 바
// -----------------------------------------------------------------------
static void printProgress(int current, int total) {
    if (total <= 0) return;
    int pct  = static_cast<int>(100.0 * current / total);
    int bars = pct / 2; // 50칸 기준
    std::cout << "\r[";
    for (int i = 0; i < 50; ++i) std::cout << (i < bars ? '#' : '-');
    std::cout << "] " << std::setw(3) << pct << "% ("
              << current << "/" << total << ")" << std::flush;
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    Args args = parseArgs(argc, argv);
    if (!args.valid) {
        std::cerr << "오류: --video 옵션이 필요합니다.\n\n";
        printUsage(argv[0]);
        return 1;
    }

    // 출력 디렉토리 생성
    fs::create_directories(args.outputDir);

    std::cout << "========================================\n"
              << "  PoseTracker — 얼굴 + 손 추적\n"
              << "========================================\n"
              << "  입력: " << args.videoPath << "\n"
              << "  출력: " << args.outputDir << "\n"
              << "  GPU:  " << args.gpuId     << "\n\n";

    // ── 추적 ─────────────────────────────────────────────────────────
    tracker::Tracker tracker;
    tracker.setGpuDevice(args.gpuId);
    tracker.setFaceDetectorMode(args.faceMode);
    tracker.setConfidenceThreshold(0.05f);

    std::vector<tracker::FrameData> frames;
    tracker::VideoInfo              info;

    bool ok = tracker.processVideo(
        args.videoPath, frames, info,
        args.showPreview,
        printProgress
    );
    std::cout << "\n";

    if (!ok || frames.empty()) {
        std::cerr << "오류: 추적 실패\n";
        return 1;
    }

    // ── 내보내기 ─────────────────────────────────────────────────────
    std::cout << "\n[내보내기 시작]\n";

    const std::string jsonPath   = args.outputDir + "/tracking_data.json";
    const std::string aeDir      = args.outputDir + "/ae_keyframes";

    ok  = tracker::Exporter::exportJSON(frames, info, jsonPath);
    ok &= tracker::Exporter::exportAEKeyframes(frames, info, aeDir);

    if (ok) {
        std::cout << "\n========================================\n"
                  << "  완료!\n"
                  << "  JSON : " << jsonPath  << "\n"
                  << "  AE   : " << aeDir     << "/\n"
                  << "========================================\n"
                  << "\nAfter Effects에서 사용하려면:\n"
                  << "  방법 1 (수동): ae_keyframes/ 안의 .txt 파일을\n"
                  << "                 Null Object 선택 후 Edit > Paste\n"
                  << "  방법 2 (자동): scripts/import_to_ae.jsx 를\n"
                  << "                 AE에서 File > Scripts > Run Script File\n";
    } else {
        std::cerr << "내보내기 중 오류 발생\n";
        return 1;
    }

    return 0;
}
