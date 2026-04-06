/**
 * import_to_ae.jsx
 * After Effects 스크립트 — tracking_data.json 을 읽어
 * Null Object 레이어 + Position 키프레임을 자동으로 생성합니다.
 *
 * 사용법:
 *   AE 메뉴 > File > Scripts > Run Script File... > 이 파일 선택
 *   이후 대화상자 안내에 따라 JSON 파일 및 컴포지션 선택
 */

(function() {
    "use strict";

    // ──────────────────────────────────────────────────────────────
    // 유틸리티
    // ──────────────────────────────────────────────────────────────
    function readFile(f) {
        if (!f.open("r")) throw new Error("파일을 열 수 없습니다: " + f.fsName);
        var content = f.read();
        f.close();
        return content;
    }

    // AE용 간단한 JSON 파서 (ExtendScript는 ES3 기반 — JSON 내장 없음)
    function parseJSON(str) {
        try {
            // ExtendScript에서 eval로 JSON 파싱 (신뢰된 로컬 파일이므로 안전)
            return eval("(" + str + ")");
        } catch (e) {
            throw new Error("JSON 파싱 오류: " + e.message);
        }
    }

    // 컴포지션 목록에서 선택 또는 새로 생성
    function selectOrCreateComp(meta) {
        var items = app.project.items;
        var comps = [];
        for (var i = 1; i <= items.length; i++) {
            if (items[i] instanceof CompItem) comps.push(items[i]);
        }

        var choices = ["[새 컴포지션 생성]"];
        for (var j = 0; j < comps.length; j++) choices.push(comps[j].name);

        var sel = choices.length > 1
            ? parseInt(prompt(
                "데이터를 적용할 컴포지션을 선택하세요 (번호 입력):\n" +
                choices.map(function(c, i) { return i + ": " + c; }).join("\n"),
                "0"))
            : 0;

        if (isNaN(sel) || sel < 0 || sel >= choices.length) sel = 0;

        if (sel === 0) {
            // 새 컴포지션 생성
            var duration = meta.total_frames / meta.fps;
            return app.project.items.addComp(
                "PoseTracker_" + new Date().getTime(),
                meta.width, meta.height,
                1.0, duration, meta.fps
            );
        }
        return comps[sel - 1];
    }

    // ──────────────────────────────────────────────────────────────
    // 단일 Null 레이어 생성 + 키프레임 적용
    // ──────────────────────────────────────────────────────────────
    function createNullWithKeyframes(comp, layerName, fps, keyframes) {
        if (keyframes.length === 0) return null;

        var duration = comp.duration;
        var nullLayer = comp.layers.addNull(duration);
        nullLayer.name = layerName;

        var posProp = nullLayer.property("Transform").property("Position");
        posProp.setInterpolationTypeAtKey; // 확인용

        for (var i = 0; i < keyframes.length; i++) {
            var kf  = keyframes[i];
            var t   = kf.frame / fps;
            if (t > duration) continue;
            posProp.setValueAtTime(t, [kf.x, kf.y]);
        }

        // 보간 방식 → Linear (부드러운 움직임 원하면 삭제)
        for (var k = 1; k <= posProp.numKeys; k++) {
            posProp.setInterpolationTypeAtKey(k,
                KeyframeInterpolationType.LINEAR,
                KeyframeInterpolationType.LINEAR);
        }

        return nullLayer;
    }

    // ──────────────────────────────────────────────────────────────
    // 추적 데이터 수집 헬퍼
    // ──────────────────────────────────────────────────────────────
    function collectFaceKF(frames, fieldName) {
        var kfs = [];
        for (var i = 0; i < frames.length; i++) {
            var f = frames[i];
            if (!f.face.detected) continue;
            var pt = f.face.named[fieldName];
            if (!pt || pt.confidence < 0.05) continue;
            kfs.push({frame: f.frame, x: pt.x, y: pt.y});
        }
        return kfs;
    }

    function collectHandKF(frames, side, lmIndex) {
        var kfs = [];
        var handKey = side === "left" ? "left_hand" : "right_hand";
        for (var i = 0; i < frames.length; i++) {
            var f = frames[i];
            var hand = f[handKey];
            if (!hand || !hand.detected) continue;
            var lm = hand.landmarks[lmIndex];
            if (!lm || lm.confidence < 0.05) continue;
            kfs.push({frame: f.frame, x: lm.x, y: lm.y});
        }
        return kfs;
    }

    // ──────────────────────────────────────────────────────────────
    // 메인
    // ──────────────────────────────────────────────────────────────
    function main() {
        // 1. JSON 파일 선택
        var jsonFile = File.openDialog("tracking_data.json 파일 선택", "JSON 파일:*.json");
        if (!jsonFile) { alert("취소되었습니다."); return; }

        var raw;
        try {
            raw = readFile(jsonFile);
        } catch(e) { alert(e.message); return; }

        var data;
        try {
            data = parseJSON(raw);
        } catch(e) { alert(e.message); return; }

        var meta   = data.metadata;
        var frames = data.frames;
        var fps    = meta.fps;

        // 2. 컴포지션 선택 / 생성
        var comp;
        try {
            comp = selectOrCreateComp(meta);
        } catch(e) { alert("컴포지션 오류: " + e.message); return; }

        app.beginUndoGroup("PoseTracker Import");

        // 3. 그룹 폴더 생성
        var faceFolder = comp.layers.addNull(comp.duration);
        faceFolder.name = "── FACE ──";
        faceFolder.enabled = false;

        // ── 얼굴 랜드마크 ─────────────────────────────────────────
        var facePoints = [
            {key: "right_eye_outer_corner", label: "Face_R_Eye_Outer"},
            {key: "right_eye_inner_corner", label: "Face_R_Eye_Inner"},
            {key: "right_pupil",            label: "Face_R_Pupil"},
            {key: "left_eye_inner_corner",  label: "Face_L_Eye_Inner"},
            {key: "left_eye_outer_corner",  label: "Face_L_Eye_Outer"},
            {key: "left_pupil",             label: "Face_L_Pupil"},
            {key: "nose_bridge_top",        label: "Face_Nose_Bridge"},
            {key: "nose_tip",               label: "Face_Nose_Tip"},
            {key: "mouth_right_corner",     label: "Face_Mouth_R"},
            {key: "mouth_upper_center",     label: "Face_Mouth_Top"},
            {key: "mouth_left_corner",      label: "Face_Mouth_L"},
            {key: "mouth_lower_center",     label: "Face_Mouth_Bottom"}
        ];

        for (var i = 0; i < facePoints.length; i++) {
            var fp = facePoints[i];
            createNullWithKeyframes(comp, fp.label, fps,
                collectFaceKF(frames, fp.key));
        }

        // ── 손 랜드마크 ───────────────────────────────────────────
        var handFolder = comp.layers.addNull(comp.duration);
        handFolder.name = "── HANDS ──";
        handFolder.enabled = false;

        var handNames = [
            "wrist",
            "thumb_cmc","thumb_mcp","thumb_ip","thumb_tip",
            "index_mcp","index_pip","index_dip","index_tip",
            "middle_mcp","middle_pip","middle_dip","middle_tip",
            "ring_mcp","ring_pip","ring_dip","ring_tip",
            "pinky_mcp","pinky_pip","pinky_dip","pinky_tip"
        ];

        for (var j = 0; j < handNames.length; j++) {
            createNullWithKeyframes(comp,
                "Hand_L_" + handNames[j], fps,
                collectHandKF(frames, "left", j));
            createNullWithKeyframes(comp,
                "Hand_R_" + handNames[j], fps,
                collectHandKF(frames, "right", j));
        }

        app.endUndoGroup();

        alert(
            "가져오기 완료!\n\n" +
            "생성된 레이어:\n" +
            "  얼굴:   " + facePoints.length + "개\n" +
            "  왼손:   21개\n" +
            "  오른손: 21개\n\n" +
            "팁: Null Object를 원하는 레이어의 부모로 연결하거나\n" +
            "    Expression으로 위치를 참조하세요."
        );
    }

    main();

})();
