// pose_to_null.jsx — PoseTracker NULL Object Generator
// Usage: File > Scripts > Run Script File... 로 실행
// 키워드를 입력하면 ae_keyframes 폴더에서 해당 txt를 찾아
// NULL 오브젝트를 만들고 Position 키프레임을 자동 적용합니다.
#target aftereffects

(function () {

    // ── 1. 키워드 입력 ───────────────────────────────────────────────
    var keyword = prompt(
        "랜드마크 키워드를 입력하세요\n" +
        "예: right_index_tip / left_shoulder / nose_tip\n" +
        "(person 번호를 앞에 붙이면 해당 사람만: person_0__right_index_tip)",
        ""
    );
    if (!keyword || keyword === "") return;

    // ── 2. ae_keyframes 폴더 선택 ───────────────────────────────────
    var baseFolder = Folder.selectDialog("ae_keyframes 폴더를 선택하세요");
    if (!baseFolder) return;

    // ── 3. 파일 검색 (재귀) ─────────────────────────────────────────
    var target = searchFile(baseFolder, keyword + ".txt");
    if (!target) {
        alert("파일을 찾을 수 없습니다: " + keyword + ".txt\n\n" +
              "폴더: " + baseFolder.fsName);
        return;
    }

    // ── 4. 키프레임 파싱 ────────────────────────────────────────────
    var data = parseKeyframeFile(target);
    if (!data || data.frames.length === 0) {
        alert("유효한 키프레임이 없습니다.\n" + target.fsName);
        return;
    }

    // ── 5. 컴포지션 확인 ────────────────────────────────────────────
    var comp = app.project.activeItem;
    if (!comp || !(comp instanceof CompItem)) {
        alert("컴포지션을 먼저 열거나 선택하세요.");
        return;
    }

    // ── 6. NULL 생성 + 키프레임 적용 ────────────────────────────────
    app.beginUndoGroup("PoseTracker: " + keyword);

    var nullLayer = comp.layers.addNull();
    nullLayer.name = keyword;

    var pos = nullLayer.property("Transform").property("Position");

    for (var i = 0; i < data.frames.length; i++) {
        var f = data.frames[i];
        var t = f.frame / data.fps;
        pos.setValueAtTime(t, [f.x, f.y]);
    }

    app.endUndoGroup();

    alert("완료!\nNULL 오브젝트 '" + keyword + "' 생성\n키프레임 수: " + data.frames.length);

})();


// ── 재귀 파일 검색 ───────────────────────────────────────────────────
function searchFile(folder, filename) {
    var items = folder.getFiles();
    for (var i = 0; i < items.length; i++) {
        if (items[i] instanceof File && items[i].name === filename) {
            return items[i];
        }
        if (items[i] instanceof Folder) {
            var result = searchFile(items[i], filename);
            if (result) return result;
        }
    }
    return null;
}


// ── AE Keyframe Data 파싱 ────────────────────────────────────────────
function parseKeyframeFile(file) {
    file.open("r");
    var content = file.read();
    file.close();

    var lines = content.split(/\r?\n/);
    var fps = 30;
    var frames = [];
    var inData = false;

    for (var i = 0; i < lines.length; i++) {
        var line = lines[i];

        if (line.indexOf("Units Per Second") !== -1) {
            var parts = line.split("\t");
            fps = parseFloat(parts[parts.length - 1]) || 30;
        }
        if (line.indexOf("Frame") !== -1 && line.indexOf("X pixels") !== -1) {
            inData = true;
            continue;
        }
        if (inData && line.indexOf("End of Keyframe") !== -1) break;

        if (inData) {
            var p = line.split("\t");
            if (p.length >= 4) {
                var fr = parseInt(p[1]);
                var x  = parseFloat(p[2]);
                var y  = parseFloat(p[3]);
                if (!isNaN(fr) && !isNaN(x) && !isNaN(y)) {
                    frames.push({ frame: fr, x: x, y: y });
                }
            }
        }
    }
    return { fps: fps, frames: frames };
}
