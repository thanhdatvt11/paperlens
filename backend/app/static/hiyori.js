"use strict";

(async function () {
  const panel = document.getElementById("narrator");
  const avatar = panel?.querySelector(".narrator-avatar");
  const robot = avatar?.querySelector("svg");

  if (!panel || !avatar || !robot) return;

  // Giữ robot làm phương án dự phòng nếu model không tải được.
  const host = document.createElement("div");
host.style.cssText =
  "width:100%;height:420px;overflow:hidden;display:none;";

  const status = document.createElement("p");
  status.className = "muted";
  status.textContent = "Đang tải Hiyori…";

  avatar.prepend(host);
  avatar.append(status);

  let app = null;
  let model = null;
  let observer = null;

  try {
    if (!window.Live2DCubismCore) {
      throw new Error("Chưa tải được Cubism Core.");
    }

    if (!window.PIXI?.live2d?.Live2DModel) {
      throw new Error("Chưa tải được PixiJS hoặc cubism4.");
    }

    app = new PIXI.Application({
      width: 180,
      height: 280,
      backgroundAlpha: 0,
      antialias: true,
      autoStart: false,
      resolution: Math.min(window.devicePixelRatio || 1, 2),resolution: Math.min(
  Math.max(window.devicePixelRatio || 1, 2),
  3
),
      autoDensity: true,
    });

    app.ticker.maxFPS = 30;

    app.view.style.display = "block";
    app.view.style.width = "100%";
    app.view.style.height = "100%";
    app.view.setAttribute("role", "img");
    app.view.setAttribute("aria-label", "Nhân vật Hiyori 2D");

    host.append(app.view);

    model = await PIXI.live2d.Live2DModel.from(
      "/harness-assets/models/hiyori/hiyori_free_t08.model3.json",
      {
        autoInteract: false,
        autoUpdate: false,
      }
    );

    app.stage.addChild(model);

    // Hiển thị toàn bộ nhân vật trong khung.
    model.anchor.set(0.5, 0.18);

    const originalWidth = model.internalModel.width;
    const originalHeight = model.internalModel.height;

    if (!(originalWidth > 0 && originalHeight > 0)) {
      throw new Error("Kích thước model không hợp lệ.");
    }

    const core = model.internalModel.coreModel;
    const mouthId = "ParamMouthOpenY";

    const hasMouth =
      typeof core.getParameterIndex === "function" &&
      core.getParameterIndex(mouthId) >= 0;

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    );

    // Ghi tham số miệng sau khi motion/physics đã cập nhật.
    model.internalModel.on("beforeModelUpdate", function () {
      if (!hasMouth) return;

      const speaking = panel.classList.contains("speaking");
      let opening = 0;

      if (speaking) {
        const time = performance.now() / 1000;

        opening = reducedMotion.matches
          ? 0.3
          : Math.min(
              0.85,
              0.12 +
                Math.abs(Math.sin(time * 12)) * 0.48 +
                Math.abs(Math.sin(time * 19)) * 0.2
            );
      }

      core.setParameterValueById(mouthId, opening);
    });

    app.ticker.add(function () {
      model.update(app.ticker.deltaMS);
    });

    function resizeAndSync() {
      const rect = host.getBoundingClientRect();

      const visible =
        !document.hidden && rect.width > 0 && rect.height > 0;

      if (!visible) {
        app.stop();
        return;
      }

      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));

      app.renderer.resize(width, height);

// Phóng to để thấy rõ mặt và thân trên.
// Phần chân nằm ngoài khung là chủ ý.
const zoom = 2.0;
const scale = (height / originalHeight) * zoom;

model.scale.set(scale);
model.position.set(width / 2, height * 0.28);

      app.start();
    }

    // Chỉ ẩn robot khi Hiyori đã tải thành công.
    robot.style.display = "none";
    host.style.display = "block";

    status.textContent = hasMouth
      ? "Hiyori · chuyển động miệng minh họa khi đọc"
      : "Hiyori đã tải, chưa tìm thấy tham số miệng.";

    observer = new ResizeObserver(resizeAndSync);
    observer.observe(host);

    document.addEventListener(
      "visibilitychange",
      resizeAndSync
    );

    resizeAndSync();
  } catch (error) {
    console.error("Không tải được Hiyori:", error);

    observer?.disconnect();
    app?.destroy(true, { children: true });

    host.remove();
    robot.style.display = "";

    status.textContent =
      "Chưa tải được Hiyori: " +
      error.message +
      " Robot vẫn dùng được.";
  }
})();