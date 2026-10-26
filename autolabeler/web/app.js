const $ = (selector) => document.querySelector(selector);
const state = {
  images: [],
  labelStates: {},
  labelStatus: "unannotated",
  selectedImage: "",
  annotations: [],
  selectedAnnotation: -1,
  labelRequestId: 0,
  drawing: null,
  dragging: null,
  addMode: false,
  keypointMode: false,
  moveKeypointMode: false,
  hoveredKeypoint: null,
  shortcuts: {},
  jobId: null,
  jobTimer: null,
  imageButtons: new Map(),
  renderedListKey: null,
  applyingSettings: false,
  undoStack: [],
  redoStack: [],
  dirty: false,
  zoom: 1,
  baseWidth: 0,
  baseHeight: 0,
};

const actionNames = {
  add: "绘制类别 0",
  previous: "上一张图片",
  next: "下一张图片",
  save: "保存标签",
  delete: "删除选中框",
  keypoint: "新建关键点",
  move_keypoint: "移动关键点",
  class_1: "绘制类别 1",
  class_2: "绘制类别 2",
  class_3: "绘制类别 3",
  class_4: "绘制类别 4",
  class_5: "绘制类别 5",
  class_6: "绘制类别 6",
  class_7: "绘制类别 7",
  class_8: "绘制类别 8",
  class_9: "绘制类别 9",
  keypoint_visible: "关键点可见",
  keypoint_occluded: "关键点遮挡",
  keypoint_invisible: "关键点不可见",
};
const naturalCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
const STATUS_NAMES = { unannotated: "未标注", buffer: "缓冲", complete: "完成" };

function setStatus(message, error = false) {
  const status = $("#status");
  status.textContent = message;
  status.style.color = error ? "#ff9292" : "";
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || `请求失败 (${response.status})`);
    error.data = data;
    throw error;
  }
  return data;
}

function queryPath(path) {
  return encodeURIComponent(path || "");
}

async function browse(path = $("#browse-path").value) {
  try {
    const data = await api(`/api/browse?path=${queryPath(path)}`);
    $("#browse-path").value = data.path;
    const folderList = $("#folder-list");
    folderList.replaceChildren();
    if (data.parent !== null) {
      const parent = document.createElement("button");
      parent.className = "folder-entry";
      parent.textContent = "↰ ..";
      parent.addEventListener("click", () => browse(data.parent));
      folderList.append(parent);
    }
    for (const directory of data.directories) {
      const button = document.createElement("button");
      button.className = "folder-entry";
      button.textContent = `📁 ${directory.name}`;
      button.title = directory.path;
      button.addEventListener("click", () => browse(directory.path));
      folderList.append(button);
    }
    await refreshLabelStates();
    renderImageList();
    $("#image-count").textContent = `(${state.images.length})`;
    setStatus(`已打开 ${data.path || "工作区根目录"}`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function refreshLabelStates() {
  const imageDir = $("#image-dir").value.trim();
  const labelDir = $("#label-dir").value.trim();
  if (!imageDir) {
    state.labelStates = {};
    state.images = [];
    return;
  }
  const params = new URLSearchParams({ image_dir: imageDir, label_dir: labelDir });
  const result = await api(`/api/label-states?${params}`);
  state.labelStates = result.states;
  state.images = Object.keys(result.states)
    .sort((left, right) => naturalCollator.compare(left, right))
    .map((path) => ({ name: path.split("/").pop(), path }));
  scheduleSaveSettings();
}

function renderImageList() {
  const list = $("#image-list");
  const key = state.images.map((image) => image.path).join("\n");
  if (state.renderedListKey !== key) {
    state.renderedListKey = key;
    state.imageButtons = new Map();
    list.replaceChildren();
    for (const image of state.images) {
      const button = document.createElement("button");
      button.className = "image-entry";
      button.title = image.path;
      button.addEventListener("click", () => openImage(image.path));
      list.append(button);
      state.imageButtons.set(image.path, button);
    }
  }
  for (const image of state.images) {
    const button = state.imageButtons.get(image.path);
    if (!button) continue;
    const status = imageStatus(image.path);
    button.className = `image-entry${image.path === state.selectedImage ? " active" : ""}`;
    button.textContent = `${image.name} · ${STATUS_NAMES[status] || status}`;
    button.dataset.status = status;
  }
}

function currentImageIndex() {
  return state.images.findIndex((image) => image.path === state.selectedImage);
}

function imageStatus(path) {
  return state.labelStates[path] || "unannotated";
}

const settingSelectors = [
  "#image-dir", "#label-dir", "#format", "#auto-copy-labels", "#auto-save", "#nav-filter", "#auto-count",
  "#train-data", "#train-model", "#epochs", "#imgsz", "#batch", "#train-device",
  "#train-workers", "#train-patience", "#optimizer", "#lr0", "#seed", "#degrees",
  "#fliplr", "#close-mosaic", "#hsv-h", "#hsv-s", "#hsv-v",
  "#pretrained", "#cache", "#amp", "#generate-data", "#aug-crop", "#aug-occlusion",
].join(",");

function setInput(selector, value) {
  const element = $(selector);
  if (!element) return;
  if (element.type === "checkbox") element.checked = Boolean(value);
  else element.value = value;
}

function collectSettings() {
  return {
    image_dir: $("#image-dir").value.trim(),
    label_dir: $("#label-dir").value.trim(),
    format: $("#format").value,
    auto_copy_labels: $("#auto-copy-labels").checked,
    auto_save: $("#auto-save").checked,
    nav_filter: $("#nav-filter").value,
    auto_count: Number($("#auto-count").value),
    data: $("#train-data").value.trim(),
    model: $("#train-model").value.trim(),
    epochs: Number($("#epochs").value),
    imgsz: Number($("#imgsz").value),
    batch: Number($("#batch").value),
    device: $("#train-device").value.trim(),
    workers: Number($("#train-workers").value),
    patience: Number($("#train-patience").value),
    optimizer: $("#optimizer").value,
    lr0: Number($("#lr0").value),
    seed: Number($("#seed").value),
    degrees: Number($("#degrees").value),
    fliplr: Number($("#fliplr").value),
    close_mosaic: Number($("#close-mosaic").value),
    hsv_h: Number($("#hsv-h").value),
    hsv_s: Number($("#hsv-s").value),
    hsv_v: Number($("#hsv-v").value),
    pretrained: $("#pretrained").checked,
    cache: $("#cache").checked,
    amp: $("#amp").checked,
    generate_data: $("#generate-data").checked,
    aug_crop: $("#aug-crop").checked,
    aug_occlusion: $("#aug-occlusion").checked,
    selected_image: state.selectedImage,
    label_states: { ...state.labelStates },
  };
}

function applySettings(settings) {
  if (!settings) return;
  state.applyingSettings = true;
  try {
    if (settings.label_states && typeof settings.label_states === "object") {
      state.labelStates = { ...settings.label_states };
    }
    state.selectedImage = typeof settings.selected_image === "string" ? settings.selected_image : "";
    setInput("#image-dir", settings.image_dir ?? "");
    setInput("#label-dir", settings.label_dir ?? ".autolabeler/labels");
    setInput("#format", settings.format ?? "yolo");
    setInput("#auto-copy-labels", settings.auto_copy_labels ?? false);
    setInput("#auto-save", settings.auto_save ?? false);
    setInput("#nav-filter", settings.nav_filter ?? "none");
    setInput("#auto-count", settings.auto_count ?? 10);
    setInput("#train-data", settings.data ?? "");
    setInput("#train-model", settings.model ?? "");
    setInput("#epochs", settings.epochs ?? 100);
    setInput("#imgsz", settings.imgsz ?? 640);
    setInput("#batch", settings.batch ?? 16);
    setInput("#train-device", settings.device ?? "cpu");
    setInput("#train-workers", settings.workers ?? 8);
    setInput("#train-patience", settings.patience ?? 100);
    setInput("#optimizer", settings.optimizer ?? "auto");
    setInput("#lr0", settings.lr0 ?? 0.01);
    setInput("#seed", settings.seed ?? 0);
    setInput("#degrees", settings.degrees ?? 0);
    setInput("#fliplr", settings.fliplr ?? 0.5);
    setInput("#close-mosaic", settings.close_mosaic ?? 10);
    setInput("#hsv-h", settings.hsv_h ?? 0.015);
    setInput("#hsv-s", settings.hsv_s ?? 0.7);
    setInput("#hsv-v", settings.hsv_v ?? 0.4);
    setInput("#pretrained", settings.pretrained ?? true);
    setInput("#cache", settings.cache ?? false);
    setInput("#amp", settings.amp ?? true);
    setInput("#generate-data", settings.generate_data ?? false);
    setInput("#aug-crop", settings.aug_crop ?? true);
    setInput("#aug-occlusion", settings.aug_occlusion ?? true);
    $("#train-data").disabled = $("#generate-data").checked;
    syncTrainBaseModel();
  } finally {
    state.applyingSettings = false;
  }
}

let settingsTimer = null;
function scheduleSaveSettings() {
  if (state.applyingSettings) return;
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(persistSettings, 400);
}

async function persistSettings() {
  try {
    const settings = collectSettings();
    await api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings }),
    });
  } catch (error) {
    setStatus(`设置保存失败：${error.message}`, true);
  }
}

async function saveCurrentIfNeeded() {
  if (!state.selectedImage || !$("#auto-save").checked || !state.dirty) return true;
  return await saveLabels();
}

async function openImage(path, copySource = null) {
  if (state.selectedImage && path !== state.selectedImage) {
    if (!(await saveCurrentIfNeeded())) return;
  }
  state.selectedImage = path;
  scheduleSaveSettings();
  const requestedFormat = $("#format").value;
  const requestId = ++state.labelRequestId;
  state.annotations = [];
  state.selectedAnnotation = -1;
  state.addMode = false;
  state.keypointMode = false;
  state.moveKeypointMode = false;
  state.hoveredKeypoint = null;
  state.undoStack = [];
  state.redoStack = [];
  state.zoom = 1;
  state.baseWidth = 0;
  state.baseHeight = 0;
  state.labelStatus = "unannotated";
  $("#review-button").disabled = true;
  $("#save-button").disabled = false;
  $("#image").src = `/api/image?path=${queryPath(path)}&t=${Date.now()}`;
  $("#canvas-wrap").style.display = "block";
  $("#empty-state").style.display = "none";
  $("#current-image").textContent = path;
  renderImageList();
  try {
    const params = new URLSearchParams({
      image: path,
      image_dir: $("#image-dir").value.trim(),
      label_dir: $("#label-dir").value.trim(),
    });
    const result = await api(`/api/labels?${params}`);
    if (requestId !== state.labelRequestId || state.selectedImage !== path || requestedFormat !== $("#format").value) return;
    state.labelStatus = result.status;
    state.annotations = parseLabels(result.content, requestedFormat);
    state.dirty = false;
    $("#review-button").disabled = result.status !== "buffer" || result.pending;
    $("#save-button").disabled = result.pending;
    $("#label-path").textContent = `${result.status === "buffer" ? "缓冲标签" : "标签文件"}：${result.status === "buffer" ? result.buffer_path : result.label_path}`;
    if (result.status === "unannotated" && copySource && $("#auto-copy-labels").checked && copySource.format === requestedFormat) {
      state.annotations = cloneAnnotations(copySource.annotations);
      if (state.annotations.length) {
        await persistAnnotations(path, state.annotations, requestedFormat);
        if (requestId !== state.labelRequestId || state.selectedImage !== path) return;
        state.labelStatus = "complete";
        state.labelStates[path] = "complete";
        renderImageList();
        setStatus("目标图片无标签，已复制并保存上一张的标注");
      } else {
        setStatus("该图片暂无标签");
      }
    } else {
      state.labelStates[path] = result.status;
      renderImageList();
      setStatus(result.pending ? "自动标注进行中" : result.status === "buffer" ? "自动标注结果待审核" : result.status === "complete" ? "已从本地刷新标签" : "该图片暂无标签");
    }
    draw();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function cloneAnnotations(annotations) {
  return annotations.map((annotation) => ({
    ...annotation,
    points: annotation.points?.map((point) => [...point]),
    keypoints: annotation.keypoints?.map((point) => point && [...point]),
  }));
}

function serializeAnnotations(annotations, format) {
  return annotations.map((annotation) => {
    if (format === "yolo-obb") {
      return `${annotation.cls} ${annotation.points.flat().map((value) => value.toFixed(6)).join(" ")}`;
    }
    const box = `${annotation.cls} ${annotation.x.toFixed(6)} ${annotation.y.toFixed(6)} ${annotation.w.toFixed(6)} ${annotation.h.toFixed(6)}`;
    if (format !== "yolo-pose") return box;
    const keypoints = annotation.keypoints.map((point) =>
      point ? `${point[0].toFixed(6)} ${point[1].toFixed(6)} ${point[2]}` : "0.000000 0.000000 0"
    ).join(" ");
    return `${box} ${keypoints}`;
  }).join("\n");
}

async function persistAnnotations(imagePath, annotations, format) {
  const content = serializeAnnotations(annotations, format);
  const result = await api("/api/labels", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image: imagePath,
      image_dir: $("#image-dir").value.trim(),
      label_dir: $("#label-dir").value.trim(),
      format,
      content: content ? `${content}\n` : "",
    }),
  });
  if (state.selectedImage === imagePath && $("#format").value === format) {
    $("#label-path").textContent = `标签文件：${result.label_path}`;
  }
  return result;
}

function parseLabels(content, format) {
  return content.split(/\r?\n/).filter(Boolean).flatMap((line) => {
    const values = line.trim().split(/\s+/).map(Number);
    if (values.some((value) => !Number.isFinite(value))) return [];
    const cls = Math.max(0, Math.trunc(values[0]));
    if (format === "yolo-obb") {
      if (values.length !== 9) return [];
      return [{ cls, points: Array.from({ length: 4 }, (_, i) => [values[1 + i * 2], values[2 + i * 2]]) }];
    }
    if (values.length < 5) return [];
    const box = { cls, x: values[1], y: values[2], w: values[3], h: values[4] };
    if (format !== "yolo-pose") return [box];
    if (values.length < 8 || (values.length - 5) % 3 !== 0) return [];
    const keypoints = Array.from({ length: (values.length - 5) / 3 }, (_, i) => [
      values[5 + i * 3],
      values[6 + i * 3],
      values[7 + i * 3],
    ]);
    $("#keypoint-count").value = String(keypoints.length);
    return [{ ...box, keypoints }];
  });
}

function draw() {
  const image = $("#image");
  const canvas = $("#overlay");
  if (!image.naturalWidth || !image.clientWidth) return;
  const width = image.clientWidth;
  const height = image.clientHeight;
  const scaleX = width / image.naturalWidth;
  const scaleY = height / image.naturalHeight;
  canvas.width = Math.round(width * devicePixelRatio);
  canvas.height = Math.round(height * devicePixelRatio);
  const context = canvas.getContext("2d");
  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  state.annotations.forEach((annotation, index) => {
    context.strokeStyle = index === state.selectedAnnotation ? "#ffbe42" : "#39e58c";
    context.lineWidth = 2;
    context.fillStyle = "#39e58c";
    context.font = "13px sans-serif";
    if (annotation.points && $("#format").value === "yolo-obb") {
      const points = annotation.points.map(([x, y]) => ({
        x: x * image.naturalWidth * scaleX,
        y: y * image.naturalHeight * scaleY,
      }));
      context.beginPath();
      points.forEach((point, pointIndex) => {
        if (pointIndex === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.closePath();
      context.stroke();
      context.fillText(String(annotation.cls), points[0].x + 3, Math.max(14, points[0].y - 4));
      return;
    }
    const x = (annotation.x - annotation.w / 2) * image.naturalWidth * scaleX;
    const y = (annotation.y - annotation.h / 2) * image.naturalHeight * scaleY;
    const w = annotation.w * image.naturalWidth * scaleX;
    const h = annotation.h * image.naturalHeight * scaleY;
    context.strokeRect(x, y, w, h);
    context.fillText(String(annotation.cls), x + 3, Math.max(14, y - 4));
    if ($("#format").value === "yolo-pose" && annotation.keypoints) {
      annotation.keypoints.forEach((point, pointIndex) => {
        if (!point || point[2] === 0) return;
        const pointX = point[0] * image.naturalWidth * scaleX;
        const pointY = point[1] * image.naturalHeight * scaleY;
        const hovered = state.hoveredKeypoint
          && state.hoveredKeypoint.annotationIndex === index
          && state.hoveredKeypoint.pointIndex === pointIndex;
        context.beginPath();
        context.arc(pointX, pointY, hovered ? 6 : 4, 0, Math.PI * 2);
        context.fillStyle = hovered ? "#ff4d4d" : (point[2] === 1 ? "#ffd166" : "#5cc8ff");
        context.fill();
        context.fillStyle = "#fff";
        context.fillText(String(pointIndex + 1), pointX + 5, pointY - 5);
      });
    }
  });
  if (state.selectedAnnotation >= 0) {
    drawEditHandles(context, state.annotations[state.selectedAnnotation], image);
  }
  if (state.drawing) {
    context.strokeStyle = "#4aa3ff";
    context.lineWidth = 2;
    context.setLineDash([6, 4]);
    if (state.drawing.kind === "box") {
      const { start, end } = state.drawing;
      context.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y);
    } else if (state.drawing.kind === "obb-base") {
      context.beginPath();
      context.moveTo(state.drawing.start.x, state.drawing.start.y);
      context.lineTo(state.drawing.end.x, state.drawing.end.y);
      context.stroke();
    } else if (state.drawing.kind === "obb-height") {
      const points = obbScreenPoints(state.drawing.start, state.drawing.end, state.drawing.height);
      context.beginPath();
      points.forEach((point, pointIndex) => {
        if (pointIndex === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.closePath();
      context.stroke();
    }
    context.setLineDash([]);
  }
}

function drawEditHandles(context, annotation, image) {
  context.fillStyle = "#ffffff";
  context.strokeStyle = "#3478f6";
  context.lineWidth = 1.5;
  if (annotation.points && $("#format").value === "yolo-obb") {
    for (const [x, y] of annotation.points) {
      context.beginPath();
      context.arc(x * image.clientWidth, y * image.clientHeight, 5, 0, Math.PI * 2);
      context.fill();
      context.stroke();
    }
    return;
  }
  const left = (annotation.x - annotation.w / 2) * image.clientWidth;
  const right = (annotation.x + annotation.w / 2) * image.clientWidth;
  const top = (annotation.y - annotation.h / 2) * image.clientHeight;
  const bottom = (annotation.y + annotation.h / 2) * image.clientHeight;
  for (const handle of boxHandles(left, top, right, bottom)) {
    context.fillRect(handle.x - 4, handle.y - 4, 8, 8);
    context.strokeRect(handle.x - 4, handle.y - 4, 8, 8);
  }
}

function boxHandles(left, top, right, bottom) {
  const middleX = (left + right) / 2;
  const middleY = (top + bottom) / 2;
  return [
    { name: "nw", x: left, y: top },
    { name: "n", x: middleX, y: top },
    { name: "ne", x: right, y: top },
    { name: "e", x: right, y: middleY },
    { name: "se", x: right, y: bottom },
    { name: "s", x: middleX, y: bottom },
    { name: "sw", x: left, y: bottom },
    { name: "w", x: left, y: middleY },
  ];
}

function nearestHandle(position, handles, radius = 10) {
  let nearest = null;
  let nearestDistance = radius;
  for (const handle of handles) {
    const distance = Math.hypot(position.x - handle.x, position.y - handle.y);
    if (distance <= nearestDistance) {
      nearest = handle;
      nearestDistance = distance;
    }
  }
  return nearest;
}

function nearestKeypoint(position) {
  const image = $("#image");
  if (!image.clientWidth) return null;
  const radius = 12;
  let best = null;
  let bestDistance = radius;
  state.annotations.forEach((annotation, annotationIndex) => {
    if (!annotation.keypoints) return;
    annotation.keypoints.forEach((point, pointIndex) => {
      if (!point || point[2] === 0) return;
      const x = point[0] * image.clientWidth;
      const y = point[1] * image.clientHeight;
      const distance = Math.hypot(position.x - x, position.y - y);
      if (distance <= bestDistance) {
        bestDistance = distance;
        best = { annotationIndex, pointIndex };
      }
    });
  });
  return best;
}

function beginAnnotationDrag(index, position, pointerId) {
  const annotation = state.annotations[index];
  const image = $("#image");
  const original = cloneAnnotations([annotation])[0];
  let drag;
  if (annotation.points) {
    const vertices = annotation.points.map(([x, y], pointIndex) => ({
      name: String(pointIndex),
      x: x * image.clientWidth,
      y: y * image.clientHeight,
    }));
    const vertex = nearestHandle(position, vertices, 10);
    drag = vertex
      ? { kind: "obb-point", pointIndex: Number(vertex.name) }
      : { kind: "obb-move" };
  } else {
    const keypoint = annotation.keypoints
      ? nearestHandle(
        position,
        annotation.keypoints.flatMap((point, pointIndex) => point && point[2] !== 0
          ? [{ name: String(pointIndex), x: point[0] * image.clientWidth, y: point[1] * image.clientHeight }]
          : []),
        10,
      )
      : null;
    if (keypoint) {
      drag = { kind: "keypoint", pointIndex: Number(keypoint.name) };
    } else {
      const left = (annotation.x - annotation.w / 2) * image.clientWidth;
      const right = (annotation.x + annotation.w / 2) * image.clientWidth;
      const top = (annotation.y - annotation.h / 2) * image.clientHeight;
      const bottom = (annotation.y + annotation.h / 2) * image.clientHeight;
      const handle = nearestHandle(position, boxHandles(left, top, right, bottom));
      drag = handle ? { kind: "box-resize", handle: handle.name } : { kind: "box-move" };
    }
  }
  state.dragging = { ...drag, index, start: position, original, moved: false };
  $("#overlay").setPointerCapture(pointerId);
}

function updateAnnotationDrag(position) {
  const drag = state.dragging;
  if (!drag) return;
  drag.moved = true;
  const image = $("#image");
  const annotation = state.annotations[drag.index];
  const original = drag.original;
  const dx = (position.x - drag.start.x) / image.clientWidth;
  const dy = (position.y - drag.start.y) / image.clientHeight;
  if (drag.kind === "keypoint") {
    const point = original.keypoints[drag.pointIndex];
    const left = original.x - original.w / 2;
    const right = original.x + original.w / 2;
    const top = original.y - original.h / 2;
    const bottom = original.y + original.h / 2;
    annotation.keypoints[drag.pointIndex] = [
      Math.max(left, Math.min(right, point[0] + dx)),
      Math.max(top, Math.min(bottom, point[1] + dy)),
      point[2],
    ];
    draw();
    return;
  }
  if (drag.kind === "obb-point") {
    const point = original.points[drag.pointIndex];
    annotation.points[drag.pointIndex] = [
      Math.max(0, Math.min(1, point[0] + dx)),
      Math.max(0, Math.min(1, point[1] + dy)),
    ];
    draw();
    return;
  }
  if (drag.kind === "obb-move") {
    const minX = Math.min(...original.points.map((point) => point[0]));
    const maxX = Math.max(...original.points.map((point) => point[0]));
    const minY = Math.min(...original.points.map((point) => point[1]));
    const maxY = Math.max(...original.points.map((point) => point[1]));
    const moveX = Math.max(-minX, Math.min(1 - maxX, dx));
    const moveY = Math.max(-minY, Math.min(1 - maxY, dy));
    annotation.points = original.points.map(([x, y]) => [x + moveX, y + moveY]);
    draw();
    return;
  }

  const originalBounds = {
    left: original.x - original.w / 2,
    right: original.x + original.w / 2,
    top: original.y - original.h / 2,
    bottom: original.y + original.h / 2,
  };
  const bounds = { ...originalBounds };
  if (drag.kind === "box-move") {
    const moveX = Math.max(-bounds.left, Math.min(1 - bounds.right, dx));
    const moveY = Math.max(-bounds.top, Math.min(1 - bounds.bottom, dy));
    bounds.left += moveX;
    bounds.right += moveX;
    bounds.top += moveY;
    bounds.bottom += moveY;
  } else {
    const sides = drag.handle;
    if (sides.includes("w")) bounds.left = Math.max(0, Math.min(bounds.right - 0.002, bounds.left + dx));
    if (sides.includes("e")) bounds.right = Math.min(1, Math.max(bounds.left + 0.002, bounds.right + dx));
    if (sides.includes("n")) bounds.top = Math.max(0, Math.min(bounds.bottom - 0.002, bounds.top + dy));
    if (sides.includes("s")) bounds.bottom = Math.min(1, Math.max(bounds.top + 0.002, bounds.bottom + dy));
  }
  annotation.x = (bounds.left + bounds.right) / 2;
  annotation.y = (bounds.top + bounds.bottom) / 2;
  annotation.w = bounds.right - bounds.left;
  annotation.h = bounds.bottom - bounds.top;
  if (annotation.keypoints) {
    annotation.keypoints = original.keypoints.map((point) => {
      if (!point) return null;
      return [
        bounds.left + ((point[0] - originalBounds.left) / original.w) * annotation.w,
        bounds.top + ((point[1] - originalBounds.top) / original.h) * annotation.h,
        point[2],
      ];
    });
  }
  draw();
}

function updatePanDrag(event) {
  const drag = state.dragging;
  if (!drag || drag.kind !== "pan") return;
  const container = document.querySelector(".canvas-scroll");
  const dx = event.clientX - drag.startX;
  const dy = event.clientY - drag.startY;
  if (Math.abs(dx) > 0 || Math.abs(dy) > 0) drag.moved = true;
  container.scrollLeft = drag.startScrollLeft - dx;
  container.scrollTop = drag.startScrollTop - dy;
}

function updateCursor() {
  const canvas = $("#overlay");
  if (state.dragging?.kind === "pan") canvas.style.cursor = "grabbing";
  else if (state.addMode || state.keypointMode) canvas.style.cursor = "crosshair";
  else if (state.moveKeypointMode) canvas.style.cursor = state.hoveredKeypoint ? "grab" : "crosshair";
  else canvas.style.cursor = "grab";
}

function obbScreenPoints(start, end, heightPoint) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  if (!heightPoint || length === 0) return [start, end];
  const nx = -dy / length;
  const ny = dx / length;
  const height = (heightPoint.x - start.x) * nx + (heightPoint.y - start.y) * ny;
  const offset = { x: nx * height, y: ny * height };
  return [
    start,
    end,
    { x: end.x + offset.x, y: end.y + offset.y },
    { x: start.x + offset.x, y: start.y + offset.y },
  ];
}

function pointerPosition(event) {
  const rect = $("#overlay").getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function fitImage() {
  const image = $("#image");
  const container = document.querySelector(".canvas-scroll");
  if (!image.naturalWidth || !container) return;
  const padding = 36;
  const availableWidth = Math.max(50, container.clientWidth - padding);
  const availableHeight = Math.max(50, container.clientHeight - padding);
  const scale = Math.min(1, availableWidth / image.naturalWidth, availableHeight / image.naturalHeight);
  state.baseWidth = image.naturalWidth * scale;
  state.baseHeight = image.naturalHeight * scale;
  state.zoom = 1;
  applyZoom();
}

function applyZoom() {
  const image = $("#image");
  if (!image.naturalWidth || !state.baseWidth) return;
  image.style.maxWidth = "none";
  image.style.maxHeight = "none";
  image.style.width = `${state.baseWidth * state.zoom}px`;
  image.style.height = `${state.baseHeight * state.zoom}px`;
  draw();
}

function startDrawing(event) {
  if (!state.selectedImage) return;
  if (event.button !== 0) return;
  const position = pointerPosition(event);
  const format = $("#format").value;
  if (state.keypointMode && format === "yolo-pose") {
    addPoseKeypoint(position);
    return;
  }
  if (state.drawing?.kind === "obb-height" && format === "yolo-obb") {
    finishObb(obbScreenPoints(state.drawing.start, state.drawing.end, position));
    return;
  }
  if (state.moveKeypointMode) {
    const target = nearestKeypoint(position);
    if (!target) return;
    state.selectedAnnotation = target.annotationIndex;
    const annotation = state.annotations[target.annotationIndex];
    state.dragging = {
      kind: "keypoint",
      pointIndex: target.pointIndex,
      index: target.annotationIndex,
      start: position,
      original: cloneAnnotations([annotation])[0],
      moved: false,
    };
    $("#overlay").setPointerCapture(event.pointerId);
    updateCursor();
    draw();
    return;
  }
  const image = $("#image");
  const clickedIndex = state.annotations.findIndex((annotation) => {
    if (annotation.points && format === "yolo-obb") {
      const points = annotation.points.map(([x, y]) => ({
        x: x * image.clientWidth,
        y: y * image.clientHeight,
      }));
      return nearestHandle(position, points, 10) !== null || isInsidePolygon(position, points);
    }
    const left = (annotation.x - annotation.w / 2) * image.clientWidth;
    const right = (annotation.x + annotation.w / 2) * image.clientWidth;
    const top = (annotation.y - annotation.h / 2) * image.clientHeight;
    const bottom = (annotation.y + annotation.h / 2) * image.clientHeight;
    const handlePadding = 5;
    return position.x >= left - handlePadding && position.x <= right + handlePadding
      && position.y >= top - handlePadding && position.y <= bottom + handlePadding;
  });
  if (clickedIndex >= 0 && !state.addMode) {
    state.selectedAnnotation = clickedIndex;
    if (format === "yolo-pose") $("#class-id").value = String(state.annotations[clickedIndex].cls);
    beginAnnotationDrag(clickedIndex, position, event.pointerId);
    draw();
    return;
  }
  if (!state.addMode) {
    const container = document.querySelector(".canvas-scroll");
    state.dragging = {
      kind: "pan",
      startX: event.clientX,
      startY: event.clientY,
      startScrollLeft: container.scrollLeft,
      startScrollTop: container.scrollTop,
      moved: false,
    };
    $("#overlay").setPointerCapture(event.pointerId);
    updateCursor();
    return;
  }
  state.drawing = {
    kind: format === "yolo-obb" ? "obb-base" : "box",
    start: position,
    end: position,
  };
  $("#overlay").setPointerCapture(event.pointerId);
  draw();
}

function isInsidePolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i];
    const b = polygon[j];
    if ((a.y > point.y) !== (b.y > point.y)
      && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

function finishDrawing() {
  if (!state.drawing) return;
  if (state.drawing.kind === "obb-base") {
    state.drawing = { ...state.drawing, kind: "obb-height", height: state.drawing.end };
    setStatus("基准边已确定；移动鼠标预览旋转框，再单击确定高度");
    draw();
    return;
  }
  if (state.drawing.kind !== "box") return;
  const image = $("#image");
  const scaleX = image.naturalWidth / image.clientWidth;
  const scaleY = image.naturalHeight / image.clientHeight;
  const x1 = Math.min(state.drawing.start.x, state.drawing.end.x) * scaleX;
  const y1 = Math.min(state.drawing.start.y, state.drawing.end.y) * scaleY;
  const x2 = Math.max(state.drawing.start.x, state.drawing.end.x) * scaleX;
  const y2 = Math.max(state.drawing.start.y, state.drawing.end.y) * scaleY;
  state.drawing = null;
  if (x2 - x1 < 2 || y2 - y1 < 2) return draw();
  const annotation = {
    cls: Math.max(0, Math.trunc(Number($("#class-id").value) || 0)),
    x: ((x1 + x2) / 2) / image.naturalWidth,
    y: ((y1 + y2) / 2) / image.naturalHeight,
    w: (x2 - x1) / image.naturalWidth,
    h: (y2 - y1) / image.naturalHeight,
  };
  if ($("#format").value === "yolo-pose") {
    const count = Math.max(1, Math.min(100, Math.trunc(Number($("#keypoint-count").value) || 17)));
    annotation.keypoints = Array.from({ length: count }, () => null);
  }
  state.annotations.push(annotation);
  state.selectedAnnotation = state.annotations.length - 1;
  pushUndo({ kind: "add", index: state.selectedAnnotation, annotation: cloneAnnotations([annotation])[0] });
  state.dirty = true;
  state.addMode = false;
  draw();
}

function finishObb(points) {
  if (points.length !== 4 || Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y) < 2
    || Math.hypot(points[3].x - points[0].x, points[3].y - points[0].y) < 2) {
    state.drawing = null;
    state.addMode = false;
    draw();
    return setStatus("矩形边长太短，请重新添加", true);
  }
  const image = $("#image");
  const normalized = points.map((point) => [
    Math.min(1, Math.max(0, point.x / image.clientWidth)),
    Math.min(1, Math.max(0, point.y / image.clientHeight)),
  ]);
  const obbAnnotation = {
    cls: Math.max(0, Math.trunc(Number($("#class-id").value) || 0)),
    points: normalized,
  };
  state.annotations.push(obbAnnotation);
  state.selectedAnnotation = state.annotations.length - 1;
  pushUndo({ kind: "add", index: state.selectedAnnotation, annotation: cloneAnnotations([obbAnnotation])[0] });
  state.dirty = true;
  state.drawing = null;
  state.addMode = false;
  draw();
}

function addPoseKeypoint(position) {
  const annotation = state.annotations[state.selectedAnnotation];
  if (!annotation || !annotation.keypoints) {
    return setStatus("请先添加或选中一个 Pose 检测框", true);
  }
  const keypointIndex = annotation.keypoints.findIndex((point) => point === null);
  if (keypointIndex < 0) return setStatus("该检测框的关键点已经全部填写");
  const image = $("#image");
  const x = position.x / image.clientWidth;
  const y = position.y / image.clientHeight;
  if (x < annotation.x - annotation.w / 2 || x > annotation.x + annotation.w / 2
    || y < annotation.y - annotation.h / 2 || y > annotation.y + annotation.h / 2) {
    return setStatus("关键点需位于当前检测框内", true);
  }
  const index = state.selectedAnnotation;
  const before = cloneAnnotations([annotation])[0];
  annotation.keypoints[keypointIndex] = [
    x,
    y,
    Number($("#keypoint-visibility").value),
  ];
  const after = cloneAnnotations([annotation])[0];
  pushUndo({ kind: "modify", index, before, after });
  state.dirty = true;
  setStatus(`已添加第 ${keypointIndex + 1} 个关键点`);
  draw();
}

function addAnnotation() {
  if (!state.selectedImage) return setStatus("请先选择图片", true);
  state.addMode = true;
  state.keypointMode = false;
  state.drawing = null;
  updateCursor();
  if ($("#format").value === "yolo-obb") {
    setStatus("按住鼠标左键拖出 OBB 基准边，松开后移动鼠标并单击确定矩形高度");
  } else {
    setStatus("在图片上按住鼠标左键并拖动绘制矩形框");
  }
}

function startDrawingClass(classId) {
  $("#class-id").value = String(classId);
  addAnnotation();
}

function setKeypointVisibility(value) {
  $("#keypoint-visibility").value = String(value);
  setStatus(value === 2 ? "关键点可见性：可见" : value === 1 ? "关键点可见性：遮挡" : "关键点可见性：不可见");
}

async function saveLabels() {
  if (!state.selectedImage) return setStatus("请先选择图片", true);
  const format = $("#format").value;
  try {
    await persistAnnotations(state.selectedImage, state.annotations, format);
    state.dirty = false;
    state.labelStatus = "complete";
    state.labelStates[state.selectedImage] = "complete";
    $("#review-button").disabled = true;
    renderImageList();
    scheduleSaveSettings();
    setStatus("标签已保存并记录到手动标注清单");
    return true;
  } catch (error) {
    setStatus(error.message, true);
    return false;
  }
}

async function approveBufferedLabels() {
  if (!state.selectedImage || state.labelStatus !== "buffer") return;
  try {
    const content = serializeAnnotations(state.annotations, $("#format").value);
    const result = await api("/api/labels/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: state.selectedImage,
        image_dir: $("#image-dir").value.trim(),
        label_dir: $("#label-dir").value.trim(),
        format: $("#format").value,
        content: content ? `${content}\n` : "",
      }),
    });
    state.dirty = false;
    state.labelStatus = "complete";
    state.labelStates[state.selectedImage] = "complete";
    $("#label-path").textContent = `标签文件：${result.label_path}`;
    $("#review-button").disabled = true;
    renderImageList();
    scheduleSaveSettings();
    setStatus("已审核通过，标签已移入标签目录");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function navFilterAllows(status) {
  const filter = $("#nav-filter").value;
  if (filter === "none") return true;
  if (filter === "unannotated") return status === "unannotated";
  if (filter === "unannotated_complete") return status === "unannotated" || status === "complete";
  if (filter === "buffer") return status === "buffer";
  if (filter === "complete") return status === "complete";
  return true;
}

function moveImage(offset) {
  const step = Math.sign(offset);
  const index = currentImageIndex();
  let cursor = index < 0 ? (step > 0 ? -1 : state.images.length) : index;
  while (true) {
    cursor += step;
    if (cursor < 0 || cursor >= state.images.length) break;
    if (navFilterAllows(imageStatus(state.images[cursor].path))) break;
  }
  const next = state.images[cursor];
  if (!next) {
    setStatus(step > 0 ? "后面没有更多符合条件的图片" : "前面没有更多符合条件的图片");
    return;
  }
  const copySource = offset === 1
    ? { annotations: cloneAnnotations(state.annotations), format: $("#format").value }
    : null;
  openImage(next.path, copySource);
}

function deleteAnnotation() {
  if (state.selectedAnnotation < 0) return;
  const index = state.selectedAnnotation;
  pushUndo({
    kind: "delete",
    index,
    annotation: cloneAnnotations([state.annotations[index]])[0],
  });
  state.annotations.splice(index, 1);
  state.dirty = true;
  state.selectedAnnotation = -1;
  state.keypointMode = false;
  draw();
}

function pushUndo(entry) {
  state.undoStack.push(entry);
  if (state.undoStack.length > 10) state.undoStack.shift();
  state.redoStack.length = 0;
}

function applyHistoryEntry(entry, direction) {
  if (entry.kind === "add") {
    if (direction === "undo") {
      const index = Math.min(entry.index, state.annotations.length - 1);
      if (index >= 0) state.annotations.splice(index, 1);
      state.selectedAnnotation = -1;
    } else {
      const index = Math.min(entry.index, state.annotations.length);
      state.annotations.splice(index, 0, cloneAnnotations([entry.annotation])[0]);
      state.selectedAnnotation = index;
    }
  } else if (entry.kind === "delete") {
    if (direction === "undo") {
      const index = Math.min(entry.index, state.annotations.length);
      state.annotations.splice(index, 0, cloneAnnotations([entry.annotation])[0]);
      state.selectedAnnotation = index;
    } else {
      const index = Math.min(entry.index, state.annotations.length - 1);
      if (index >= 0) state.annotations.splice(index, 1);
      state.selectedAnnotation = -1;
    }
  } else if (entry.kind === "modify") {
    if (entry.index < state.annotations.length) {
      const value = direction === "undo" ? entry.before : entry.after;
      state.annotations[entry.index] = cloneAnnotations([value])[0];
      state.selectedAnnotation = entry.index;
    }
  }
}

function undo() {
  const entry = state.undoStack.pop();
  if (!entry) return setStatus("没有可撤销的操作");
  applyHistoryEntry(entry, "undo");
  state.dirty = true;
  state.redoStack.push(entry);
  if (state.redoStack.length > 10) state.redoStack.shift();
  state.addMode = false;
  state.keypointMode = false;
  updateCursor();
  draw();
  setStatus("已撤销上一步操作");
}

function redo() {
  const entry = state.redoStack.pop();
  if (!entry) return setStatus("没有可恢复的操作");
  applyHistoryEntry(entry, "redo");
  state.dirty = true;
  state.undoStack.push(entry);
  if (state.undoStack.length > 10) state.undoStack.shift();
  state.addMode = false;
  state.keypointMode = false;
  updateCursor();
  draw();
  setStatus("已恢复上一步撤销");
}

function addKeypointMode() {
  if ($("#format").value !== "yolo-pose") return setStatus("关键点只适用于 YOLO-Pose", true);
  if (state.selectedAnnotation < 0 || !state.annotations[state.selectedAnnotation]?.keypoints) {
    return setStatus("请先添加或选中一个 Pose 检测框", true);
  }
  state.keypointMode = !state.keypointMode;
  state.addMode = false;
  updateCursor();
  setStatus(state.keypointMode ? "按关键点顺序在框内点击；可见性由右侧选项控制" : "已退出关键点添加模式");
}

function toggleMoveKeypointMode() {
  if ($("#format").value !== "yolo-pose") return setStatus("移动关键点只适用于 YOLO-Pose", true);
  if (!state.selectedImage) return setStatus("请先选择图片", true);
  state.moveKeypointMode = !state.moveKeypointMode;
  state.hoveredKeypoint = null;
  if (state.moveKeypointMode) {
    state.addMode = false;
    state.keypointMode = false;
  }
  updateCursor();
  draw();
  setStatus(state.moveKeypointMode ? "移动关键点模式：鼠标靠近关键点变红，拖动可调整位置" : "已退出移动关键点模式");
}

async function loadModels() {
  try {
    const { models } = await api("/api/models");
    const select = $("#model-list");
    select.replaceChildren();
    if (!models.length) select.add(new Option("暂无模型", ""));
    for (const model of models) select.add(new Option(`${model.name} — ${model.source}`, model.path));
    await loadModelInfo();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function enabledKeypoints() {
  const result = {};
  for (const group of document.querySelectorAll("#keypoint-toggles .keypoint-group")) {
    result[group.dataset.cls] = [...group.querySelectorAll("input:checked")].map((input) => Number(input.dataset.index));
  }
  return result;
}

function setAllKeypoints(enabled) {
  for (const input of document.querySelectorAll("#keypoint-toggles input")) {
    input.checked = enabled;
  }
}

async function loadModelInfo() {
  const model = $("#model-list").value;
  const format = $("#format").value;
  const wrap = $("#keypoint-toggles-wrap");
  const container = $("#keypoint-toggles");
  if (!model || format !== "yolo-pose") {
    wrap.style.display = "none";
    return;
  }
  try {
    const result = await api(`/api/model-info?model=${encodeURIComponent(model)}`);
    if (result.task !== "pose" || !result.keypoints) {
      wrap.style.display = "none";
      return;
    }
    const classes = result.names && result.names.length ? result.names : [String(0)];
    container.replaceChildren();
    classes.forEach((name, classIndex) => {
      const group = document.createElement("div");
      group.className = "keypoint-group";
      group.dataset.cls = String(classIndex);

      const title = document.createElement("div");
      title.className = "keypoint-group-title";
      title.textContent = `类别 ${classIndex} · ${name}`;
      group.append(title);

      const grid = document.createElement("div");
      grid.className = "keypoint-toggles";
      for (let index = 0; index < result.keypoints; index += 1) {
        const label = document.createElement("label");
        label.className = "keypoint-toggle";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.dataset.index = String(index);
        const text = document.createElement("span");
        text.textContent = String(index + 1);
        label.append(checkbox, text);
        grid.append(label);
      }
      group.append(grid);
      container.append(group);
    });
    wrap.style.display = "block";
  } catch (error) {
    wrap.style.display = "none";
    setStatus(error.message, true);
  }
}

async function handleModelUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const result = await api(`/api/models/upload?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    setStatus(`模型已复制到 ${result.path}`);
    await loadModels();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    event.target.value = "";
  }
}

async function loadBaseModels() {
  try {
    const { models } = await api("/api/base-models");
    const select = $("#train-base-model");
    select.replaceChildren(new Option("默认 yolo11n.pt（自动下载）", ""));
    for (const model of models) select.add(new Option(model.name, `base:${model.name}`));
  } catch (error) {
    setStatus(error.message, true);
  }
}

function syncTrainBaseModel() {
  const select = $("#train-base-model");
  const value = $("#train-model").value.trim();
  if (value.startsWith("base:") && [...select.options].some((option) => option.value === value)) {
    select.value = value;
  } else {
    select.value = "";
  }
}

async function handleTrainModelUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const result = await api(`/api/models/upload?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    $("#train-model").value = result.path;
    syncTrainBaseModel();
    scheduleSaveSettings();
    setStatus(`训练模型已复制到 ${result.path}`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    event.target.value = "";
  }
}

async function autoConfigureTraining() {
  const imageDir = $("#image-dir").value.trim();
  const labelDir = $("#label-dir").value.trim();
  if (!imageDir) return setStatus("请先设置图片目录", true);
  if (!labelDir) return setStatus("请先设置标签目录", true);
  try {
    const params = new URLSearchParams({
      image_dir: imageDir,
      label_dir: labelDir,
      format: $("#format").value,
    });
    const result = await api(`/api/train-params?${params}`);
    const p = result.parameters;
    if (p.epochs !== undefined) $("#epochs").value = p.epochs;
    if (p.batch !== undefined) $("#batch").value = p.batch;
    if (p.imgsz !== undefined) $("#imgsz").value = p.imgsz;
    if (p.workers !== undefined) $("#train-workers").value = p.workers;
    if (p.patience !== undefined) $("#train-patience").value = p.patience;
    if (p.close_mosaic !== undefined) $("#close-mosaic").value = p.close_mosaic;
    scheduleSaveSettings();
    setStatus(`已扫描 ${result.count} 张参与训练的图片，训练参数已更新（epochs=${p.epochs}，batch=${p.batch}）`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function submitTrain() {
  persistSettings();
  const parameters = {
    epochs: Number($("#epochs").value),
    imgsz: Number($("#imgsz").value),
    batch: Number($("#batch").value),
    data: $("#train-data").value.trim(),
    model: $("#train-model").value.trim(),
    device: $("#train-device").value.trim(),
    workers: Number($("#train-workers").value),
    patience: Number($("#train-patience").value),
    optimizer: $("#optimizer").value,
    lr0: Number($("#lr0").value),
    seed: Number($("#seed").value),
    degrees: Number($("#degrees").value),
    fliplr: Number($("#fliplr").value),
    close_mosaic: Number($("#close-mosaic").value),
    hsv_h: Number($("#hsv-h").value),
    hsv_s: Number($("#hsv-s").value),
    hsv_v: Number($("#hsv-v").value),
    pretrained: $("#pretrained").checked,
    cache: $("#cache").checked,
    amp: $("#amp").checked,
    aug_crop: $("#aug-crop").checked,
    aug_occlusion: $("#aug-occlusion").checked,
    image_dir: $("#image-dir").value.trim(),
    label_dir: $("#label-dir").value.trim(),
    generate_data: $("#generate-data").checked,
  };
  try {
    const result = await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parameters, format: $("#format").value }),
    });
    watchJob(result.id);
    setStatus(`训练任务已启动：${result.run}`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function submitAutoLabel() {
  persistSettings();
  try {
    const body = {
      count: Number($("#auto-count").value),
      model: $("#model-list").value,
      image_dir: $("#image-dir").value.trim(),
      label_dir: $("#label-dir").value.trim(),
      format: $("#format").value,
    };
    if ($("#format").value === "yolo-pose" && $("#keypoint-toggles-wrap").style.display !== "none") {
      body.enabled_keypoints = enabledKeypoints();
    }
    const result = await api("/api/auto-label", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    watchJob(result.id);
    for (const image of result.images) state.labelStates[image] = "buffer";
    renderImageList();
    await refreshLabelStates();
    renderImageList();
    setStatus(`自动标注已启动：${result.count} 张图片`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

function watchJob(jobId) {
  state.jobId = jobId;
  $("#cancel-job-button").disabled = false;
  if (state.jobTimer) clearInterval(state.jobTimer);
  pollJob();
  state.jobTimer = setInterval(pollJob, 1200);
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const job = await api(`/api/jobs?id=${encodeURIComponent(state.jobId)}`);
    const progress = job.progress || {};
    const position = job.kind === "train"
      ? (progress.epoch ? `Epoch ${progress.epoch}/${progress.epochs}` : "")
      : (progress.completed ? `${progress.completed}/${progress.total} · ${progress.image}` : "");
    $("#job-status").textContent = `${job.kind === "train" ? "训练" : "自动标注"}：${job.status}${position ? ` · ${position}` : ""}${job.error ? ` · ${job.error}` : ""}`;
    if (["completed", "failed", "cancelled"].includes(job.status)) {
      clearInterval(state.jobTimer);
      state.jobTimer = null;
      state.jobId = null;
      $("#cancel-job-button").disabled = true;
      let message = job.error || `任务${job.status === "completed" ? "完成" : job.status === "cancelled" ? "已停止" : "失败"}`;
      if (job.error && /CUDA/i.test(job.error)) {
        message += "（CUDA 上下文可能已损坏：请重启服务后重试，并可尝试关闭 AMP、减小 batch）";
      }
      setStatus(message, job.status === "failed");
      await loadModels();
      if (job.kind === "auto-label") {
        await refreshLabelStates();
        renderImageList();
      }
      if (job.kind === "auto-label" && state.selectedImage) await openImage(state.selectedImage);
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function cancelJob() {
  if (!state.jobId) return;
  try {
    await api("/api/jobs/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: state.jobId }),
    });
    $("#job-status").textContent = "任务：正在停止";
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function openShortcutSettings() {
  const data = await api("/api/keys");
  state.shortcuts = data.shortcuts;
  const fields = $("#shortcut-fields");
  fields.replaceChildren();
  for (const [action, title] of Object.entries(actionNames)) {
    const label = document.createElement("label");
    label.textContent = title;
    const input = document.createElement("input");
    input.value = state.shortcuts[action] || "";
    input.dataset.action = action;
    input.readOnly = true;
    input.addEventListener("keydown", (event) => {
      event.preventDefault();
      if (event.key === "Escape") {
        input.value = "";
        input.placeholder = "无";
      } else {
        input.value = event.key;
        input.placeholder = "";
      }
    });
    label.append(input);
    fields.append(label);
  }
  $("#settings-dialog").showModal();
}

async function saveShortcutSettings() {
  const shortcuts = Object.fromEntries(
    [...$("#shortcut-fields").querySelectorAll("input")].map((input) => [input.dataset.action, input.value])
  );
  try {
    const result = await api("/api/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ shortcuts }),
    });
    state.shortcuts = result.shortcuts;
    $("#settings-dialog").close();
    setStatus("快捷键已保存到程序目录 key.yaml");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function shortcutAction(key) {
  return Object.entries(state.shortcuts).find(([, value]) => value.toLowerCase() === key.toLowerCase())?.[0];
}

document.addEventListener("keydown", (event) => {
  if ($("#settings-dialog").open) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undo();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "x") {
    event.preventDefault();
    redo();
    return;
  }
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  const action = shortcutAction(event.key);
  if (!action) return;
  event.preventDefault();
  if (action === "add") startDrawingClass(0);
  if (action.startsWith("class_")) startDrawingClass(Number(action.slice(6)));
  if (action === "previous") moveImage(-1);
  if (action === "next") moveImage(1);
  if (action === "save") saveLabels();
  if (action === "delete") deleteAnnotation();
  if (action === "keypoint") addKeypointMode();
  if (action === "move_keypoint") toggleMoveKeypointMode();
  if (action === "keypoint_visible") setKeypointVisibility(2);
  if (action === "keypoint_occluded") setKeypointVisibility(1);
  if (action === "keypoint_invisible") setKeypointVisibility(0);
});

$("#browse-button").addEventListener("click", () => browse());
$("#browse-path").addEventListener("keydown", (event) => { if (event.key === "Enter") browse(); });
$("#use-image-dir").addEventListener("click", () => {
  $("#image-dir").value = $("#browse-path").value.trim();
  browse($("#browse-path").value.trim());
});
$("#use-label-dir").addEventListener("click", () => {
  $("#label-dir").value = $("#browse-path").value.trim();
  refreshLabelStates()
    .then(() => {
      renderImageList();
      if (state.selectedImage) return openImage(state.selectedImage);
    })
    .catch((error) => setStatus(error.message, true));
});
$("#image-dir").addEventListener("change", () => {
  refreshLabelStates().then(renderImageList).catch((error) => setStatus(error.message, true));
});
$("#label-dir").addEventListener("change", () => {
  refreshLabelStates().then(renderImageList).catch((error) => setStatus(error.message, true));
});
$("#auto-copy-labels").checked = localStorage.getItem("autolabeler.autoCopyLabels") === "true";
$("#auto-copy-labels").addEventListener("change", (event) => {
  localStorage.setItem("autolabeler.autoCopyLabels", String(event.target.checked));
  setStatus(event.target.checked ? "已开启下一张无标签时复制上一张标注" : "已关闭自动复制标注");
});
$("#previous-button").addEventListener("click", () => moveImage(-1));
$("#next-button").addEventListener("click", () => moveImage(1));
$("#add-button").addEventListener("click", addAnnotation);
$("#save-button").addEventListener("click", saveLabels);
$("#review-button").addEventListener("click", approveBufferedLabels);
$("#delete-button").addEventListener("click", deleteAnnotation);
$("#add-keypoint-button").addEventListener("click", addKeypointMode);
$("#overlay").addEventListener("pointerdown", startDrawing);
$("#overlay").addEventListener("pointermove", (event) => {
  if (state.dragging) {
    if (state.dragging.kind === "pan") {
      updatePanDrag(event);
      return;
    }
    updateAnnotationDrag(pointerPosition(event));
    return;
  }
  if (state.drawing) {
    const position = pointerPosition(event);
    if (state.drawing.kind === "obb-height") state.drawing.height = position;
    else state.drawing.end = position;
    draw();
  }
  if (state.moveKeypointMode) {
    const position = pointerPosition(event);
    const target = nearestKeypoint(position);
    const previous = state.hoveredKeypoint
      ? `${state.hoveredKeypoint.annotationIndex}:${state.hoveredKeypoint.pointIndex}`
      : null;
    state.hoveredKeypoint = target;
    const current = target ? `${target.annotationIndex}:${target.pointIndex}` : null;
    if (previous !== current) {
      updateCursor();
      draw();
    }
  }
});
$("#overlay").addEventListener("pointerup", (event) => {
  if (state.dragging) {
    if (state.dragging.kind !== "pan" && state.dragging.moved) {
      pushUndo({
        kind: "modify",
        index: state.dragging.index,
        before: cloneAnnotations([state.dragging.original])[0],
        after: cloneAnnotations([state.annotations[state.dragging.index]])[0],
      });
      state.dirty = true;
    }
    state.dragging = null;
    updateCursor();
    draw();
    return;
  }
  finishDrawing();
  updateCursor();
});
$("#overlay").addEventListener("pointercancel", () => {
  state.dragging = null;
  state.drawing = null;
  state.hoveredKeypoint = null;
  updateCursor();
  draw();
});
$("#image").addEventListener("load", fitImage);
window.addEventListener("resize", fitImage);
$("#canvas-wrap").addEventListener("wheel", (event) => {
  if (!state.selectedImage) return;
  event.preventDefault();
  const container = document.querySelector(".canvas-scroll");
  const wrap = $("#canvas-wrap");
  const rect = wrap.getBoundingClientRect();
  const fx = rect.width ? (event.clientX - rect.left) / rect.width : 0.5;
  const fy = rect.height ? (event.clientY - rect.top) / rect.height : 0.5;
  const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
  state.zoom = Math.min(8, Math.max(0.2, state.zoom * factor));
  applyZoom();
  const newRect = wrap.getBoundingClientRect();
  const targetLeft = event.clientX - fx * newRect.width;
  const targetTop = event.clientY - fy * newRect.height;
  container.scrollLeft += newRect.left - targetLeft;
  container.scrollTop += newRect.top - targetTop;
}, { passive: false });
$("#format").addEventListener("change", () => {
  state.addMode = false;
  state.keypointMode = false;
  if (state.selectedImage) openImage(state.selectedImage);
});
$("#model-upload").addEventListener("change", handleModelUpload);
$("#model-list").addEventListener("change", loadModelInfo);
$("#format").addEventListener("change", loadModelInfo);
$("#keypoints-all").addEventListener("click", () => setAllKeypoints(true));
$("#keypoints-none").addEventListener("click", () => setAllKeypoints(false));
$("#train-base-model").addEventListener("change", (event) => {
  if (event.target.value) $("#train-model").value = event.target.value;
  scheduleSaveSettings();
});
$("#train-model-upload").addEventListener("change", handleTrainModelUpload);
$("#generate-data").addEventListener("change", (event) => {
  $("#train-data").disabled = event.target.checked;
  setStatus(event.target.checked ? "将根据当前目录的标签自动生成临时数据集 YAML" : "将使用指定的数据集 YAML");
});
$("#auto-train-params-button").addEventListener("click", autoConfigureTraining);
$("#train-button").addEventListener("click", submitTrain);
$("#auto-button").addEventListener("click", submitAutoLabel);
$("#cancel-job-button").addEventListener("click", cancelJob);
$("#settings-button").addEventListener("click", () => openShortcutSettings().catch((error) => setStatus(error.message, true)));
$("#save-keys-button").addEventListener("click", saveShortcutSettings);

document.addEventListener("change", (event) => {
  if (event.target.matches && event.target.matches(settingSelectors)) scheduleSaveSettings();
});

async function initialize() {
  try {
    const [keysData, settingsData] = await Promise.all([api("/api/keys"), api("/api/settings")]);
    state.shortcuts = keysData.shortcuts;
    await loadBaseModels();
    applySettings(settingsData.settings);
    await browse("");
    const selected = settingsData.settings.selected_image;
    if (selected && state.images.some((image) => image.path === selected)) {
      await openImage(selected);
    }
    await loadModels();
  } catch (error) {
    setStatus(error.message, true);
  }
}

initialize();
