/* The mask lives on the server; this client serializes edits and renders its state. */
window.inpaintAnythingEditor = {
    clientId: crypto.randomUUID(),
    generation: 0,
    epoch: null,
    sequence: 0,
    queue: Promise.resolve(),
    pending: new Map(),
    ready: false,
    hasSam: false,
    failed: false,

    newImage() {
        this.epoch = this.nextEpoch();
        this.ready = false;
        this.failed = false;
        this.hasSam = false;
        this.clearLayers();
        this.updateMode();
        this.updateRecoveryButton();
        return this.epoch;
    },

    nextEpoch() {
        return `${this.clientId}:${++this.generation}`;
    },

    clearLayers() {
        for (const layer of [this.maskLayer, this.samLayer]) {
            if (layer) layer.getContext("2d").clearRect(0, 0, layer.width, layer.height);
        }
    },

    async wait() {
        const epoch = this.epoch;
        await this.queue;
        if (epoch !== this.epoch || !this.ready || this.failed) throw new Error("The mask editor is not ready");
    },

    enqueue(action) {
        const epoch = this.epoch;
        if (!this.ready || this.failed || !epoch) return;
        const job = this.queue.then(() => {
            if (epoch !== this.epoch || this.failed) return;
            return this.send({ ...action, epoch, seq: ++this.sequence });
        });
        this.queue = job.catch((error) => {
            if (epoch === this.epoch) {
                this.failed = true;
                this.updateRecoveryButton();
                this.setStatus(`Mask update failed: ${error.message}. Use Reload mask.`);
            }
        });
    },

    send(command) {
        const root = gradioApp();
        const input = root.querySelector("#ia_editor_request textarea");
        const element = root.querySelector("#ia_editor_submit");
        const button = element?.matches("button") ? element : element?.querySelector("button");
        if (!input || !button) return Promise.reject(new Error("editor controls unavailable"));
        Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set.call(input, JSON.stringify(command));
        input.dispatchEvent(new Event("input", { bubbles: true }));
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                this.pending.delete(command.seq);
                reject(new Error("timed out"));
            }, 30000);
            this.pending.set(command.seq, {
                resolve: () => { clearTimeout(timer); resolve(); },
                reject: (error) => { clearTimeout(timer); reject(error); },
            });
            button.click();
        });
    },

    async receive(raw) {
        let state;
        try { state = JSON.parse(raw); } catch (_) { return; }
        const waiter = state.seq == null ? null : this.pending.get(state.seq);
        if (state.seq != null && !waiter) return;
        if (waiter) this.pending.delete(state.seq);
        if (state.epoch !== this.epoch || state.stale) {
            waiter?.resolve();
            return;
        }
        if (state.error) {
            waiter?.reject(new Error(state.error));
            return;
        }
        try {
            if (Object.hasOwn(state, "colors")) await this.drawColors(state.colors, state.epoch);
            await this.drawMask(state.mask, state.epoch);
            if (state.epoch !== this.epoch) { waiter?.resolve(); return; }
            this.hasSam = state.has_sam;
            if (state.seq == null) { this.ready = !!state.mask; this.failed = false; this.updateRecoveryButton(); }
            this.updateMode();
            this.setStatus("");
            waiter?.resolve();
        } catch (error) {
            if (waiter) waiter.reject(error);
            else { this.failed = true; this.updateRecoveryButton(); this.setStatus(`Mask display failed: ${error.message}`); }
        }
    },

    setStatus(message) {
        const root = gradioApp().querySelector("#ia_editor_status");
        if (root) root.textContent = message;
    },

    updateRecoveryButton() {
        const element = gradioApp().querySelector("#ia_editor_reload");
        const button = element?.matches("button") ? element : element?.querySelector("button");
        if (button) button.disabled = !this.failed;
    },

    async image(base64) {
        if (!base64) return null;
        const image = new Image();
        image.src = `data:image/png;base64,${base64}`;
        await image.decode();
        return image;
    },

    sizeLayer(layer, width, height) {
        if (layer.width !== width) layer.width = width;
        if (layer.height !== height) layer.height = height;
    },

    async drawMask(base64, epoch) {
        const image = await this.image(base64);
        if (!image || !this.maskLayer || epoch !== this.epoch) return;
        const layer = this.maskLayer;
        this.sizeLayer(layer, image.width, image.height);
        const ctx = layer.getContext("2d");
        ctx.clearRect(0, 0, layer.width, layer.height);
        ctx.drawImage(image, 0, 0);
        const pixels = ctx.getImageData(0, 0, layer.width, layer.height);
        for (let i = 0; i < pixels.data.length; i += 4) {
            const selected = pixels.data[i] !== 0;
            pixels.data[i] = 255; pixels.data[i + 1] = 0; pixels.data[i + 2] = 0;
            pixels.data[i + 3] = selected ? 128 : 0;
        }
        ctx.putImageData(pixels, 0, 0);
    },

    async drawColors(base64, epoch) {
        const image = await this.image(base64);
        if (!this.samLayer || epoch !== this.epoch) return;
        const layer = this.samLayer;
        if (image) this.sizeLayer(layer, image.width, image.height);
        const ctx = layer.getContext("2d");
        ctx.clearRect(0, 0, layer.width, layer.height);
        if (image) {
            ctx.globalAlpha = 0.5;
            ctx.drawImage(image, 0, 0);
            ctx.globalAlpha = 1;
        }
    },

    updateMode() {
        const regionOption = gradioApp().querySelector('#ia_editor_mode input[value="Region selection"]');
        if (regionOption) regionOption.disabled = !this.hasSam;
        const selected = gradioApp().querySelector("#ia_editor_mode input:checked");
        const label = selected?.closest("label")?.textContent?.trim() || "Region selection";
        this.mode = label === "Add mask" ? "add" : label === "Remove mask" ? "delete" : "select";
        if (this.samLayer) this.samLayer.style.display = this.mode === "select" && this.hasSam ? "block" : "none";
    },
};

function inpaintAnything_receiveEditorState(value) {
    window.inpaintAnythingEditor.receive(value);
    return [];
}

async function inpaintAnything_finishEditorSam(value) {
    await window.inpaintAnythingEditor.receive(value);
    inpaintAnything_setForgeCanvasProgress("ia_sam_image", false);
    return [];
}

function inpaintAnything_startEditorRecovery() {
    const editor = window.inpaintAnythingEditor;
    editor.epoch = editor.nextEpoch();
    editor.ready = false;
    return [editor.epoch];
}

async function inpaintAnything_startEditorPadding(...args) {
    const editor = window.inpaintAnythingEditor;
    await editor.wait();
    const epoch = editor.newImage();
    const root = gradioApp().querySelector("#ia_sam_image");
    root?.querySelector('[id^="centerButton_"]')?.click();
    const drawing = root?.querySelector(".forge-drawing-canvas");
    if (drawing) drawing.getContext("2d").clearRect(0, 0, drawing.width, drawing.height);
    return [...args.slice(0, 6), epoch];
}

async function inpaintAnything_waitForEditor(count, ...args) {
    await window.inpaintAnythingEditor.wait();
    return args.slice(0, count);
}

onUiLoaded(() => {
    const app = gradioApp();
    const root = app.querySelector("#ia_sam_image");
    const drawing = root?.querySelector(".forge-drawing-canvas");
    if (!drawing) return;
    const editor = window.inpaintAnythingEditor;
    const container = drawing.parentElement;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", "0"); svg.setAttribute("height", "0");
    svg.innerHTML = `<filter id="ia-mask-outline" x="-20%" y="-20%" width="140%" height="140%" color-interpolation-filters="sRGB">
      <feComponentTransfer in="SourceAlpha" result="solid"><feFuncA type="linear" slope="255"/></feComponentTransfer>
      <feMorphology in="solid" operator="dilate" radius="1" result="inner"/>
      <feMorphology in="solid" operator="dilate" radius="2" result="outer"/>
      <feComposite in="outer" in2="inner" operator="out" result="outerRing"/>
      <feComposite in="inner" in2="solid" operator="out" result="innerRing"/>
      <feFlood flood-color="black" result="black"/><feComposite in="black" in2="outerRing" operator="in" result="blackEdge"/>
      <feFlood flood-color="white" result="white"/><feComposite in="white" in2="innerRing" operator="in" result="whiteEdge"/>
      <feMerge><feMergeNode in="blackEdge"/><feMergeNode in="whiteEdge"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>`;
    container.appendChild(svg);
    editor.samLayer = document.createElement("canvas");
    editor.maskLayer = document.createElement("canvas");
    for (const layer of [editor.samLayer, editor.maskLayer]) {
        layer.style.position = "absolute";
        layer.style.pointerEvents = "none";
        container.insertBefore(layer, drawing);
    }
    editor.maskLayer.style.filter = "url(#ia-mask-outline)";
    function syncGeometry() {
        for (const layer of [editor.samLayer, editor.maskLayer]) {
            for (const key of ["width", "height", "left", "top"]) layer.style[key] = drawing.style[key];
        }
    }
    new MutationObserver(syncGeometry).observe(drawing, { attributes: true, attributeFilter: ["style"] });
    syncGeometry();
    editor.updateMode();
    editor.updateRecoveryButton();
    app.querySelector("#ia_editor_mode")?.addEventListener("change", () => editor.updateMode());
    for (const id of ["undoButton_", "redoButton_", "resetButton_", "removeButton_"]) {
        const button = root.querySelector(`[id^="${id}"]`);
        if (button) button.style.display = "none";
    }
    document.addEventListener("keydown", (event) => {
        const target = event.target;
        const typing = target.nodeName === "TEXTAREA" || target.isContentEditable ||
            (target.nodeName === "INPUT" && !["radio", "checkbox", "range", "button"].includes(target.type));
        if (!root.matches(":hover") || !event.ctrlKey || typing) return;
        const key = event.key.toLowerCase();
        if (!["z", "y", "x"].includes(key)) return;
        event.preventDefault(); event.stopImmediatePropagation();
        if (key === "z") editor.enqueue({ action: "undo" });
    }, true);
    const button = (id) => {
        const element = app.querySelector(`#${id}`);
        return element?.matches("button") ? element : element?.querySelector("button");
    };
    button("ia_editor_undo")?.addEventListener("click", () => editor.enqueue({ action: "undo" }));
    button("ia_editor_invert")?.addEventListener("click", () => editor.enqueue({ action: "invert" }));
    button("ia_editor_clear")?.addEventListener("click", () => editor.enqueue({ action: "clear" }));
    button("expand_mask_btn")?.addEventListener("click", () => {
        const slider = app.querySelector("#expand_mask_iteration_count input[type=range]");
        editor.enqueue({ action: "expand", iterations: Number(slider?.value || 1) });
    });

    let stroke = null;
    const position = (event) => {
        const box = drawing.getBoundingClientRect();
        return [Math.max(0, Math.min(drawing.width - 1, Math.floor((event.clientX - box.left) * drawing.width / box.width))),
                Math.max(0, Math.min(drawing.height - 1, Math.floor((event.clientY - box.top) * drawing.height / box.height)))];
    };
    drawing.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault(); event.stopImmediatePropagation();
        if (!editor.ready || editor.failed || !drawing.width || (editor.mode === "select" && !editor.hasSam)) return;
        const widthInput = root.querySelector('[id^="scribbleWidth_"]');
        const scale = drawing.getBoundingClientRect().width / drawing.width;
        const indicator = root.querySelector('[id^="scribbleIndicator_"]');
        const indicatorWidth = parseFloat(indicator?.style.width);
        const width = Math.max(1, Math.round(Number.isFinite(indicatorWidth) ? indicatorWidth / scale : Number(widthInput?.value || 2) * 4));
        stroke = { points: [position(event)], width, action: editor.mode, epoch: editor.epoch };
        drawing.setPointerCapture(event.pointerId);
    }, true);
    drawing.addEventListener("pointermove", (event) => {
        if (!stroke) return;
        event.stopImmediatePropagation();
        stroke.points.push(position(event));
        const ctx = drawing.getContext("2d");
        ctx.clearRect(0, 0, drawing.width, drawing.height);
        ctx.strokeStyle = "white"; ctx.lineWidth = stroke.width; ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.beginPath(); ctx.moveTo(...stroke.points[0]);
        for (const point of stroke.points.slice(1)) ctx.lineTo(...point);
        ctx.stroke();
    }, true);
    function endStroke(event, submit) {
        if (!stroke) return;
        event.stopImmediatePropagation();
        drawing.getContext("2d").clearRect(0, 0, drawing.width, drawing.height);
        if (submit && stroke.epoch === editor.epoch) {
            stroke.points.push(position(event));
            const ignore = app.querySelector("#ignore_black_chk input[type=checkbox]");
            editor.enqueue({ action: stroke.action, points: stroke.points, width: stroke.width, ignore_black: !!ignore?.checked });
        }
        stroke = null;
    }
    drawing.addEventListener("pointerup", (event) => endStroke(event, true), true);
    drawing.addEventListener("pointercancel", (event) => endStroke(event, false), true);
});
