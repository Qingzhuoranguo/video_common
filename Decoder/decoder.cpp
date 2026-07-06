#include "decoder.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <gst/gl/gl.h>
#include <gst/gl/x11/gstgldisplay_x11.h>

#include <X11/Xlib.h>
#include <GL/glx.h>

#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <cstdio>
#include <chrono>

namespace Video {

namespace {

void ensureGstInit() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        int argc = 0;
        gst_init(&argc, nullptr);
    });
}

bool isGpuDecoderFactoryName(const std::string& name) {
    return name.find("nv")    != std::string::npos ||
           name.find("vaapi") != std::string::npos ||
           name.find("cuda")  != std::string::npos;
}

} // namespace

struct Decoder::Impl_ {
    std::string     url;
    GLContextHandle glCtxHandle;
    DecoderConfig   config;

    GstGLDisplay* glDisplay = nullptr;
    GstGLContext* glContext = nullptr;

    Display* xDisplay    = nullptr;
    bool ownsXDisplay    = false;

    GstElement* pipeline = nullptr;
    GstElement* appsink  = nullptr;
    GMainLoop*  loop     = nullptr;
    std::thread loopThread;
    guint       busWatchId = 0;

    std::mutex            frameMutex;
    GstSample*            pendingSample = nullptr;
    GstSample*            currentSample = nullptr;
    Frame                 publicFrame;

    std::atomic<bool>     running{false};
    std::atomic<bool>     shouldRun{false};  // 外部意图：是否应该保持运行

    // 看门狗：记录最后一次收到新帧的时间
    std::chrono::steady_clock::time_point lastFrameTime;
    std::mutex                            lastFrameMutex;

    // 重试管理线程
    std::thread retryThread;

    ~Impl_() {
        shouldRun = false;
        stopInternal();
        if (retryThread.joinable()) retryThread.join();
        if (glContext) gst_object_unref(glContext);
        if (glDisplay) gst_object_unref(glDisplay);
        if (xDisplay && ownsXDisplay) XCloseDisplay(xDisplay);
    }

    // ---------- GL context ----------

    bool setupGLContext() {
        GLXContext ctx = static_cast<GLXContext>(glCtxHandle.context);
        if (!ctx) {
            g_printerr("Decoder: GLContextHandle.context 为空\n");
            return false;
        }

        xDisplay = glXGetCurrentDisplay();
        ownsXDisplay = false;
        if (!xDisplay) {
            xDisplay = XOpenDisplay(nullptr);
            ownsXDisplay = true;
        }
        if (!xDisplay) {
            g_printerr("Decoder: 无法获取 X11 Display\n");
            return false;
        }

        if (glContext) { gst_object_unref(glContext); glContext = nullptr; }
        if (glDisplay) { gst_object_unref(glDisplay); glDisplay = nullptr; }

        glDisplay = GST_GL_DISPLAY(gst_gl_display_x11_new_with_display(xDisplay));
        if (!glDisplay) {
            g_printerr("Decoder: failed to create GstGLDisplayX11\n");
            return false;
        }

        glContext = gst_gl_context_new_wrapped(
            glDisplay,
            reinterpret_cast<guintptr>(ctx),
            GST_GL_PLATFORM_GLX,
            static_cast<GstGLAPI>(GST_GL_API_OPENGL3 | GST_GL_API_OPENGL));
        if (!glContext) {
            g_printerr("Decoder: failed to wrap GLXContext\n");
            return false;
        }

        GError* err = nullptr;
        if (!gst_gl_context_activate(glContext, TRUE)) {
            g_printerr("Decoder: gst_gl_context_activate failed\n");
            return false;
        }
        if (!gst_gl_context_fill_info(glContext, &err)) {
            g_printerr("Decoder: gst_gl_context_fill_info failed: %s\n",
                       err ? err->message : "unknown");
            if (err) g_error_free(err);
            gst_gl_context_activate(glContext, FALSE);
            return false;
        }
        gst_gl_context_activate(glContext, FALSE);
        return true;
    }

    // ---------- bus handlers ----------

    static GstBusSyncReply busSyncHandler(GstBus*, GstMessage* msg, gpointer userData) {
        auto* self = static_cast<Impl_*>(userData);
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_NEED_CONTEXT) {
            const gchar* contextType = nullptr;
            gst_message_parse_context_type(msg, &contextType);

            if (g_strcmp0(contextType, GST_GL_DISPLAY_CONTEXT_TYPE) == 0) {
                GstContext* ctx = gst_context_new(GST_GL_DISPLAY_CONTEXT_TYPE, TRUE);
                gst_context_set_gl_display(ctx, self->glDisplay);
                gst_element_set_context(GST_ELEMENT(GST_MESSAGE_SRC(msg)), ctx);
                gst_context_unref(ctx);
                return GST_BUS_DROP;
            } else if (g_strcmp0(contextType, "gst.gl.app_context") == 0) {
                GstContext* ctx = gst_context_new("gst.gl.app_context", TRUE);
                GstStructure* s = gst_context_writable_structure(ctx);
                gst_structure_set(s, "context", GST_TYPE_GL_CONTEXT, self->glContext, nullptr);
                gst_element_set_context(GST_ELEMENT(GST_MESSAGE_SRC(msg)), ctx);
                gst_context_unref(ctx);
                return GST_BUS_DROP;
            }
        }
        return GST_BUS_PASS;
    }

    static GstFlowReturn onNewSample(GstAppSink* sink, gpointer userData) {
        auto* self = static_cast<Impl_*>(userData);
        GstSample* sample = gst_app_sink_pull_sample(sink);
        if (!sample) return GST_FLOW_ERROR;

        {
            std::lock_guard<std::mutex> lock(self->frameMutex);
            if (self->pendingSample) gst_sample_unref(self->pendingSample);
            self->pendingSample = sample;
        }

        // 更新看门狗时间戳
        {
            std::lock_guard<std::mutex> lock(self->lastFrameMutex);
            self->lastFrameTime = std::chrono::steady_clock::now();
        }

        return GST_FLOW_OK;
    }

    static gboolean onBusMessage(GstBus*, GstMessage* msg, gpointer userData) {
        auto* self = static_cast<Impl_*>(userData);
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError* err = nullptr;
                gchar*  dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);
                g_printerr("Decoder pipeline error: %s (%s)\n",
                           err ? err->message : "?", dbg ? dbg : "");
                if (err) g_error_free(err);
                if (dbg) g_free(dbg);
                self->running = false;  // 触发重试线程检测到异常
                break;
            }
            case GST_MESSAGE_EOS:
                self->running = false;
                break;
            default:
                break;
        }
        return TRUE;
    }

    // ---------- pipeline 构建 ----------

    static const char* transportStr(RtspTransport t) {
        return t == RtspTransport::TCP ? "tcp" : "udp";
    }

    static std::string codecDecodeChain(VideoCodec codec, DecodeMode mode) {
        switch (codec) {
            case VideoCodec::H264:
                return mode == DecodeMode::GPU
                    ? "rtph264depay ! h264parse ! nvh264dec"
                    : "rtph264depay ! h264parse ! avdec_h264";
            case VideoCodec::H265:
                return mode == DecodeMode::GPU
                    ? "rtph265depay ! h265parse ! nvh265dec"
                    : "rtph265depay ! h265parse ! avdec_h265";
            case VideoCodec::MJPEG:
                return mode == DecodeMode::GPU
                    ? "rtpjpegdepay ! nvjpegdec"
                    : "rtpjpegdepay ! jpegdec";
            default:
                return "";
        }
    }

    std::string buildPipelineDesc(RtspTransport transport, DecodeMode mode, VideoCodec codec) const {
        std::string desc = "rtspsrc name=src location=\"" + url +
                           "\" latency=0 protocols=" + transportStr(transport) + " ! ";
        if (codec == VideoCodec::Auto) {
            desc += "decodebin name=dbin ! ";
        } else {
            desc += codecDecodeChain(codec, mode) + " ! ";
        }
        desc += "glupload ! glcolorconvert ! "
                "appsink name=sink caps=\"video/x-raw(memory:GLMemory),format=RGBA\" "
                "sync=false max-buffers=1 drop=true emit-signals=true";
        return desc;
    }

    static gboolean autoplugSelectFilter(GstElement*, GstPad*, GstCaps*,
                                          GstElementFactory* factory, gpointer userData) {
        auto wantGpu = *static_cast<DecodeMode*>(userData) == DecodeMode::GPU;
        const gchar* klass = gst_element_factory_get_klass(factory);
        if (klass && g_strstr_len(klass, -1, "Decoder") && g_strstr_len(klass, -1, "Video")) {
            std::string name = GST_OBJECT_NAME(factory);
            bool isGpu = isGpuDecoderFactoryName(name);
            if (isGpu != wantGpu) return 2; // SKIP
        }
        return 0; // TRY
    }

    bool tryStart(RtspTransport transport, DecodeMode mode, VideoCodec codec) {
        std::string desc = buildPipelineDesc(transport, mode, codec);
        g_print("Decoder: trying pipeline: %s\n", desc.c_str());

        GError* err = nullptr;
        GstElement* pl = gst_parse_launch(desc.c_str(), &err);
        if (!pl) {
            g_printerr("Decoder: gst_parse_launch failed: %s\n", err ? err->message : "?");
            if (err) g_error_free(err);
            return false;
        }
        if (err) { g_error_free(err); err = nullptr; }

        GstElement* sink = gst_bin_get_by_name(GST_BIN(pl), "sink");
        DecodeMode modeHolder = mode;
        GstElement* dbin = gst_bin_get_by_name(GST_BIN(pl), "dbin");
        gulong autoplugHandler = 0;
        if (dbin) {
            autoplugHandler = g_signal_connect(dbin, "autoplug-select",
                                               G_CALLBACK(autoplugSelectFilter), &modeHolder);
        }

        GstBus* bus = gst_element_get_bus(pl);
        gst_bus_set_sync_handler(bus, busSyncHandler, this, nullptr);
        g_signal_connect(sink, "new-sample", G_CALLBACK(onNewSample), this);

        gst_element_set_state(pl, GST_STATE_PLAYING);

        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus, config.timeout * GST_SECOND,
            static_cast<GstMessageType>(GST_MESSAGE_ASYNC_DONE | GST_MESSAGE_ERROR));

        bool ok = msg && GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ASYNC_DONE;
        if (msg) gst_message_unref(msg);
        if (dbin) g_object_unref(dbin);

        if (!ok) {
            if (autoplugHandler && dbin) g_signal_handler_disconnect(dbin, autoplugHandler);
            gst_bus_set_sync_handler(bus, nullptr, nullptr, nullptr);
            gst_object_unref(bus);
            if (sink) g_object_unref(sink);
            gst_element_set_state(pl, GST_STATE_NULL);
            gst_object_unref(pl);
            return false;
        }

        pipeline   = pl;
        appsink    = sink;
        busWatchId = gst_bus_add_watch(bus, onBusMessage, this);
        gst_object_unref(bus);

        loop       = g_main_loop_new(nullptr, FALSE);
        loopThread = std::thread([this]() { g_main_loop_run(loop); });

        running = true;

        // 重置看门狗时间戳
        {
            std::lock_guard<std::mutex> lock(lastFrameMutex);
            lastFrameTime = std::chrono::steady_clock::now();
        }

        return true;
    }

    // 尝试一轮所有组合，成功返回 true
    bool tryAllCombinations() {
        if (!setupGLContext()) return false;

        std::vector<RtspTransport> transports = (config.transport == RtspTransport::Auto)
            ? std::vector<RtspTransport>{RtspTransport::UDP, RtspTransport::TCP}
            : std::vector<RtspTransport>{config.transport};

        // DecodeMode::Auto：GPU 优先，CPU 保底
        std::vector<DecodeMode> modes = (config.decodeMode == DecodeMode::Auto)
            ? std::vector<DecodeMode>{DecodeMode::GPU, DecodeMode::CPU}
            : std::vector<DecodeMode>{config.decodeMode};

        for (auto transport : transports) {
            for (auto mode : modes) {
                if (!shouldRun) return false;
                if (tryStart(transport, mode, config.codec)) return true;
            }
        }
        return false;
    }

    void stopInternal() {
        running = false;
        if (loop) g_main_loop_quit(loop);
        if (loopThread.joinable()) loopThread.join();
        if (loop) { g_main_loop_unref(loop); loop = nullptr; }
        if (busWatchId) { g_source_remove(busWatchId); busWatchId = 0; }
        if (pipeline) gst_element_set_state(pipeline, GST_STATE_NULL);
        if (appsink)  { gst_object_unref(appsink);  appsink  = nullptr; }
        if (pipeline) { gst_object_unref(pipeline); pipeline = nullptr; }

        std::lock_guard<std::mutex> lock(frameMutex);
        if (pendingSample) { gst_sample_unref(pendingSample); pendingSample = nullptr; }
        if (currentSample) { gst_sample_unref(currentSample); currentSample = nullptr; }
        publicFrame = Frame{};
    }

    // 检查看门狗：是否超过 watchdogSeconds 没有新帧
    bool isWatchdogExpired() const {
        if (config.watchdogSeconds == 0) return false;
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(lastFrameMutex));
        auto elapsed = std::chrono::steady_clock::now() - lastFrameTime;
        return elapsed > std::chrono::seconds(config.watchdogSeconds);
    }

    // 重试管理线程：负责监控 running 状态和看门狗，失败时等待 timeout 后重启
    void retryLoop() {
        while (shouldRun) {
            if (!running || isWatchdogExpired()) {
                if (running) {
                    g_printerr("Decoder: watchdog expired, restarting pipeline\n");
                } else {
                    g_printerr("Decoder: pipeline stopped, retrying in %zu seconds\n",
                               config.timeout);
                }

                stopInternal();

                // 等待 timeout 秒再重试（每 200ms 检查一次 shouldRun）
                for (size_t i = 0; i < config.timeout * 5 && shouldRun; ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }

                if (!shouldRun) break;

                g_print("Decoder: retrying %s\n", url.c_str());
                tryAllCombinations();
                // 无论成功与否都继续循环监控
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    bool start(const GLContextHandle& glContextIn) {
        if (shouldRun) return true;
        glCtxHandle = glContextIn;
        shouldRun   = true;

        if (!tryAllCombinations()) {
            g_printerr("Decoder: initial start failed, will retry\n");
        }

        // 启动重试管理线程
        retryThread = std::thread([this]() { retryLoop(); });
        return true;  // 始终返回 true，重试线程会持续尝试
    }

    void stop() {
        shouldRun = false;
        stopInternal();
        if (retryThread.joinable()) retryThread.join();
    }

    // ---------- getFrame / hasFrame ----------

    const Frame* getFrame() {
        std::lock_guard<std::mutex> lock(frameMutex);
        if (pendingSample) {
            if (currentSample) gst_sample_unref(currentSample);
            currentSample = pendingSample;
            pendingSample = nullptr;
        }
        if (!currentSample) return nullptr;

        GstBuffer* buffer = gst_sample_get_buffer(currentSample);
        if (!buffer) return nullptr;

        GstMemory* mem = gst_buffer_peek_memory(buffer, 0);
        if (!mem || !gst_is_gl_memory(mem)) return nullptr;
        GstGLMemory* glMem = reinterpret_cast<GstGLMemory*>(mem);

        publicFrame.texture = gst_gl_memory_get_texture_id(glMem);

        GstCaps* caps = gst_sample_get_caps(currentSample);
        GstVideoInfo info;
        if (caps && gst_video_info_from_caps(&info, caps)) {
            publicFrame.width  = static_cast<uint32_t>(GST_VIDEO_INFO_WIDTH(&info));
            publicFrame.height = static_cast<uint32_t>(GST_VIDEO_INFO_HEIGHT(&info));
        }

        return &publicFrame;
    }

    bool hasFrame() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(frameMutex));
        return pendingSample != nullptr || currentSample != nullptr;
    }
};

// ---------------- Decoder ----------------

Decoder::Decoder(const std::string& url) : m_impl(new Impl_()) {
    ensureGstInit();
    m_impl->url = url;
}

Decoder::~Decoder() {
    delete m_impl;
}

void Decoder::setConfig(const DecoderConfig& config) {
    m_impl->config = config;
}

bool Decoder::start(const GLContextHandle& glContext) {
    return m_impl->start(glContext);
}

void Decoder::stop() {
    m_impl->stop();
}

void Decoder::reset() {
    m_impl->stop();
}

bool Decoder::isRunning() const {
    return m_impl->running;
}

bool Decoder::hasFrame() const {
    return m_impl->hasFrame();
}

const Frame* Decoder::getFrame() const {
    return m_impl->getFrame();
}

} // namespace Video