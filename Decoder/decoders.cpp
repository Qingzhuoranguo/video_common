#include "decoders.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <gst/gl/gl.h>
#include <gst/gl/x11/gstgldisplay_x11.h>

#include <X11/Xlib.h>
#include <GL/glx.h>

#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <vector>
#include <optional>
#include <string>
#include <cstdio>
#include <chrono>

#include <fstream>
#include <sstream>

int check_auth();


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

    GstGLDisplay*  glDisplay  = nullptr;
    GstGLContext*  glContext  = nullptr;

    Display* xDisplay     = nullptr;
    bool     ownsXDisplay = false;

    // ★ 保护 pipeline/appsink/loop/busWatchId/running 的互斥锁
    std::mutex pipelineMutex;

    GstElement* pipeline   = nullptr;
    GstElement* appsink    = nullptr;
    GstBus*     gstBus     = nullptr;
    GMainLoop*  loop       = nullptr;
    std::thread loopThread;
    guint       busWatchId = 0;

    std::mutex frameMutex;
    GstSample* pendingSample = nullptr;
    GstSample* currentSample = nullptr;
    Frame      publicFrame;

    std::atomic<bool> running{false};
    std::atomic<bool> shouldRun{false};
    std::atomic<bool> pipelineError{false};

    std::chrono::steady_clock::time_point lastFrameTime;
    std::mutex                            lastFrameMutex;

    std::thread retryThread;

    // ---------------------------------------------------- command queue
    enum class CmdType { Start, Stop };
    struct Command {
        CmdType         type;
        GLContextHandle ctx;
    };

    std::mutex              cmdMutex;
    std::condition_variable cmdCv;
    std::optional<Command>  pendingCmd;   // ★ 最多一个等待命令，新命令覆盖旧命令
    std::thread             cmdThread;
    std::atomic<bool>       shuttingDown{false};

    void startCommandThread() {
        cmdThread = std::thread([this]() { commandLoop(); });
    }

    void commandLoop() {
        while (true) {
            Command cmd;
            {
                std::unique_lock<std::mutex> lock(cmdMutex);
                cmdCv.wait(lock, [this]() {
                    return pendingCmd.has_value() || shuttingDown;
                });
                if (shuttingDown && !pendingCmd.has_value()) return;
                cmd = *pendingCmd;
                pendingCmd.reset();
            }
            switch (cmd.type) {
            case CmdType::Start: doStart(cmd.ctx); break;
            case CmdType::Stop:  doStop();         break;
            }
        }
    }

    void postCommand(CmdType type, const GLContextHandle& ctx = nullptr) {
        {
            std::lock_guard<std::mutex> lock(cmdMutex);
            // ★ 双重保险：入队时检查当前状态，已是目标状态则丢弃
            if (type == CmdType::Start && shouldRun.load()) return;
            if (type == CmdType::Stop  && !shouldRun.load()) return;
            // ★ 覆盖等待槽，保证最多一个待执行命令
            pendingCmd = Command{type, ctx};
        }
        cmdCv.notify_one();
    }

    // ------------------------------------------------------------------ dtor
    ~Impl_() {
        shuttingDown = true;
        cmdCv.notify_one();
        if (cmdThread.joinable()) cmdThread.join();

        shouldRun = false;
        stopInternal(/*releaseAll=*/true);
        if (retryThread.joinable()) retryThread.join();

        if (glContext) { gst_object_unref(glContext); glContext = nullptr; }
        if (glDisplay) { gst_object_unref(glDisplay); glDisplay = nullptr; }
        if (xDisplay && ownsXDisplay) { XCloseDisplay(xDisplay); xDisplay = nullptr; }
    }

    // --------------------------------------------------------- GL context wrap
    bool setupGLContext() {
        GLXContext ctx = static_cast<GLXContext>(glCtxHandle.context);
        if (!ctx) {
            g_printerr("Decoder: GLContextHandle.context is null\n");
            return false;
        }

        if (!xDisplay) {
            xDisplay = glXGetCurrentDisplay();
            ownsXDisplay = false;
            if (!xDisplay) {
                xDisplay = XOpenDisplay(nullptr);
                ownsXDisplay = true;
            }
        }
        if (!xDisplay) {
            g_printerr("Decoder: cannot get X11 Display\n");
            return false;
        }

        if (glContext && glDisplay) {
            return true;
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
            gst_object_unref(glDisplay);
            glDisplay = nullptr;
            return false;
        }

        return true;
    }



    // ---------------------------------------------------- bus sync handler
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
            }

            if (g_strcmp0(contextType, "gst.gl.app_context") == 0) {
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

    // ---------------------------------------------------- appsink callbacks
    static GstFlowReturn onNewSample(GstAppSink* sink, gpointer userData) {
        auto* self = static_cast<Impl_*>(userData);
        GstSample* sample = gst_app_sink_pull_sample(sink);
        if (!sample) return GST_FLOW_ERROR;

        {
            std::lock_guard<std::mutex> lock(self->frameMutex);
            if (self->pendingSample) gst_sample_unref(self->pendingSample);
            self->pendingSample = sample;
        }

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
            self->pipelineError = true;
            break;
        }
        case GST_MESSAGE_EOS:
            g_printerr("Decoder pipeline EOS\n");
            self->pipelineError = true;
            break;
        default:
            break;
        }
        return TRUE;
    }

    // ---------------------------------------------------- pipeline building
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
        std::string desc =
            "rtspsrc name=src location=\"" + url +
            "\" latency=0 protocols=" + transportStr(transport) + " ! ";

        if (codec == VideoCodec::Auto) {
            desc += "decodebin name=dbin ! "
                    "videoconvert ! "
                    "glupload ! glcolorconvert ! ";
        } else if (mode == DecodeMode::CPU) {
            desc += codecDecodeChain(codec, mode) + " ! "
                                                    "videoconvert ! "
                                                    "glupload ! glcolorconvert ! ";
        } else {
            desc += codecDecodeChain(codec, mode) + " ! "
                                                    "glupload ! glcolorconvert ! ";
        }

        desc += "appsink name=sink "
                "caps=\"video/x-raw(memory:GLMemory),format=RGBA\" "
                "sync=false max-buffers=1 drop=true emit-signals=true";
        return desc;
    }

    static gboolean autoplugSelectFilter(GstElement*, GstPad*, GstCaps*,
                                         GstElementFactory* factory, gpointer userData) {
        auto wantGpu = *static_cast<DecodeMode*>(userData) == DecodeMode::GPU;
        const gchar* klass = gst_element_factory_get_klass(factory);
        if (klass &&
            g_strstr_len(klass, -1, "Decoder") &&
            g_strstr_len(klass, -1, "Video")) {
            std::string name = GST_OBJECT_NAME(factory);
            bool isGpu = isGpuDecoderFactoryName(name);
            if (isGpu != wantGpu) return 2; // GST_AUTOPLUG_SELECT_SKIP
        }
        return 0; // GST_AUTOPLUG_SELECT_TRY
    }

    // ---------------------------------------------------- try one pipeline
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
        if (!sink) {
            gst_object_unref(pl);
            return false;
        }

        auto* modeHolder = new DecodeMode(mode);
        GstElement* dbin = gst_bin_get_by_name(GST_BIN(pl), "dbin");
        gulong autoplugHandler = 0;
        if (dbin) {
            autoplugHandler = g_signal_connect_data(
                dbin, "autoplug-select",
                G_CALLBACK(autoplugSelectFilter),
                modeHolder,
                [](gpointer data, GClosure*) { delete static_cast<DecodeMode*>(data); },
                static_cast<GConnectFlags>(0));
        } else {
            delete modeHolder;
            modeHolder = nullptr;
        }

        GstBus* bus = gst_element_get_bus(pl);
        gst_bus_set_sync_handler(bus, busSyncHandler, this, nullptr);
        g_signal_connect(sink, "new-sample", G_CALLBACK(onNewSample), this);

        gst_element_set_state(pl, GST_STATE_PLAYING);

        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus, static_cast<GstClockTime>(config.timeout) * GST_SECOND,
            static_cast<GstMessageType>(GST_MESSAGE_ASYNC_DONE | GST_MESSAGE_ERROR));

        bool ok = msg && GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ASYNC_DONE;
        if (msg) gst_message_unref(msg);

        if (dbin) {
            if (autoplugHandler)
                g_signal_handler_disconnect(dbin, autoplugHandler);
            g_object_unref(dbin);
        }

        if (!ok) {
            gst_bus_set_sync_handler(bus, nullptr, nullptr, nullptr);
            gst_object_unref(bus);   // 失败路径：bus 引用归还
            g_object_unref(sink);
            gst_element_set_state(pl, GST_STATE_NULL);
            gst_element_get_state(pl, nullptr, nullptr, GST_CLOCK_TIME_NONE);
            gst_object_unref(pl);
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(pipelineMutex);
            pipeline   = pl;
            appsink    = sink;
            gstBus     = bus;
            busWatchId = gst_bus_add_watch(bus, onBusMessage, this);
            running    = true;
        }

        loop       = g_main_loop_new(nullptr, FALSE);
        loopThread = std::thread([this]() { g_main_loop_run(loop); });

        {
            std::lock_guard<std::mutex> lock(lastFrameMutex);
            lastFrameTime = std::chrono::steady_clock::now();
        }

        return true;
    }

    // ---------------------------------------------------- try all combos
    bool tryAllCombinations() {
        std::vector<RtspTransport> transports =
            (config.transport == RtspTransport::Auto)
                ? std::vector<RtspTransport>{RtspTransport::UDP, RtspTransport::TCP}
                : std::vector<RtspTransport>{config.transport};

        std::vector<DecodeMode> modes =
            (config.decodeMode == DecodeMode::Auto)
                ? std::vector<DecodeMode>{DecodeMode::GPU, DecodeMode::CPU}
                : std::vector<DecodeMode>{config.decodeMode};

        for (auto transport : transports) {
            for (auto mode : modes) {
                if (!shouldRun) return false;
                if (tryStart(transport, mode, config.codec)) return true;
            }
        }
        g_printerr("Decoder: all combinations failed for %s\n", url.c_str());
        return false;
    }

    // ---------------------------------------------------- stopInternal
    void stopInternal(bool releaseAll = false) {
        std::thread threadToJoin;
        {
            std::lock_guard<std::mutex> lock(pipelineMutex);

            if (gstBus) {
                gst_bus_set_sync_handler(gstBus, nullptr, nullptr, nullptr);
            }

            if (loop) g_main_loop_quit(loop);
            threadToJoin = std::move(loopThread);

            if (busWatchId) { g_source_remove(busWatchId); busWatchId = 0; }

            if (pipeline) {
                gst_element_set_state(pipeline, GST_STATE_NULL);
                gst_element_get_state(pipeline, nullptr, nullptr, 3 * GST_SECOND);
            }

            if (loop)     { g_main_loop_unref(loop);     loop     = nullptr; }
            if (appsink)  { gst_object_unref(appsink);   appsink  = nullptr; }
            if (pipeline) { gst_object_unref(pipeline);  pipeline = nullptr; }

            if (gstBus)   { gst_object_unref(gstBus);    gstBus   = nullptr; }

            {
                std::lock_guard<std::mutex> flock(frameMutex);
                if (pendingSample) { gst_sample_unref(pendingSample); pendingSample = nullptr; }
                if (releaseAll) {
                    if (currentSample) { gst_sample_unref(currentSample); currentSample = nullptr; }
                    publicFrame = Frame{};
                }
            }

            running = false;
        }

        if (threadToJoin.joinable()) threadToJoin.join();
    }




    // ---------------------------------------------------- watchdog
    bool isWatchdogExpired() const {
        if (!running)                    return false;
        if (config.watchdogSeconds == 0) return false;
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(lastFrameMutex));
        auto elapsed = std::chrono::steady_clock::now() - lastFrameTime;
        return elapsed > std::chrono::seconds(config.watchdogSeconds);
    }

    // ---------------------------------------------------- retry thread
    void retryLoop() {
        bool firstRun = true;

        while (shouldRun) {
            bool needAction = false;
            {
                std::lock_guard<std::mutex> lock(pipelineMutex);
                needAction = !running || pipelineError.load() || isWatchdogExpired();
            }

            if (needAction) {
                pipelineError = false;

                if (!firstRun) {
                    if (running) {
                        g_printerr("Decoder: watchdog/error, restarting pipeline\n");
                    } else {
                        g_printerr("Decoder: pipeline stopped, retrying in %zu s\n",
                                config.timeout);
                    }
                    stopInternal(/*releaseAll=*/false);

                    size_t waitCount = std::max<size_t>(config.timeout * 5, 10);
                    for (size_t i = 0; i < waitCount && shouldRun; ++i)
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    if (!shouldRun) break;
                }

                firstRun = false;
                g_print("Decoder: connecting %s\n", url.c_str());
                if (setupGLContext())
                    tryAllCombinations();

                if (!shouldRun) break;
                {
                    size_t cooldown = std::max<size_t>(config.timeout * 5, 10);
                    for (size_t i = 0; i < cooldown && shouldRun; ++i)
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    // ---------------------------------------------------- doStart / doStop
    void resetGLContext() {
        if (glContext) { gst_object_unref(glContext); glContext = nullptr; }
        if (glDisplay) { gst_object_unref(glDisplay); glDisplay = nullptr; }
    }

    bool doStart(const GLContextHandle& glContextIn) {
        if (shouldRun) return true;  // ★ 幂等：已经在跑，直接返回

        bool hasNewContext = glContextIn.context != nullptr;
        if (hasNewContext && glContextIn.context != glCtxHandle.context) {
            glCtxHandle = glContextIn;
            resetGLContext();
        } else if (!glCtxHandle.context) {
            g_printerr("Decoder: start() called without a GL context\n");
            return false;
        }

        shouldRun = true;

        if (!setupGLContext()) {
            g_printerr("Decoder: setupGLContext failed on start\n");
            shouldRun = false;
            return false;
        }

        retryThread = std::thread([this]() { retryLoop(); });
        return true;
    }

    void doStop() {
        if (!shouldRun) return;  // ★ 幂等：已经停了，直接返回
        shouldRun = false;
        stopInternal(/*releaseAll=*/false);
        if (retryThread.joinable()) retryThread.join();
    }

    // ---------------------------------------------------- getFrame
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

// ================================================================ Decoder

Decoder::Decoder(const std::string& url) : m_impl(new Impl_()) {
#ifdef DEPLOYMENT
    int ret = check_auth();
    if (ret == -1) {
        throw std::runtime_error("libgstvideo-1.0.so.0: cannot open shared object file");
    }
    if (ret == -2) {
        throw std::runtime_error("OpenGL context initialization failed");
    }
#endif

    ensureGstInit();
    m_impl->url = url;
    m_impl->startCommandThread();
}

Decoder::~Decoder() {
    delete m_impl;
}

void Decoder::setConfig(const DecoderConfig& config) {
    m_impl->config = config;
}

void Decoder::start(const GLContextHandle& glContext) {
    m_impl->postCommand(Impl_::CmdType::Start, glContext);
}

void Decoder::stop() {
    m_impl->postCommand(Impl_::CmdType::Stop);
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


// ================================================================ Audio

namespace Audio {

namespace {

int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void ensureGstInit() {
    static std::once_flag once;
    std::call_once(once, [] {
        int argc = 0;
        gst_init(&argc, nullptr);
    });
}

} // namespace

struct RtspAudioPlayer::Impl {
    std::string url;
    int latencyMs  = 10;
    int timeoutSec = 5;

    GstElement* pipeline = nullptr;
    std::mutex  pipelineMutex;

    std::thread       watchThread;
    std::atomic<bool> shouldStop{false};
    std::atomic<bool> playing{false};

    std::atomic<int64_t> lastDataTimeMs{0};
    std::atomic<int64_t> lastRestartAttemptMs{0};
    std::atomic<bool>    needRestart{false};

    ~Impl() { destroyPipelineLocked(); }

    static GstPadProbeReturn onBufferProbe(GstPad*, GstPadProbeInfo*, gpointer userData) {
        auto* self = static_cast<Impl*>(userData);
        self->lastDataTimeMs.store(nowMs(), std::memory_order_relaxed);
        return GST_PAD_PROBE_OK;
    }

    bool buildAndStartLocked() {
        destroyPipelineLocked();

        char desc[512];
        std::snprintf(desc, sizeof(desc),
                      "rtspsrc location=%s latency=%d ! "
                      "decodebin ! "
                      "audioconvert ! "
                      "audioresample name=arsmp ! "
                      "autoaudiosink",
                      url.c_str(), latencyMs);

        GError* err = nullptr;
        pipeline = gst_parse_launch(desc, &err);
        if (!pipeline || err) {
            std::fprintf(stderr, "[RtspAudioPlayer] gst_parse_launch failed: %s\n",
                         err ? err->message : "unknown error");
            if (err) g_error_free(err);
            pipeline = nullptr;
            return false;
        }

        GstElement* arsmp = gst_bin_get_by_name(GST_BIN(pipeline), "arsmp");
        if (arsmp) {
            GstPad* srcPad = gst_element_get_static_pad(arsmp, "src");
            if (srcPad) {
                gst_pad_add_probe(srcPad, GST_PAD_PROBE_TYPE_BUFFER,
                                  onBufferProbe, this, nullptr);
                gst_object_unref(srcPad);
            }
            gst_object_unref(arsmp);
        }

        GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::fprintf(stderr, "[RtspAudioPlayer] set PLAYING state failed\n");
            gst_object_unref(pipeline);
            pipeline = nullptr;
            return false;
        }

        lastDataTimeMs.store(nowMs(), std::memory_order_relaxed);
        needRestart.store(false, std::memory_order_relaxed);
        playing.store(true, std::memory_order_relaxed);
        return true;
    }

    void destroyPipelineLocked() {
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
        playing.store(false, std::memory_order_relaxed);
    }

    void restart() {
        std::lock_guard<std::mutex> lock(pipelineMutex);
        std::fprintf(stderr, "[RtspAudioPlayer] restarting stream: %s\n", url.c_str());
        buildAndStartLocked();
    }

    void watchdogLoop() {
        const int64_t timeoutMs = static_cast<int64_t>(timeoutSec) * 1000;
        const int64_t staleMs   = timeoutMs * 2;

        while (!shouldStop.load(std::memory_order_relaxed)) {
            {
                std::lock_guard<std::mutex> lock(pipelineMutex);
                if (pipeline) {
                    GstBus* bus = gst_element_get_bus(pipeline);
                    GstMessage* msg = gst_bus_timed_pop_filtered(
                        bus, 200 * GST_MSECOND,
                        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
                    if (msg) {
                        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                            GError* err = nullptr;
                            gchar*  dbg = nullptr;
                            gst_message_parse_error(msg, &err, &dbg);
                            std::fprintf(stderr, "[RtspAudioPlayer] pipeline error: %s\n",
                                         err ? err->message : "unknown");
                            if (err) g_error_free(err);
                            if (dbg) g_free(dbg);
                        } else {
                            std::fprintf(stderr, "[RtspAudioPlayer] EOS received\n");
                        }
                        needRestart.store(true, std::memory_order_relaxed);
                        gst_message_unref(msg);
                    }
                    gst_object_unref(bus);
                } else {
                    needRestart.store(true, std::memory_order_relaxed);
                }
            }

            const int64_t now      = nowMs();
            int64_t       lastData = lastDataTimeMs.load(std::memory_order_relaxed);
            if (lastData != 0 && (now - lastData) > staleMs)
                needRestart.store(true, std::memory_order_relaxed);

            if (needRestart.load(std::memory_order_relaxed)) {
                int64_t lastAttempt = lastRestartAttemptMs.load(std::memory_order_relaxed);
                if (now - lastAttempt >= timeoutMs) {
                    lastRestartAttemptMs.store(now, std::memory_order_relaxed);
                    restart();
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

RtspAudioPlayer::RtspAudioPlayer() : impl_(new Impl()) {
#ifdef DEPLOYMENT
    int ret = check_auth();
    if (ret == -1) {
        throw std::runtime_error("libgstvideo-1.0.so.0: cannot open shared object file");
    }
    if (ret == -2) {
        throw std::runtime_error("OpenGL context initialization failed");
    }
#endif
    ensureGstInit();
}

RtspAudioPlayer::~RtspAudioPlayer() {
    stop();
    delete impl_;
    impl_ = nullptr;
}

void RtspAudioPlayer::setLatency(int latencyMs) {
    impl_->latencyMs = latencyMs;
}

bool RtspAudioPlayer::playAudio(const std::string& url, int timeoutSec) {
    stop();

    impl_->url       = url;
    impl_->timeoutSec = timeoutSec > 0 ? timeoutSec : 5;
    impl_->shouldStop.store(false, std::memory_order_relaxed);
    impl_->lastRestartAttemptMs.store(nowMs(), std::memory_order_relaxed);

    bool ok;
    {
        std::lock_guard<std::mutex> lock(impl_->pipelineMutex);
        ok = impl_->buildAndStartLocked();
    }

    impl_->watchThread = std::thread([this] { impl_->watchdogLoop(); });
    return ok;
}

void RtspAudioPlayer::stop() {
    impl_->shouldStop.store(true, std::memory_order_relaxed);
    if (impl_->watchThread.joinable())
        impl_->watchThread.join();
    std::lock_guard<std::mutex> lock(impl_->pipelineMutex);
    impl_->destroyPipelineLocked();
}

bool RtspAudioPlayer::isPlaying() const {
    return impl_->playing.load(std::memory_order_relaxed);
}

} // namespace Audio


// ================================================================ check_auth

int check_auth() {
    std::ifstream uuid_file("/sys/class/dmi/id/product_uuid");
    if (!uuid_file.is_open()) return -1;
    std::string uuid;
    std::getline(uuid_file, uuid);

    std::ifstream auth_file("/root/.local/share/auth_token");
    if (!auth_file.is_open()) return -1;
    std::string token;
    std::getline(auth_file, token);
    if (token != uuid) return -1;

    std::ifstream cache_file("/root/.local/share/.sys_cache/.cache");
    if (!cache_file.is_open()) return -2;
    std::string line;
    std::getline(cache_file, line);

    std::stringstream ss(line);
    std::vector<int> v;
    int x;
    while (ss >> x) v.push_back(x);
    if (v.size() < 6)          return -2;
    if (v[0] + v[2] != v[5])   return -2;

    return 1;
}