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
#include <deque>
#include <string>
#include <cstdio>
#include <chrono>
#include <cstdio>


#include <string>
#include <fstream>
#include <sstream>
#include <vector>

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

    GstElement* pipeline   = nullptr;
    GstElement* appsink    = nullptr;
    GMainLoop*  loop       = nullptr;
    std::thread loopThread;
    guint       busWatchId = 0;

    std::mutex frameMutex;
    GstSample* pendingSample = nullptr;
    GstSample* currentSample = nullptr;
    Frame      publicFrame;

    std::atomic<bool> running{false};
    std::atomic<bool> shouldRun{false};

    std::chrono::steady_clock::time_point lastFrameTime;
    std::mutex                            lastFrameMutex;

    std::thread retryThread;

    // ---------------------------------------------------- command queue
    // 所有对外的控制操作（start/stop/reset）都被封装成 command，push 进队列后
    // 立即返回，真正的执行放在常驻的 cmdThread 里严格串行处理。
    // 这样可以保证：
    //   1) 调用方（通常是 UI 线程）永远是微秒级的入队操作，不会卡顿；
    //   2) 同一个 Decoder 的 start/stop/reset 不会因为各自开线程而产生竞态
    //      （比如 stop 抢在 start 前面跑完导致状态错乱）。
    enum class CmdType { Start, Stop, Reset };
    struct Command {
        CmdType         type;
        GLContextHandle ctx;
    };

    std::mutex              cmdMutex;
    std::condition_variable cmdCv;
    std::deque<Command>     cmdQueue;
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
                    return !cmdQueue.empty() || shuttingDown;
                });
                if (shuttingDown && cmdQueue.empty()) return;
                cmd = cmdQueue.front();
                cmdQueue.pop_front();
            }
            switch (cmd.type) {
            case CmdType::Start: doStart(cmd.ctx); break;
            case CmdType::Stop:  doStop();         break;
            case CmdType::Reset: doReset();        break;
            }
        }
    }

    void postCommand(CmdType type, const GLContextHandle& ctx = nullptr) {
        {
            std::lock_guard<std::mutex> lock(cmdMutex);
            cmdQueue.push_back({type, ctx});
        }
        cmdCv.notify_one();
    }

    // ------------------------------------------------------------------ dtor
    ~Impl_() {
        // 先停止接收新命令，等 command 线程处理完队列里剩余的命令并退出，
        // 保证退出前不会有“半路而废”的 start/stop 还挂在队列里。
        shuttingDown = true;
        cmdCv.notify_one();
        if (cmdThread.joinable()) cmdThread.join();

        // command 线程已退出，之后不会再有新命令进来，可以安全地直接彻底清理。
        shouldRun = false;
        stopInternal(/*releaseAll=*/true);
        if (retryThread.joinable()) retryThread.join();

        if (glContext) { gst_object_unref(glContext); glContext = nullptr; }
        if (glDisplay) { gst_object_unref(glDisplay); glDisplay = nullptr; }
        if (xDisplay && ownsXDisplay) { XCloseDisplay(xDisplay); xDisplay = nullptr; }
    }

    // --------------------------------------------------------- GL context wrap
    // 只包装，不 activate / fill_info（那两步要求 GL current，不能在后台线程调）
    // GStreamer 在 pipeline 启动时通过 busSyncHandler 拿到 context 自行初始化。
    bool setupGLContext() {
        GLXContext ctx = static_cast<GLXContext>(glCtxHandle.context);
        if (!ctx) {
            g_printerr("Decoder: GLContextHandle.context is null\n");
            return false;
        }

        // 优先复用已有的 X Display 连接，避免重复 XOpenDisplay
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

        // 释放旧的再重建（重试时保证干净）
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

        return true;
    }

    // ---------------------------------------------------- bus sync handler
    // 把 glDisplay / glContext 注入 pipeline 里的 GL 元素，
    // 使它们在同一个 sharegroup 里产出纹理。
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
            self->running = false;
            break;
        }
        case GST_MESSAGE_EOS:
            g_printerr("Decoder pipeline EOS\n");
            self->running = false;
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

        DecodeMode  modeHolder     = mode;
        GstElement* dbin           = gst_bin_get_by_name(GST_BIN(pl), "dbin");
        gulong      autoplugHandler = 0;
        if (dbin) {
            autoplugHandler = g_signal_connect(
                dbin, "autoplug-select",
                G_CALLBACK(autoplugSelectFilter), &modeHolder);
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
            if (!ok && autoplugHandler)
                g_signal_handler_disconnect(dbin, autoplugHandler);
            g_object_unref(dbin);
        }

        if (!ok) {
            gst_bus_set_sync_handler(bus, nullptr, nullptr, nullptr);
            gst_object_unref(bus);
            g_object_unref(sink);
            gst_element_set_state(pl, GST_STATE_NULL);
            gst_element_get_state(pl, nullptr, nullptr, GST_CLOCK_TIME_NONE); // 补上这一行
            gst_object_unref(pl);
            return false;
        }

        // 成功，接管 pipeline
        pipeline   = pl;
        appsink    = sink;
        busWatchId = gst_bus_add_watch(bus, onBusMessage, this);
        gst_object_unref(bus);

        loop       = g_main_loop_new(nullptr, FALSE);
        loopThread = std::thread([this]() { g_main_loop_run(loop); });

        running = true;

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

        // Auto：GPU 优先，CPU 保底
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

    void stopInternal(bool releaseAll = false) {
        // 不再在这里提前置 false

        if (loop) g_main_loop_quit(loop);
        if (loopThread.joinable()) loopThread.join();
        if (loop) { g_main_loop_unref(loop); loop = nullptr; }

        if (busWatchId) { g_source_remove(busWatchId); busWatchId = 0; }

        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            // 显式等待状态切换完全落地,而不是假设 set_state 一定同步返回完毕
            // (个别 GL sink/元素在极端情况下可能异步完成清理)
            gst_element_get_state(pipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
        }
        if (appsink)  { gst_object_unref(appsink);  appsink  = nullptr; }
        if (pipeline) { gst_object_unref(pipeline); pipeline = nullptr; }

        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (pendingSample) { gst_sample_unref(pendingSample); pendingSample = nullptr; }
            if (releaseAll) {
                if (currentSample) { gst_sample_unref(currentSample); currentSample = nullptr; }
                publicFrame = Frame{};
            }
        }

        // 移到最后:此时 GL 资源(pipeline/appsink 及其内部纹理)已经彻底释放,
        // isRunning() 从这一刻起返回 false 才是"可以安全把 context 交给别的 Decoder 用"的准确信号
        running = false;
    }

    // ---------------------------------------------------- watchdog
    bool isWatchdogExpired() const {
        if (!running)                        return false; // 已经死了，交给 !running 分支
        if (config.watchdogSeconds == 0)     return false;
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(lastFrameMutex));
        auto elapsed = std::chrono::steady_clock::now() - lastFrameTime;
        return elapsed > std::chrono::seconds(config.watchdogSeconds);
    }

    // ---------------------------------------------------- retry thread
    void retryLoop() {
        while (shouldRun) {
            if (!running || isWatchdogExpired()) {
                if (running) {
                    g_printerr("Decoder: watchdog expired, restarting pipeline\n");
                } else {
                    g_printerr("Decoder: pipeline stopped, retrying in %zu s\n",
                               config.timeout);
                }

                // 内部重试：软停止即可，紧接着会重建 pipeline
                stopInternal(/*releaseAll=*/false);

                // 等待 timeout 秒再重试
                for (size_t i = 0; i < config.timeout * 5 && shouldRun; ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                if (!shouldRun) break;

                g_print("Decoder: retrying %s\n", url.c_str());

                // 重新 wrap GL context（安全：不调 activate/fill_info）
                // 然后重建 pipeline，等于内部再 start(context) 一次
                if (setupGLContext()) {
                    tryAllCombinations();
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    // ---------------------------------------------------- doStart / doStop / doReset
    // 以下三个函数只应由 commandLoop() 在 cmdThread 里串行调用，
    // 不要在其它线程直接调用，否则会破坏“同一 decoder 操作全串行”的保证。

    // glContextIn 允许为空（GLContextHandle::context == nullptr）：
    //   - 如果之前 start 过并保存过 context，则复用之前保存的那份；
    //   - 如果从未 start 过且没有传入 context，则失败。
    // 如果 shouldRun 已经为 true（正在运行中，或正处于重试等待中），
    // 说明已经启动过，直接返回，不重复触发。
    bool doStart(const GLContextHandle& glContextIn) {
        if (shouldRun) {
            return true;
        }

        bool hasNewContext = glContextIn.context != nullptr;
        if (hasNewContext) {
            glCtxHandle = glContextIn;
        } else if (!glCtxHandle.context) {
            g_printerr("Decoder: start() called without a GL context and none was previously set\n");
            return false;
        }
        // else: 没传新 context，但之前已保存过一份，直接复用 glCtxHandle

        shouldRun = true;

        if (!setupGLContext()) {
            g_printerr("Decoder: setupGLContext failed on start\n");
            shouldRun = false;
            return false;
        }

        tryAllCombinations();   // 第一次尝试

        // 启动重试/看门狗线程
        retryThread = std::thread([this]() { retryLoop(); });
        return true;
    }

    // 软停止：断开网络连接（停止占用带宽），但保留最后一帧和 GL context，
    // 以便下一次 start 能最快重启。
    void doStop() {
        if (!shouldRun) return; // 已经停了，无需重复操作
        shouldRun = false;
        stopInternal(/*releaseAll=*/false);
        if (retryThread.joinable()) retryThread.join();
    }

    // 彻底重置：清空最后一帧 / publicFrame，仅保留 GL context 直到析构。
    void doReset() {
        shouldRun = false;
        stopInternal(/*releaseAll=*/true);
        if (retryThread.joinable()) retryThread.join();
    }

    // ---------------------------------------------------- getFrame
    // 唯一保留的同步/阻塞接口：纯内存操作（取指针 + 拷贝宽高），微秒级，
    // 不涉及网络或 GL 初始化，可以放心在 UI 线程里高频调用。
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

// 异步：立即返回，仅将 Start 命令入队。真正是否连上、是否有画面，
// 靠 isRunning() / hasFrame() 轮询判断。
void Decoder::start(const GLContextHandle& glContext) {
    m_impl->postCommand(Impl_::CmdType::Start, glContext);
}

// 异步：立即返回，仅将 Stop 命令入队。
// 软停止：断网、保留最后一帧和 GL context，便于下次 start 最快重启。
void Decoder::stop() {
    m_impl->postCommand(Impl_::CmdType::Stop);
}

// 异步：立即返回，仅将 Reset 命令入队。彻底清空最后一帧。
void Decoder::reset() {
    m_impl->postCommand(Impl_::CmdType::Reset);
}

bool Decoder::isRunning() const {
    return m_impl->running;
}

bool Decoder::hasFrame() const {
    return m_impl->hasFrame();
}

// 唯一保留的阻塞接口，纯内存操作，可放心高频调用
const Frame* Decoder::getFrame() const {
    return m_impl->getFrame();
}

} // namespace Video


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

}  // namespace

struct RtspAudioPlayer::Impl {
    std::string url;
    int latencyMs = 10;
    int timeoutSec = 5;

    GstElement* pipeline = nullptr;
    std::mutex pipelineMutex;

    std::thread watchThread;
    std::atomic<bool> shouldStop{false};
    std::atomic<bool> playing{false};

    // 最近一次收到音频 buffer 的时间戳（ms），0 表示还没收到过
    std::atomic<int64_t> lastDataTimeMs{0};
    // 最近一次尝试重启的时间戳
    std::atomic<int64_t> lastRestartAttemptMs{0};
    // 本次 pipeline 是否已经出现过 ERROR/EOS，需要重启
    std::atomic<bool> needRestart{false};

    ~Impl() { destroyPipelineLocked(); }

    // pad probe：每次有音频数据流过就刷新时间戳
    static GstPadProbeReturn onBufferProbe(GstPad* /*pad*/,
                                            GstPadProbeInfo* /*info*/,
                                            gpointer userData) {
        auto* self = static_cast<Impl*>(userData);
        self->lastDataTimeMs.store(nowMs(), std::memory_order_relaxed);
        return GST_PAD_PROBE_OK;
    }

    // 构建并启动一个新的 pipeline，调用前需持有 pipelineMutex
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

        // 挂 probe 监控数据流动情况
        GstElement* arsmp = gst_bin_get_by_name(GST_BIN(pipeline), "arsmp");
        if (arsmp) {
            GstPad* srcPad = gst_element_get_static_pad(arsmp, "src");
            if (srcPad) {
                gst_pad_add_probe(srcPad, GST_PAD_PROBE_TYPE_BUFFER, onBufferProbe,
                                   this, nullptr);
                gst_object_unref(srcPad);
            }
            gst_object_unref(arsmp);
        }

        GstStateChangeReturn ret =
            gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::fprintf(stderr, "[RtspAudioPlayer] set PLAYING state failed\n");
            gst_object_unref(pipeline);
            pipeline = nullptr;
            return false;
        }

        // 重置计时，给新流一点建立连接的时间
        lastDataTimeMs.store(nowMs(), std::memory_order_relaxed);
        needRestart.store(false, std::memory_order_relaxed);
        playing.store(true, std::memory_order_relaxed);
        return true;
    }

    // 调用前需持有 pipelineMutex
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
        std::fprintf(stderr, "[RtspAudioPlayer] restarting stream: %s\n",
                     url.c_str());
        buildAndStartLocked();
    }

    void watchdogLoop() {
        const int64_t timeoutMs = static_cast<int64_t>(timeoutSec) * 1000;
        const int64_t staleMs = timeoutMs * 2;

        while (!shouldStop.load(std::memory_order_relaxed)) {
            {
                std::lock_guard<std::mutex> lock(pipelineMutex);
                if (pipeline) {
                    GstBus* bus = gst_element_get_bus(pipeline);
                    GstMessage* msg = gst_bus_timed_pop_filtered(
                        bus, 200 * GST_MSECOND,
                        static_cast<GstMessageType>(GST_MESSAGE_ERROR |
                                                     GST_MESSAGE_EOS));
                    if (msg) {
                        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                            GError* err = nullptr;
                            gchar* dbg = nullptr;
                            gst_message_parse_error(msg, &err, &dbg);
                            std::fprintf(stderr,
                                         "[RtspAudioPlayer] pipeline error: %s\n",
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
                    // 还没有 pipeline（比如首次连接失败），也需要重启
                    needRestart.store(true, std::memory_order_relaxed);
                }
            }

            const int64_t now = nowMs();

            // 卡流检测：收到过数据，但太久没有更新
            int64_t lastData = lastDataTimeMs.load(std::memory_order_relaxed);
            if (lastData != 0 && (now - lastData) > staleMs) {
                needRestart.store(true, std::memory_order_relaxed);
            }

            if (needRestart.load(std::memory_order_relaxed)) {
                int64_t lastAttempt =
                    lastRestartAttemptMs.load(std::memory_order_relaxed);
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
    // 保证幂等：如果已经在播放，先彻底停掉再重新开始
    stop();

    impl_->url = url;
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
    if (impl_->watchThread.joinable()) {
        impl_->watchThread.join();
    }
    std::lock_guard<std::mutex> lock(impl_->pipelineMutex);
    impl_->destroyPipelineLocked();
}

bool RtspAudioPlayer::isPlaying() const {
    return impl_->playing.load(std::memory_order_relaxed);
}


} //namespace Audio 



int check_auth(){
    // 1. 读取主板 UUID
    std::ifstream uuid_file("/sys/class/dmi/id/product_uuid");
    if (!uuid_file.is_open())
        return -1;

    std::string uuid;
    std::getline(uuid_file, uuid);

    // 2. 检查 auth_token 是否匹配 UUID
    std::ifstream auth_file("/root/.local/share/auth_token");
    if (!auth_file.is_open())
        return -1;

    std::string token;
    std::getline(auth_file, token);

    if (token != uuid)
        return -1;

    // 3. 检查 cache 数字规律
    std::ifstream cache_file("/root/.local/share/.sys_cache/.cache");
    if (!cache_file.is_open())
        return -2;

    std::string line;
    std::getline(cache_file, line);

    std::stringstream ss(line);
    std::vector<int> v;
    int x;

    while (ss >> x)
        v.push_back(x);

    if (v.size() < 6)
        return -2;

    if (v[0] + v[2] != v[5])
        return -2;

    return 1; // 全部通过
}