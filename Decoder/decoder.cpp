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

namespace Video {

namespace {

void ensureGstInit() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        int argc = 0;
        gst_init(&argc, nullptr);
    });
}

// 是否是我们认为的 "GPU 解码器" element（按 factory 名字粗略判断，NVIDIA 优先）。
bool isGpuDecoderFactoryName(const std::string& name) {
    return name.find("nv") != std::string::npos ||      // nvh264dec / nvh265dec / nvjpegdec / nvdec
           name.find("vaapi") != std::string::npos ||    // vaapih264dec 等（Intel/AMD）
           name.find("cuda") != std::string::npos;
}

} // namespace

struct Decoder::Impl_ {
    std::string url;
    GLContextHandle glCtxHandle; // 构造时传入的目标 GLXContext（只存 context，不存 display）
    DecoderConfig config;

    GstGLDisplay* glDisplay = nullptr;
    GstGLContext* glContext = nullptr; // 包装 glCtxHandle.context 得到的 wrapped context

    Display* xDisplay = nullptr;
    bool ownsXDisplay = false; // 如果是我们自己 XOpenDisplay 出来的，析构时要关掉

    GstElement* pipeline = nullptr;
    GstElement* appsink = nullptr;
    GMainLoop* loop = nullptr;
    std::thread loopThread;
    guint busWatchId = 0;

    std::mutex frameMutex;
    GstSample* pendingSample = nullptr; // 最新到达、还没被 getFrame 消费的
    GstSample* currentSample = nullptr; // getFrame 正在持有/暴露给外部的

    Frame publicFrame;
    std::atomic<bool> running{false};

    ~Impl_() {
        teardownPipeline();
        if (glContext) gst_object_unref(glContext);
        if (glDisplay) gst_object_unref(glDisplay);
        if (xDisplay && ownsXDisplay) XCloseDisplay(xDisplay);
    }

    // ---------- 共享 GL context ----------

    // 用构造时传入的 GLContextHandle.context（GLXContext）去建一个共享它的
    // wrapped GstGLContext，从而让 gst-gl 产出的纹理落在同一个 sharegroup
    // 里，实现零拷贝。Display 本身没有随 handle 传进来，这里优先用
    // glXGetCurrentDisplay()（要求调用线程此刻已经 makeCurrent 过某个
    // context，通常就是同一个 X 连接），拿不到就退回默认 XOpenDisplay(nullptr)
    // （单 X server / 单 :DISPLAY 环境下，比如你现在这种 Xorg 三屏拼接场景）。
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
            g_printerr("Decoder: 无法获取 X11 Display（既没有 current GLX context，"
                        "XOpenDisplay(nullptr) 也失败了，检查 $DISPLAY）\n");
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
            g_printerr("Decoder: failed to wrap GLXContext from GLContextHandle\n");
            return false;
        }

        // 必须在 start() 调用线程上、且这个 GLXContext 已经是 current 的
        // 情况下做这一步（要求 Qt 侧在调用 decoder->start() 之前，用同一个
        // context 先 makeCurrent() 过）。
        GError* err = nullptr;
        if (!gst_gl_context_activate(glContext, TRUE)) {
            g_printerr("Decoder: gst_gl_context_activate failed\n");
            return false;
        }
        if (!gst_gl_context_fill_info(glContext, &err)) {
            g_printerr("Decoder: gst_gl_context_fill_info failed: %s\n",
                        err ? err->message : "unknown error");
            if (err) g_error_free(err);
            gst_gl_context_activate(glContext, FALSE);
            return false;
        }
        gst_gl_context_activate(glContext, FALSE);

        return true;
    }

    // 响应 gst-gl 元素在 pipeline 里发出的 NEED_CONTEXT 查询，把我们包装好的
    // GstGLDisplay / GstGLContext 塞回去，这样 glupload/glcolorconvert/appsink
    // 才会在我们指定的 sharegroup 里创建纹理，而不是自己另开一个 context。
    static GstBusSyncReply busSyncHandler(GstBus* /*bus*/, GstMessage* msg, gpointer userData) {
        auto* self = static_cast<Impl_*>(userData);

        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_NEED_CONTEXT) {
            const gchar* contextType = nullptr;
            gst_message_parse_context_type(msg, &contextType);

            if (g_strcmp0(contextType, GST_GL_DISPLAY_CONTEXT_TYPE) == 0) {
                GstContext* context = gst_context_new(GST_GL_DISPLAY_CONTEXT_TYPE, TRUE);
                gst_context_set_gl_display(context, self->glDisplay);
                gst_element_set_context(GST_ELEMENT(GST_MESSAGE_SRC(msg)), context);
                gst_context_unref(context);
                return GST_BUS_DROP;
            } else if (g_strcmp0(contextType, "gst.gl.app_context") == 0) {
                GstContext* context = gst_context_new("gst.gl.app_context", TRUE);
                GstStructure* s = gst_context_writable_structure(context);
                gst_structure_set(s, "context", GST_TYPE_GL_CONTEXT, self->glContext, nullptr);
                gst_element_set_context(GST_ELEMENT(GST_MESSAGE_SRC(msg)), context);
                gst_context_unref(context);
                return GST_BUS_DROP;
            }
        }
        return GST_BUS_PASS;
    }

    // ---------- appsink 回调 ----------

    static GstFlowReturn onNewSample(GstAppSink* sink, gpointer userData) {
        auto* self = static_cast<Impl_*>(userData);
        GstSample* sample = gst_app_sink_pull_sample(sink);
        if (!sample) return GST_FLOW_ERROR;

        std::lock_guard<std::mutex> lock(self->frameMutex);
        if (self->pendingSample) gst_sample_unref(self->pendingSample);
        self->pendingSample = sample; // 转移所有权
        return GST_FLOW_OK;
    }

    static gboolean onBusMessage(GstBus* /*bus*/, GstMessage* msg, gpointer userData) {
        auto* self = static_cast<Impl_*>(userData);
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError* err = nullptr;
                gchar* dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);
                g_printerr("Decoder pipeline error: %s (%s)\n",
                           err ? err->message : "?", dbg ? dbg : "");
                if (err) g_error_free(err);
                if (dbg) g_free(dbg);
                self->running = false;
                break;
            }
            case GST_MESSAGE_EOS:
                self->running = false;
                break;
            default:
                break;
        }
        return TRUE; // 继续接收后续消息
    }

    // ---------- pipeline 构建 ----------

    static const char* transportStr(RtspTransport t) {
        return t == RtspTransport::TCP ? "tcp" : "udp";
    }

    // 生成 depay!parse!decode 片段。decodeMode 只能是 CPU 或 GPU（不接受 Auto）。
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
        std::string desc = "rtspsrc name=src location=\"" + url + "\" latency=0 protocols=" +
                            transportStr(transport) + " ! ";

        if (codec == VideoCodec::Auto) {
            // decodebin 自动探测编码格式；GPU/CPU 偏好通过 autoplug-select 信号过滤，
            // 在 tryStart 里对拿到的 decodebin element 单独连接。
            desc += "decodebin name=dbin ! ";
        } else {
            desc += codecDecodeChain(codec, mode) + " ! ";
        }

        desc += "glupload ! glcolorconvert ! "
                "appsink name=sink caps=\"video/x-raw(memory:GLMemory),format=RGBA\" "
                "sync=false max-buffers=1 drop=true emit-signals=true";
        return desc;
    }

    static gboolean autoplugSelectFilter(GstElement* /*bin*/, GstPad* /*pad*/,
                                          GstCaps* /*caps*/, GstElementFactory* factory,
                                          gpointer userData) {
        auto wantGpu = *static_cast<DecodeMode*>(userData) == DecodeMode::GPU;
        const gchar* klass = gst_element_factory_get_klass(factory);
        if (klass && g_strstr_len(klass, -1, "Decoder") && g_strstr_len(klass, -1, "Video")) {
            std::string name = GST_OBJECT_NAME(factory);
            bool isGpu = isGpuDecoderFactoryName(name);
            if (isGpu != wantGpu) {
                return /* GST_AUTOPLUG_SELECT_SKIP */ 2;
            }
        }
        return /* GST_AUTOPLUG_SELECT_TRY */ 0;
    }

    // 尝试用给定组合起一次 pipeline，成功进入 PLAYING 返回 true；
    // 失败会自己清理掉这次创建的 pipeline，调用方可以换下一个组合重试。
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
        if (err) { g_error_free(err); err = nullptr; } // 非致命 warning

        GstElement* sink = gst_bin_get_by_name(GST_BIN(pl), "sink");
        DecodeMode modeHolder = mode; // autoplugSelectFilter 用
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

        // 等待要么 ASYNC_DONE（起流成功）要么 ERROR（起流失败），最多 5 秒。
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus, 5 * GST_SECOND,
            static_cast<GstMessageType>(GST_MESSAGE_ASYNC_DONE | GST_MESSAGE_ERROR));

        bool ok = msg && GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ASYNC_DONE;
        if (msg) gst_message_unref(msg);

        if (dbin) { g_object_unref(dbin); }

        if (!ok) {
            if (autoplugHandler && dbin) g_signal_handler_disconnect(dbin, autoplugHandler);
            gst_bus_set_sync_handler(bus, nullptr, nullptr, nullptr);
            gst_object_unref(bus);
            if (sink) g_object_unref(sink);
            gst_element_set_state(pl, GST_STATE_NULL);
            gst_object_unref(pl);
            return false;
        }

        // 成功，接管为正式 pipeline
        pipeline = pl;
        appsink = sink; // 引用计数已经通过 gst_bin_get_by_name 持有
        busWatchId = gst_bus_add_watch(bus, onBusMessage, this);
        gst_object_unref(bus);

        loop = g_main_loop_new(nullptr, FALSE);
        loopThread = std::thread([this]() { g_main_loop_run(loop); });

        running = true;
        return true;
    }

    bool start(const GLContextHandle& glContext) {
        if (running) return true;
        glCtxHandle = glContext;          
        if (!setupGLContext()) {
            return false;
        }

        std::vector<RtspTransport> transports = (config.transport == RtspTransport::Auto)
            ? std::vector<RtspTransport>{RtspTransport::UDP, RtspTransport::TCP}
            : std::vector<RtspTransport>{config.transport};

        std::vector<DecodeMode> modes = (config.decodeMode == DecodeMode::Auto)
            ? std::vector<DecodeMode>{DecodeMode::GPU, DecodeMode::CPU}
            : std::vector<DecodeMode>{config.decodeMode};

        for (auto transport : transports) {
            for (auto mode : modes) {
                if (tryStart(transport, mode, config.codec)) {
                    return true;
                }
            }
        }
        g_printerr("Decoder: all transport/decode combinations failed for %s\n", url.c_str());
        return false;
    }

    void teardownPipeline() {
        running = false;

        if (loop) {
            g_main_loop_quit(loop);
        }
        if (loopThread.joinable()) {
            loopThread.join();
        }
        if (loop) {
            g_main_loop_unref(loop);
            loop = nullptr;
        }
        if (busWatchId) {
            g_source_remove(busWatchId);
            busWatchId = 0;
        }
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
        }
        if (appsink) {
            gst_object_unref(appsink);
            appsink = nullptr;
        }
        if (pipeline) {
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }

        std::lock_guard<std::mutex> lock(frameMutex);
        if (pendingSample) { gst_sample_unref(pendingSample); pendingSample = nullptr; }
        if (currentSample) { gst_sample_unref(currentSample); currentSample = nullptr; }
        publicFrame = Frame{};
    }

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

        // 跨 context 使用前，等待生产者那边的 GL 命令真正执行完（同 sharegroup
        // 下纹理内容才有效），避免读到还没画完的纹理。
        GstGLSyncMeta* syncMeta = gst_buffer_get_gl_sync_meta(buffer);
        if (syncMeta) {
            gst_gl_sync_meta_wait(syncMeta, glContext);
        }

        publicFrame.texture = gst_gl_memory_get_texture_id(glMem);

        GstCaps* caps = gst_sample_get_caps(currentSample);
        GstVideoInfo info;
        if (caps && gst_video_info_from_caps(&info, caps)) {
            publicFrame.width = static_cast<uint32_t>(GST_VIDEO_INFO_WIDTH(&info));
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
    m_impl->teardownPipeline();
}

void Decoder::reset() {
    m_impl->teardownPipeline();
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