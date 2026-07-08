#pragma once
#include <cstdint>
#include <cstddef>
#include <GL/gl.h>

namespace Video {

enum class DecodeMode {
    Auto,
    GPU,
    CPU
};

enum class VideoCodec {
    Auto,
    H264,
    H265,
    MJPEG
};

enum class RtspTransport {
    Auto,
    UDP,
    TCP
};

struct DecoderConfig {
    DecodeMode   decodeMode = DecodeMode::Auto;
    VideoCodec   codec      = VideoCodec::Auto;
    RtspTransport transport = RtspTransport::Auto;
    size_t       timeout    = 5;
    size_t       watchdogSeconds = 10;
};

struct GLContextHandle {
    void* context = nullptr;
    GLContextHandle(void* ctx = nullptr) : context(ctx) {}
};

struct Frame {
    GLuint   texture = 0;
    uint32_t width   = 0;
    uint32_t height  = 0;
};

} // namespace Video
