#include <cstdint>
#include <GL/gl.h>

namespace Video {

enum class DecodeMode {
    Auto,
    CPU,
    GPU
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
    DecodeMode decodeMode = DecodeMode::Auto;
    VideoCodec codec = VideoCodec::Auto;
    RtspTransport transport = RtspTransport::Auto;
};

struct GLContextHandle {
    void* context = nullptr;
};


struct Frame {
    GLuint texture = 0;

    uint32_t width = 0;
    uint32_t height = 0;
};


} // namespace Video