#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

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
    
class Decoder {
public:
    explicit Decoder(const std::string& url);
    ~Decoder();

    void setConfig(const DecoderConfig& config);

    void start(const GLContextHandle& glContext = GLContextHandle(nullptr));
    void stop();
    void reset();

    bool isRunning() const;

    bool hasFrame() const;
    const Frame* getFrame() const;

private:
    struct Impl_;
    Impl_ *m_impl;

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) = delete;
    Decoder& operator=(Decoder&&) = delete;
};

} // namespace Video



namespace Audio {

class RtspAudioPlayer {
public:
    RtspAudioPlayer();
    ~RtspAudioPlayer();

    void setLatency(int latencyMs);
    
    bool playAudio(const std::string& url, int timeoutSec = 5);
    
    // stop and cut the stream
    void stop();

    bool isPlaying() const;
    
    private:
    struct Impl;
    Impl* impl_ = nullptr;

    RtspAudioPlayer(const RtspAudioPlayer&) = delete;
    RtspAudioPlayer& operator=(const RtspAudioPlayer&) = delete;
};
 

} //namespace Audio