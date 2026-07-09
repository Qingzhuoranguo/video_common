#pragma once
#include "decoder_common.h"

#include <string>

namespace Video {
    
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